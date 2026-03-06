import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_process import (
    TifSegDataset_15d_on_the_fly,
    TifSegDataset_15d_on_the_fly_4d,
)
from nets.nets_unetres import *
from nets.nets import UNet_MSHD_Heavy


EXPERT_SPECS = {
    "128128": ("images_15d", "masks", (128, 128)),
    "128256": ("images_15d_128256", "masks_128256", (128, 256)),
    "256128": ("images_15d_256128", "masks_256128", (256, 128)),
    "256256": ("images_15d_256256", "masks_256256", (256, 256)),
}

BOUNDARY_LOSS_WEIGHT = 2.5
DICE_LOSS_WEIGHT = 0.2


def parse_args():
    parser = argparse.ArgumentParser("Train expert model without stitch/validation.")
    parser.add_argument("--expert", type=str, default="256128", choices=list(EXPERT_SPECS.keys()))
    parser.add_argument("--epochs", type=int, default=147)
    parser.add_argument("--batch-128", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=np.random.randint(1000, 5000))
    parser.add_argument(
        "--model",
        type=str,
        default="uconvnextb",
        choices=["ures34", "ures50", "uconvnextb", "uconvnexts", "ueffb4", "ueffb5", "umshd"],
    )
    parser.add_argument("--save-dir", type=str, default="./pths_expert")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_size_for(res_key: str, base_128: int) -> int:
    if res_key == "128128":
        return base_128
    if res_key in ("128256", "256128"):
        return base_128 - 2
    return base_128 - 6


def build_model(args, n_channels: int, device: torch.device):
    if args.model == "ures50":
        model = UNetResNet50(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "ures34":
        model = UNetResNet34(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "uconvnextb":
        model = UNetConvNeXtBase(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "uconvnexts":
        model = UNetConvNeXtSmall(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "ueffb4":
        model = UNetEfficientNetB4(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "ueffb5":
        model = UNetEfficientNetB5(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    elif args.model == "umshd":
        model = UNet_MSHD_Heavy(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model.to(device)


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for out in outputs:
            if torch.is_tensor(out) and out.ndim == 4:
                return out
        for out in outputs:
            if torch.is_tensor(out):
                return out
    raise TypeError("Model output does not contain 4D logits tensor.")


def build_datasets(project_root: str):
    train_datasets = {}

    train_root = os.path.join(project_root, "data", "train")
    if not os.path.isdir(train_root):
        train_root = os.path.join(project_root, "train")

    for key, (img_dir_name, mask_dir_name, _hw) in EXPERT_SPECS.items():
        if key in ("128256", "256128"):
            train_dataset_cls = TifSegDataset_15d_on_the_fly_4d
        else:
            train_dataset_cls = TifSegDataset_15d_on_the_fly

        img_dir = os.path.join(train_root, img_dir_name)
        mask_dir = os.path.join(train_root, mask_dir_name)
        train_ds = train_dataset_cls(
            input_dir=img_dir,
            label_dir=mask_dir,
            boundary=True,
            band=15,
            stitch=False,
        )
        train_datasets[key] = train_ds

    return train_datasets


def soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    den = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels, weights in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        logits = extract_logits(model(inputs))
        loss_pixel = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss_pixel = loss_pixel * (1 + BOUNDARY_LOSS_WEIGHT * weights)
        loss_bce = loss_pixel.mean()
        loss_dice = soft_dice_loss_from_logits(logits, labels)
        loss = loss_bce + DICE_LOSS_WEIGHT * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
    return total_loss / max(1, len(loader.dataset))


def choose_train_bucket(epoch_idx, total_epochs, expert_key, rng):
    if epoch_idx < total_epochs // 2:
        keys = list(EXPERT_SPECS.keys())
        probs = [0.4 if k == expert_key else 0.2 for k in keys]
        return rng.choice(keys, p=probs)
    return expert_key


def safe_save_checkpoint(state, final_path: str, retries: int = 2):
    save_dir = os.path.dirname(final_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    last_err = "unknown error"

    for attempt in range(1, retries + 1):
        use_legacy = attempt > 1
        try:
            with open(tmp_path, "wb") as f:
                if use_legacy:
                    torch.save(state, f, _use_new_zipfile_serialization=False)
                else:
                    torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
            return True, ""
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(0.5 * attempt)
    return False, last_err


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.dirname(os.path.abspath(__file__))

    train_datasets = build_datasets(project_root)
    for k in EXPERT_SPECS:
        print(f"[dataset] {k}: train={len(train_datasets[k])}")

    if len(train_datasets[args.expert]) == 0:
        raise RuntimeError(f"Expert dataset {args.expert} has no train samples.")

    train_loaders = {}
    for k in EXPERT_SPECS:
        bs = batch_size_for(k, args.batch_128)
        train_loaders[k] = DataLoader(
            train_datasets[k],
            batch_size=bs,
            shuffle=True,
            drop_last=False,
        )
        print(f"[loader] train {k}: bs={bs}")

    model = build_model(args, n_channels=17, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"expert_{args.expert}_{args.model}_full.pth",
    )

    rng = np.random.RandomState(args.seed + 1234)
    last_train_loss = 0.0
    last_bucket = args.expert
    save_epoch = args.epochs - 2
    saved_ok = False
    saved_err = ""

    print(
        f"[config] expert={args.expert}, epochs={args.epochs}, "
        f"half_mix={args.epochs // 2}, model={args.model}"
    )

    for epoch in range(args.epochs):
        chosen_bucket = choose_train_bucket(epoch, args.epochs, args.expert, rng)
        loader = train_loaders[chosen_bucket]

        train_loss = train_one_epoch(model, loader, optimizer, device)
        last_train_loss = train_loss
        last_bucket = chosen_bucket
        log_msg = (
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_bucket={chosen_bucket} | train_loss={train_loss:.4f}"
        )

        if (epoch + 1) == save_epoch:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "expert": args.expert,
                "seed": args.seed,
                "train_loss": last_train_loss,
                "last_bucket": last_bucket,
            }
            ok, err = safe_save_checkpoint(state, save_path, retries=2)
            if ok:
                saved_ok = True
                log_msg += f" | epoch{save_epoch}_saved"
            else:
                saved_err = err
                log_msg += f" | epoch{save_epoch}_save_failed={err}"

        print(log_msg)

    if saved_ok:
        print(f"[done] saved_epoch={save_epoch}, save_path={save_path}")
    else:
        print(f"[done] save_failed_at_epoch={save_epoch}, err={saved_err}, save_path={save_path}")


if __name__ == "__main__":
    main()
