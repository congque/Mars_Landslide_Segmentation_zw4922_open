import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_process import (
    TifSegDataset_15d,
    TifSegDataset_15d_on_the_fly,
    TifSegDataset_15d_on_the_fly_4d,
)
from nets.nets_segformer import SegFormer
from nets.nets_unetres import UNetResNet50
from nets.nets import UNet_MSHD_Heavy


EXPERT_SPECS = {
    "128128": ("images_15d", "masks", (128, 128)),
    "128256": ("images_15d_128256", "masks_128256", (128, 256)),
    "256128": ("images_15d_256128", "masks_256128", (256, 128)),
    "256256": ("images_15d_256256", "masks_256256", (256, 256)),
}
RECT_VAL_TTA_EXPERTS = {"128256", "256128"}
RECT4_TTA_MODES = ("id", "flip_ud", "flip_lr", "rot180")


def parse_args():

    parser = argparse.ArgumentParser("Train expert model for one target resolution.")

    parser.add_argument("--expert", type=str, default="256128", choices=list(EXPERT_SPECS.keys()))

    parser.add_argument("--stitch", type=int, default=1, choices=list(range(1, 7)))

    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--val-every", type=int, default=10)

    parser.add_argument("--tta-n", type=int, default=8)

    parser.add_argument("--batch-128", type=int, default=16)

    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--threshold", type=float, default=0.35)

    parser.add_argument("--seed", type=int, default=np.random.randint(1000,2000))

    parser.add_argument("--model", type=str, default="ures50", choices=["ures50", "segformer", "umshd"])

    parser.add_argument("--segformer-decoder-dim", type=int, default=256)

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
        return max(1, base_128)
    if res_key in ("128256", "256128"):
        return max(1, base_128 // 2)
    return max(1, base_128 // 4)


def select_val_tta_modes(expert_key: str, default_tta_n: int):
    if expert_key in RECT_VAL_TTA_EXPERTS:
        return list(RECT4_TTA_MODES), "rect4"
    n = max(1, min(8, int(default_tta_n)))
    return [f"d4_{t}" for t in range(n)], f"d4_{n}"


def build_model(args, n_channels: int, device: torch.device):
    if args.model == "segformer":
        model = SegFormer(
            n_channels=n_channels,
            n_classes=1,
            decoder_dim=args.segformer_decoder_dim,
        )
    elif args.model == "ures50":
        model = UNetResNet50(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
        )
    elif args.model == "umshd":
        model = UNet_MSHD_Heavy(
            n_channels=n_channels,
            n_classes=1,
            base_channel=32,
        )
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


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def build_datasets(project_root: str, stitch: int):
    if stitch < 1 or stitch > 6:
        raise ValueError("stitch 必须在 [1, 6]，第0列固定保留且用于翻倍。")
    stitch = [stitch]
    train_datasets = {}
    val_datasets = {}
    val_keep_one = list(set(range(7)) - set(stitch))

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
            stitch=stitch,
        )
        val_ds = TifSegDataset_15d(
            input_dir=img_dir,
            label_dir=mask_dir,
            boundary=False,
            band=15,
            stitch=val_keep_one,
        )
        train_datasets[key] = train_ds
        val_datasets[key] = val_ds
        
    return train_datasets, val_datasets


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels, weights in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        logits = extract_logits(model(inputs))
        loss_pixel = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss_pixel = loss_pixel * (1 + 4 * weights)
        loss = loss_pixel.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
    return total_loss / max(1, len(loader.dataset))


def _d4_apply(x: torch.Tensor, t: int) -> torch.Tensor:
    if t < 4:
        return torch.rot90(x, k=t, dims=(2, 3))
    y = torch.flip(x, dims=[3])
    return torch.rot90(y, k=t - 4, dims=(2, 3))


def _d4_inverse(x: torch.Tensor, t: int) -> torch.Tensor:
    if t < 4:
        return torch.rot90(x, k=(4 - t) % 4, dims=(2, 3))
    k = t - 4
    y = torch.rot90(x, k=(4 - k) % 4, dims=(2, 3))
    return torch.flip(y, dims=[3])


def _parse_d4_mode(mode: str) -> int:
    if not mode.startswith("d4_"):
        raise ValueError(f"Unsupported TTA mode: {mode}")
    try:
        t = int(mode.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid D4 TTA mode: {mode}") from exc
    if t < 0 or t > 7:
        raise ValueError(f"D4 TTA index out of range [0,7]: {mode}")
    return t


def _tta_apply(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "id":
        return x
    if mode == "flip_ud":
        return torch.flip(x, dims=[2])
    if mode == "flip_lr":
        return torch.flip(x, dims=[3])
    if mode == "rot180":
        return torch.rot90(x, k=2, dims=(2, 3))
    if mode.startswith("d4_"):
        return _d4_apply(x, _parse_d4_mode(mode))
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _tta_inverse(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "id":
        return x
    if mode == "flip_ud":
        return torch.flip(x, dims=[2])
    if mode == "flip_lr":
        return torch.flip(x, dims=[3])
    if mode == "rot180":
        return torch.rot90(x, k=2, dims=(2, 3))
    if mode.startswith("d4_"):
        return _d4_inverse(x, _parse_d4_mode(mode))
    raise ValueError(f"Unsupported TTA mode: {mode}")


@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor, tta_n: int = 8, tta_modes=None) -> torch.Tensor:
    model.eval()
    if tta_modes is None:
        n = max(1, min(8, int(tta_n)))
        tta_modes = [f"d4_{t}" for t in range(n)]
    prob_sum = None
    for mode in tta_modes:
        x_t = _tta_apply(x, mode)
        logits_t = extract_logits(model(x_t))
        prob_t = torch.sigmoid(logits_t)
        prob_t = _tta_inverse(prob_t, mode)
        if prob_sum is None:
            prob_sum = torch.zeros_like(prob_t)
        prob_sum += prob_t
    return prob_sum / float(max(1, len(tta_modes)))


@torch.no_grad()
def validate_iou(model, loader, device, threshold=0.35, tta_n=8, tta_modes=None, tta_tag=None):
    model.eval()
    if tta_modes is None:
        n = max(1, min(8, int(tta_n)))
        tta_modes = [f"d4_{t}" for t in range(n)]
    desc = f"val_{tta_tag}" if tta_tag is not None else f"val_tta{len(tta_modes)}"
    iou_sum = 0.0
    count = 0
    for inputs, labels in tqdm(loader, desc=desc, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        probs = tta_predict(model, inputs, tta_n=tta_n, tta_modes=tta_modes)
        preds = (probs > threshold).float()
        iou_sum += compute_iou(preds, labels)
        count += 1
    return iou_sum / max(1, count)


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

    train_datasets, val_datasets = build_datasets(project_root, args.stitch)
    for k in EXPERT_SPECS:
        print(f"[dataset] {k}: train={len(train_datasets[k])}, val={len(val_datasets[k])}")

    if len(train_datasets[args.expert]) == 0:
        raise RuntimeError(f"Expert dataset {args.expert} has no train samples after stitch filter.")
    if len(val_datasets[args.expert]) == 0:
        raise RuntimeError(f"Expert dataset {args.expert} has no val samples after stitch filter.")

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

    expert_bs = batch_size_for(args.expert, args.batch_128)
    val_loader = DataLoader(
        val_datasets[args.expert],
        batch_size=expert_bs,
        shuffle=False,
        drop_last=False,
    )
    print(f"[loader] val expert={args.expert}: bs={expert_bs}")

    model = build_model(args, n_channels=17, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir,
        f"expert_{args.expert}_s{args.stitch}_{args.model}_seed{args.seed}.pth",
    )

    rng = np.random.RandomState(args.seed + 1234)
    best_iou = -1.0
    save_epoch = min(120, args.epochs)
    saved_ok = False
    saved_err = ""
    val_tta_modes, val_tta_tag = select_val_tta_modes(args.expert, args.tta_n)
    if args.expert in RECT_VAL_TTA_EXPERTS and args.tta_n != 4:
        print(f"[tta] expert={args.expert} uses fixed rect4 TTA, ignore --tta-n={args.tta_n}")

    print(
        f"[config] expert={args.expert}, stitch={args.stitch}, epochs={args.epochs}, "
        f"half_mix={args.epochs // 2}, model={args.model}, val_tta={val_tta_tag}"
    )

    for epoch in range(args.epochs):
        chosen_bucket = choose_train_bucket(epoch, args.epochs, args.expert, rng)
        loader = train_loaders[chosen_bucket]

        train_loss = train_one_epoch(model, loader, optimizer, device)
        log_msg = (
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_bucket={chosen_bucket} | train_loss={train_loss:.4f}"
        )

        if (epoch + 1) % args.val_every == 0:
            val_iou = validate_iou(
                model,
                val_loader,
                device,
                threshold=args.threshold,
                tta_n=args.tta_n,
                tta_modes=val_tta_modes,
                tta_tag=val_tta_tag,
            )
            log_msg += f" | val_iou_{val_tta_tag}({args.expert})={val_iou:.4f}"
            if val_iou > best_iou:
                best_iou = val_iou

        if (epoch + 1) == save_epoch:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "expert": args.expert,
                "stitch": args.stitch,
                "seed": args.seed,
                "best_iou_so_far": best_iou,
                "tta_n": args.tta_n,
                "tta_tag": val_tta_tag,
                "tta_modes": val_tta_modes,
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
        print(f"[done] saved_epoch={save_epoch}, best_iou={best_iou:.4f}, save_path={save_path}")
    else:
        print(
            f"[done] save_failed_at_epoch={save_epoch}, err={saved_err}, "
            f"best_iou={best_iou:.4f}, save_path={save_path}"
        )


if __name__ == "__main__":
    main()
