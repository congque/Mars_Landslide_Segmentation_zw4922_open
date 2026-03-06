import argparse
import os
import re

import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
from tqdm import tqdm

from dataset_process import CHANNEL_MAXS_15, CHANNEL_MINS_15
from nets.nets import UNet_MSHD_Heavy
from nets.nets_segformer import SegFormer
from nets.nets_unetres import (
    UNetResNet34,
    UNetResNet50,
    UNetConvNeXtBase,
    UNetConvNeXtSmall,
    UNetEfficientNetB4,
    UNetEfficientNetB5,
)


RES_SPECS = {
    "128128": {"folder": "images_15d", "tta": [f"d4_{t}" for t in range(8)]},
    "128256": {"folder": "images_15d_128256", "tta": ["id", "flip_ud", "flip_lr", "rot180"]},
    "256128": {"folder": "images_15d_256128", "tta": ["id", "flip_ud", "flip_lr", "rot180"]},
    "256256": {"folder": "images_15d_256256", "tta": [f"d4_{t}" for t in range(8)]},
}

SUPPORTED_MODELS = ("ures34", "ures50", "uconvnextb", "uconvnexts", "ueffb4", "ueffb5", "segformer", "umshd")
SQRT_IDXS = {0, 1, 8, 9, 11, 12}
CBRT_IDXS = {3, 4, 5, 6}
SIGN_SQRT_IDXS = {7, 10}


def parse_args():
    parser = argparse.ArgumentParser("Fuse expert predictions into one georeferenced probability mosaic.")
    parser.add_argument("--test-root", type=str, default="./data/test_2")
    parser.add_argument("--base-images-dir", type=str, default="images")
    parser.add_argument("--pths-dir", type=str, default="./pths_expert")
    parser.add_argument("--save-dir", type=str, default="./data/test_2/fused_expert_pred")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS),
        help="Auto-load 4 experts by model name, e.g. --model umshd",
    )
    parser.add_argument("--ckpt-128128", type=str, default="")
    parser.add_argument("--ckpt-128256", type=str, default="")
    parser.add_argument("--ckpt-256128", type=str, default="")
    parser.add_argument("--ckpt-256256", type=str, default="")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--save-debug-rasters",
        action="store_true",
        help="Also save fused_weight_sum.tif and fused_weighted_prob_sum.tif (large files).",
    )
    parser.add_argument(
        "--no-export-sample-mask",
        action="store_true",
        help="Disable exporting per-sample mask tif files under <save_dir>/pred_mask.",
    )
    parser.add_argument(
        "--export-sample-prob",
        action="store_true",
        help="Also export per-sample probability tif files under <save_dir>/sample_prob (float32).",
    )
    parser.add_argument("--segformer-decoder-dim", type=int, default=256)
    return parser.parse_args()


def _list_tifs(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")])


def _extract_logits(outputs):
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


def _d4_apply(x, t):
    if t < 4:
        return torch.rot90(x, k=t, dims=(2, 3))
    y = torch.flip(x, dims=[3])
    return torch.rot90(y, k=t - 4, dims=(2, 3))


def _d4_inverse(x, t):
    if t < 4:
        return torch.rot90(x, k=(4 - t) % 4, dims=(2, 3))
    k = t - 4
    y = torch.rot90(x, k=(4 - k) % 4, dims=(2, 3))
    return torch.flip(y, dims=[3])


def _tta_apply(x, mode):
    if mode == "id":
        return x
    if mode == "flip_ud":
        return torch.flip(x, dims=[2])
    if mode == "flip_lr":
        return torch.flip(x, dims=[3])
    if mode == "rot180":
        return torch.rot90(x, k=2, dims=(2, 3))
    if mode.startswith("d4_"):
        t = int(mode.split("_", 1)[1])
        return _d4_apply(x, t)
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _tta_inverse(x, mode):
    if mode == "id":
        return x
    if mode == "flip_ud":
        return torch.flip(x, dims=[2])
    if mode == "flip_lr":
        return torch.flip(x, dims=[3])
    if mode == "rot180":
        return torch.rot90(x, k=2, dims=(2, 3))
    if mode.startswith("d4_"):
        t = int(mode.split("_", 1)[1])
        return _d4_inverse(x, t)
    raise ValueError(f"Unsupported TTA mode: {mode}")


@torch.no_grad()
def tta_predict(model, x, tta_modes):
    model.eval()
    prob_sum = None
    for mode in tta_modes:
        x_t = _tta_apply(x, mode)
        logits_t = _extract_logits(model(x_t))
        prob_t = torch.sigmoid(logits_t)
        prob_t = _tta_inverse(prob_t, mode)
        if prob_sum is None:
            prob_sum = torch.zeros_like(prob_t)
        prob_sum += prob_t
    return prob_sum / float(max(1, len(tta_modes)))


def _parse_model_from_ckpt_name(path):
    name = os.path.basename(path).lower()
    for model_name in SUPPORTED_MODELS:
        if f"_{model_name}_" in name:
            return model_name
    return ""


def _build_model(model_name, args, device):
    if model_name == "segformer":
        model = SegFormer(
            n_channels=17,
            n_classes=1,
            decoder_dim=args.segformer_decoder_dim,
        )
    elif model_name == "ures34":
        model = UNetResNet34(n_channels=17, n_classes=1, base_channel=32)
    elif model_name == "ures50":
        model = UNetResNet50(n_channels=17, n_classes=1, base_channel=32)
    elif model_name == "uconvnextb":
        model = UNetConvNeXtBase(n_channels=17, n_classes=1, base_channel=32, pretrained=False)
    elif model_name == "uconvnexts":
        model = UNetConvNeXtSmall(n_channels=17, n_classes=1, base_channel=32, pretrained=False)
    elif model_name == "ueffb4":
        model = UNetEfficientNetB4(n_channels=17, n_classes=1, base_channel=32, pretrained=False)
    elif model_name == "ueffb5":
        model = UNetEfficientNetB5(n_channels=17, n_classes=1, base_channel=32, pretrained=False)
    elif model_name == "umshd":
        model = UNet_MSHD_Heavy(n_channels=17, n_classes=1, base_channel=32)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def _load_model_from_ckpt(ckpt_path, model_name, args, device):
    model = _build_model(model_name, args, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _select_checkpoints(args):
    manual = {
        "128128": args.ckpt_128128,
        "128256": args.ckpt_128256,
        "256128": args.ckpt_256128,
        "256256": args.ckpt_256256,
    }

    if not os.path.isdir(args.pths_dir):
        raise RuntimeError(f"Checkpoint directory not found: {args.pths_dir}")

    selected = {}
    for res in RES_SPECS.keys():
        if manual[res]:
            ckpt_path = manual[res]
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.abspath(ckpt_path)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{res} checkpoint not found: {ckpt_path}")
            model_name = _parse_model_from_ckpt_name(ckpt_path)
            if model_name == "":
                raise ValueError(
                    f"Cannot infer model type from checkpoint name for {res}: {ckpt_path}. "
                    f"Please use filename containing one of: {', '.join(SUPPORTED_MODELS)}."
                )
            if model_name != args.model:
                raise ValueError(
                    f"Manual checkpoint model mismatch for {res}: expected {args.model}, got {model_name}."
                )
            selected[res] = (ckpt_path, model_name)
            continue

        # Naming convention: expert_<res>_<model>_*.pth (e.g. expert_128256_umshd_full.pth)
        pat = re.compile(rf"^expert_{res}_{re.escape(args.model)}(?:_.+)?\.pth$", re.IGNORECASE)
        candidates = []
        for f in os.listdir(args.pths_dir):
            if not f.endswith(".pth"):
                continue
            if pat.match(f) is None:
                continue
            path = os.path.join(args.pths_dir, f)
            candidates.append((path, args.model))

        if len(candidates) == 0:
            if args.allow_missing:
                continue
            raise RuntimeError(
                f"No checkpoint found for expert {res} and model {args.model} in {args.pths_dir}"
            )

        candidates = sorted(candidates, key=lambda x: os.path.getmtime(x[0]), reverse=True)
        selected[res] = candidates[0]

    return selected


def _build_union_grid(base_tifs):
    if len(base_tifs) == 0:
        raise RuntimeError("No base tif found for building global mosaic grid.")

    min_x = None
    min_y = None
    max_x = None
    max_y = None
    crs = None
    res_x = None
    res_y = None

    for p in base_tifs:
        with rasterio.open(p) as ds:
            if crs is None:
                crs = ds.crs
                res_x = float(ds.transform.a)
                res_y = float(-ds.transform.e)
                if not np.isclose(ds.transform.b, 0.0) or not np.isclose(ds.transform.d, 0.0):
                    raise RuntimeError(f"Only north-up rasters are supported, got rotated transform: {p}")
            else:
                if ds.crs != crs:
                    raise RuntimeError(f"CRS mismatch: {p}, {ds.crs} vs {crs}")
                if not np.isclose(float(ds.transform.a), res_x) or not np.isclose(float(-ds.transform.e), res_y):
                    raise RuntimeError(f"Resolution mismatch in {p}")

            b = ds.bounds
            min_x = b.left if min_x is None else min(min_x, b.left)
            min_y = b.bottom if min_y is None else min(min_y, b.bottom)
            max_x = b.right if max_x is None else max(max_x, b.right)
            max_y = b.top if max_y is None else max(max_y, b.top)

    width = int(np.ceil((max_x - min_x) / res_x))
    height = int(np.ceil((max_y - min_y) / res_y))
    transform = Affine(res_x, 0, min_x, 0, -res_y, max_y)
    return {
        "min_x": float(min_x),
        "max_y": float(max_y),
        "res_x": float(res_x),
        "res_y": float(res_y),
        "width": width,
        "height": height,
        "transform": transform,
        "crs": crs,
    }


def _offset_from_bounds(bounds, grid):
    col_off = int(round((bounds.left - grid["min_x"]) / grid["res_x"]))
    row_off = int(round((grid["max_y"] - bounds.top) / grid["res_y"]))
    return row_off, col_off


def _window_intersection(bounds, h, w, grid):
    r0, c0 = _offset_from_bounds(bounds, grid)
    r1 = r0 + h
    c1 = c0 + w

    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(grid["height"], r1)
    cc1 = min(grid["width"], c1)
    if rr0 >= rr1 or cc0 >= cc1:
        return None

    pr0 = rr0 - r0
    pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    return rr0, rr1, cc0, cc1, pr0, pr1, pc0, pc1


def _preprocess_to_17ch(img_hwc):
    if img_hwc.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={img_hwc.shape}")
    if img_hwc.shape[2] < 15:
        raise ValueError(f"Expected at least 15 channels, got C={img_hwc.shape[2]}")

    x = img_hwc.astype(np.float32).copy()
    x[:, :, 1][x[:, :, 1] < 0] = 0

    dem = x[:, :, 2].astype(np.float32)
    m1 = float(dem.mean())
    s1 = float(dem.std())
    if s1 < 1e-6:
        b3_z_raw = np.zeros_like(dem, dtype=np.float32)
    else:
        b3_z_raw = (dem - m1) / (s1 + 1e-6)

    q25 = np.percentile(dem, 25)
    q75 = np.percentile(dem, 75)
    iqr = q75 - q25 + 1e-6
    dem_robust = (dem - np.median(dem)) / iqr

    for c in range(15):
        chan = x[:, :, c]
        if c in SQRT_IDXS:
            chan = np.sqrt(chan + 1e-4)
        elif c in CBRT_IDXS:
            chan = np.power(chan + 1e-4, 1.0 / 3.0)
        elif c in SIGN_SQRT_IDXS:
            chan = np.sign(chan) * np.sqrt(np.abs(chan) + 1e-4)

        denom = CHANNEL_MAXS_15[c] - CHANNEL_MINS_15[c]
        if abs(denom) < 1e-12:
            chan = np.zeros_like(chan, dtype=np.float32)
        else:
            chan = (chan - CHANNEL_MINS_15[c]) / denom
        x[:, :, c] = np.clip(chan, 0, 1)

    x15 = x[:, :, :15]
    chw = np.transpose(x15, (2, 0, 1)).astype(np.float32)
    chw = np.concatenate([chw, b3_z_raw[None, :, :], dem_robust[None, :, :]], axis=0)
    return chw


def _distance_weight(h, w, min_weight=0.5):
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ny = np.abs(yy - cy) / max(cy, 1.0)
    nx = np.abs(xx - cx) / max(cx, 1.0)
    dist = np.maximum(nx, ny)  # normalized to [0,1]
    weight = 1.0 - (1.0 - min_weight) * dist
    return np.clip(weight, min_weight, 1.0).astype(np.float32)


def _build_tiff_profile(
    dtype,
    nodata,
    height,
    width,
    base_profile=None,
    crs=None,
    transform=None,
):
    profile = dict(base_profile) if base_profile is not None else {}
    profile.update(
        {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": 1,
            "dtype": dtype,
            "compress": "deflate",
            "predictor": 3 if np.dtype(dtype).kind == "f" else 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "zlevel": 6,
            "bigtiff": "IF_SAFER",
            "nodata": nodata,
        }
    )
    if crs is not None:
        profile["crs"] = crs
    if transform is not None:
        profile["transform"] = transform
    return profile


def _save_raster(path, arr2d, grid, dtype="float32", nodata=None):
    profile = _build_tiff_profile(
        dtype=dtype,
        nodata=nodata,
        height=arr2d.shape[0],
        width=arr2d.shape[1],
        crs=grid["crs"],
        transform=grid["transform"],
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype), 1)


def _safe_save_raster(path, arr2d, grid, dtype="float32", nodata=None):
    try:
        _save_raster(path, arr2d, grid, dtype=dtype, nodata=nodata)
        print(f"[done] saved={path}")
        return True
    except Exception as e:
        print(f"[warn] save failed: {path}")
        print(f"[warn] reason: {e}")
        # Cleanup partial files if write was interrupted (e.g., disk quota exceeded).
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"[warn] removed partial file: {path}")
        except Exception as e2:
            print(f"[warn] failed to remove partial file: {path}, reason: {e2}")
        return False


def _save_like_reference(path, arr2d, dtype, ref_profile, nodata=None):
    profile = _build_tiff_profile(
        dtype=dtype,
        nodata=nodata,
        height=arr2d.shape[0],
        width=arr2d.shape[1],
        base_profile=ref_profile,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype), 1)


def _export_sample_outputs(base_tifs, pred_mask, final_prob, grid, save_dir, export_prob=False):
    pred_dir = os.path.join(save_dir, "pred_mask")
    os.makedirs(pred_dir, exist_ok=True)
    prob_dir = None
    if export_prob:
        prob_dir = os.path.join(save_dir, "sample_prob")
        os.makedirs(prob_dir, exist_ok=True)

    exported = 0
    for path in tqdm(base_tifs, desc="export_sample", leave=False):
        name = os.path.basename(path)
        with rasterio.open(path) as ds:
            bounds = ds.bounds
            h = int(ds.height)
            w = int(ds.width)
            ref_profile = ds.profile.copy()

        out_mask = np.zeros((h, w), dtype=np.uint8)
        out_prob = np.zeros((h, w), dtype=np.float32) if export_prob else None

        win = _window_intersection(bounds=bounds, h=h, w=w, grid=grid)
        if win is not None:
            rr0, rr1, cc0, cc1, pr0, pr1, pc0, pc1 = win
            out_mask[pr0:pr1, pc0:pc1] = pred_mask[rr0:rr1, cc0:cc1]
            if export_prob:
                out_prob[pr0:pr1, pc0:pc1] = final_prob[rr0:rr1, cc0:cc1]

        _save_like_reference(
            os.path.join(pred_dir, name),
            out_mask,
            dtype="uint8",
            ref_profile=ref_profile,
            nodata=0,
        )
        if export_prob:
            _save_like_reference(
                os.path.join(prob_dir, name),
                out_prob,
                dtype="float32",
                ref_profile=ref_profile,
                nodata=None,
            )
        exported += 1

    print(f"[done] exported sample masks: {exported}, dir={pred_dir}")
    if export_prob and prob_dir is not None:
        print(f"[done] exported sample probs: {exported}, dir={prob_dir}")


def run(args):
    test_root = os.path.abspath(args.test_root)
    base_folder = os.path.join(test_root, args.base_images_dir)
    base_tifs = _list_tifs(base_folder)
    if len(base_tifs) == 0:
        raise RuntimeError(f"No tif found in base folder: {base_folder}")

    selected = _select_checkpoints(args)
    if len(selected) == 0:
        raise RuntimeError("No expert checkpoint selected.")

    print("[ckpt] selected experts:")
    for res, (path, model_name) in selected.items():
        print(f"  - {res}: {model_name}, {os.path.basename(path)}")

    grid = _build_union_grid(base_tifs)
    print(
        f"[grid] size=({grid['height']}, {grid['width']}), "
        f"res=({grid['res_x']}, {grid['res_y']}), crs={grid['crs']}"
    )

    prob_sum = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
    weight_sum = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
    weight_cache = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    models = {}
    for res, (ckpt_path, model_name) in selected.items():
        models[res] = _load_model_from_ckpt(ckpt_path, model_name, args, device)

    total_tiles = 0
    for res, spec in RES_SPECS.items():
        if res not in models:
            print(f"[skip] {res}: no checkpoint selected")
            continue
        folder = os.path.join(test_root, spec["folder"])
        tifs = _list_tifs(folder)
        if len(tifs) == 0:
            if args.allow_missing:
                print(f"[skip] {res}: no tif in {folder}")
                continue
            raise RuntimeError(f"No tif found for {res}: {folder}")

        model = models[res]
        tta_modes = spec["tta"]
        print(f"[run] {res}: files={len(tifs)}, tta={tta_modes}")
        total_tiles += len(tifs)

        for path in tqdm(tifs, desc=f"predict_{res}", leave=False):
            with rasterio.open(path) as ds:
                if ds.crs != grid["crs"]:
                    raise RuntimeError(f"CRS mismatch: {path}")
                if not np.isclose(float(ds.transform.a), grid["res_x"]) or not np.isclose(
                    float(-ds.transform.e), grid["res_y"]
                ):
                    raise RuntimeError(f"Resolution mismatch: {path}")
                arr = ds.read().transpose(1, 2, 0).astype(np.float32)
                bounds = ds.bounds

            chw = _preprocess_to_17ch(arr)
            inp = torch.from_numpy(chw).unsqueeze(0).to(device)
            prob = tta_predict(model, inp, tta_modes=tta_modes)[0, 0].detach().cpu().numpy().astype(np.float32)

            h, w = prob.shape
            if (h, w) not in weight_cache:
                weight_cache[(h, w)] = _distance_weight(h, w, min_weight=0.5)
            wmap = weight_cache[(h, w)]

            win = _window_intersection(bounds=bounds, h=h, w=w, grid=grid)
            if win is None:
                continue
            rr0, rr1, cc0, cc1, pr0, pr1, pc0, pc1 = win

            p = prob[pr0:pr1, pc0:pc1]
            w_patch = wmap[pr0:pr1, pc0:pc1]
            prob_sum[rr0:rr1, cc0:cc1] += p * w_patch
            weight_sum[rr0:rr1, cc0:cc1] += w_patch

    if total_tiles == 0:
        raise RuntimeError("No tiles were processed.")

    eps = 1e-6
    final_prob = np.zeros_like(prob_sum, dtype=np.float32)
    valid = weight_sum > eps
    final_prob[valid] = prob_sum[valid] / weight_sum[valid]

    pred_mask = (final_prob > args.threshold).astype(np.uint8) * 255
    valid_mask = valid.astype(np.uint8) * 255

    os.makedirs(args.save_dir, exist_ok=True)
    prob_path = os.path.join(args.save_dir, "fused_prob.tif")
    wsum_path = os.path.join(args.save_dir, "fused_weight_sum.tif")
    sum_path = os.path.join(args.save_dir, "fused_weighted_prob_sum.tif")
    mask_path = os.path.join(args.save_dir, "fused_pred_mask.tif")
    valid_path = os.path.join(args.save_dir, "fused_valid_mask.tif")

    if not args.no_export_sample_mask:
        _export_sample_outputs(
            base_tifs=base_tifs,
            pred_mask=pred_mask,
            final_prob=final_prob,
            grid=grid,
            save_dir=args.save_dir,
            export_prob=args.export_sample_prob,
        )
    else:
        print("[done] skip sample export (--no-export-sample-mask).")

    # Save light-weight outputs first so quota pressure does not lose all results.
    _safe_save_raster(mask_path, pred_mask, grid, dtype="uint8", nodata=0)
    _safe_save_raster(valid_path, valid_mask, grid, dtype="uint8", nodata=0)
    _safe_save_raster(prob_path, final_prob, grid, dtype="float32", nodata=None)
    if args.save_debug_rasters:
        _safe_save_raster(wsum_path, weight_sum, grid, dtype="float32", nodata=0.0)
        _safe_save_raster(sum_path, prob_sum, grid, dtype="float32", nodata=0.0)
    else:
        print("[done] skip debug rasters (use --save-debug-rasters to enable).")

    print(f"[done] valid_pixels={int(valid.sum())}, all_pixels={valid.size}")


if __name__ == "__main__":
    run(parse_args())
