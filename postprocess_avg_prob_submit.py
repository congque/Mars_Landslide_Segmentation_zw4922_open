import argparse
import os
import shutil
import zipfile

import numpy as np
import rasterio
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser("Average fused probs, postprocess mask, and export submission zip.")
    parser.add_argument("--test-root", type=str, default="./data/test_2")
    parser.add_argument("--prob-dirs", nargs="+", required=True)
    parser.add_argument("--prob-name", type=str, default="fused_prob.tif")
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-fg-area", type=int, default=64)
    parser.add_argument("--min-bg-area", type=int, default=32)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--zip-name", type=str, default="submission_pred_mask.zip")
    return parser.parse_args()


def _load_and_average(prob_paths):
    prob_sum = None
    profile = None
    transform = None
    crs = None
    shape = None

    for p in prob_paths:
        with rasterio.open(p) as ds:
            arr = ds.read(1).astype(np.float32)
            if prob_sum is None:
                prob_sum = np.zeros_like(arr, dtype=np.float32)
                profile = ds.profile.copy()
                transform = ds.transform
                crs = ds.crs
                shape = (ds.height, ds.width)
            else:
                if (ds.height, ds.width) != shape:
                    raise RuntimeError(f"Shape mismatch: {p}")
                if ds.crs != crs:
                    raise RuntimeError(f"CRS mismatch: {p}")
                if ds.transform != transform:
                    raise RuntimeError(f"Transform mismatch: {p}")
            prob_sum += arr
    return prob_sum / float(len(prob_paths)), profile


def _save_like(path, arr2d, ref_profile, dtype, nodata):
    profile = dict(ref_profile)
    profile.update(
        {
            "driver": "GTiff",
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
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(dtype), 1)


def _remove_small_connected(mask_bool, min_area):
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n = ndimage.label(mask_bool, structure=structure)
    if n <= 0:
        return mask_bool, 0, 0
    sizes = np.bincount(labels.ravel())
    remove_ids = np.where((sizes < min_area) & (np.arange(sizes.shape[0]) != 0))[0]
    if remove_ids.size == 0:
        return mask_bool, int(n), 0
    out = mask_bool.copy()
    out[np.isin(labels, remove_ids)] = False
    return out, int(n), int(remove_ids.size)


def _offset_from_bounds(bounds, transform):
    min_x = float(transform.c)
    max_y = float(transform.f)
    res_x = float(transform.a)
    res_y = float(-transform.e)
    col_off = int(round((bounds.left - min_x) / res_x))
    row_off = int(round((max_y - bounds.top) / res_y))
    return row_off, col_off


def run(args):
    test_root = os.path.abspath(args.test_root)
    out_dir = os.path.abspath(args.out_dir)
    images_path = os.path.join(test_root, args.images_dir)
    if not os.path.isdir(images_path):
        raise RuntimeError(f"Images directory not found: {images_path}")

    prob_paths = []
    for d in args.prob_dirs:
        p = os.path.join(test_root, d, args.prob_name)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        prob_paths.append(p)

    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, "pred_mask")
    if os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)

    avg_prob, ref_profile = _load_and_average(prob_paths)
    avg_prob_path = os.path.join(out_dir, "fused_prob_avg4.tif")
    _save_like(avg_prob_path, avg_prob, ref_profile, dtype="float32", nodata=None)

    raw_bool = avg_prob > float(args.threshold)
    raw_mask = (raw_bool.astype(np.uint8) * 255)
    raw_path = os.path.join(out_dir, f"fused_pred_mask_th{args.threshold:.3f}_raw.tif")
    _save_like(raw_path, raw_mask, ref_profile, dtype="uint8", nodata=0)

    fg_clean, fg_n_before, fg_removed = _remove_small_connected(raw_bool, int(args.min_fg_area))
    fg_path = os.path.join(out_dir, f"fused_pred_mask_th{args.threshold:.3f}_fg{args.min_fg_area}.tif")
    _save_like(fg_path, fg_clean.astype(np.uint8) * 255, ref_profile, dtype="uint8", nodata=0)

    bg_bool = ~fg_clean
    bg_clean, bg_n_before, bg_removed = _remove_small_connected(bg_bool, int(args.min_bg_area))
    # Filled tiny background holes are where bg changed from True to False.
    final_bool = fg_clean | (~bg_clean & bg_bool)
    final_mask = (final_bool.astype(np.uint8) * 255)
    final_path = os.path.join(
        out_dir,
        f"fused_pred_mask_th{args.threshold:.3f}_fg{args.min_fg_area}_bg{args.min_bg_area}.tif",
    )
    _save_like(final_path, final_mask, ref_profile, dtype="uint8", nodata=0)

    sample_names = sorted([f for f in os.listdir(images_path) if f.lower().endswith(".tif")])
    h_all = int(ref_profile["height"])
    w_all = int(ref_profile["width"])
    transform = ref_profile["transform"]

    for name in sample_names:
        p = os.path.join(images_path, name)
        with rasterio.open(p) as ds:
            h = int(ds.height)
            w = int(ds.width)
            bounds = ds.bounds
            sample_profile = ds.profile.copy()

        r0, c0 = _offset_from_bounds(bounds, transform)
        r1 = r0 + h
        c1 = c0 + w

        out = np.zeros((h, w), dtype=np.uint8)
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(h_all, r1)
        cc1 = min(w_all, c1)
        if rr0 < rr1 and cc0 < cc1:
            pr0 = rr0 - r0
            pc0 = cc0 - c0
            pr1 = pr0 + (rr1 - rr0)
            pc1 = pc0 + (cc1 - cc0)
            out[pr0:pr1, pc0:pc1] = final_mask[rr0:rr1, cc0:cc1]

        sample_out = os.path.join(pred_dir, name)
        _save_like(sample_out, out, sample_profile, dtype="uint8", nodata=0)

    zip_path = os.path.join(out_dir, args.zip_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sample_names:
            zf.write(os.path.join(pred_dir, name), arcname=name)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] prob_paths={len(prob_paths)}")
    print(f"[done] sample_count={len(sample_names)}")
    print(
        f"[done] fg_components_before={fg_n_before}, fg_removed={fg_removed}, "
        f"bg_components_before={bg_n_before}, bg_removed={bg_removed}"
    )
    print(
        f"[done] fg_pixels raw={int(raw_bool.sum())}, "
        f"after_fg={int(fg_clean.sum())}, final={int(final_bool.sum())}"
    )
    print(f"[done] zip={zip_path}")


if __name__ == "__main__":
    run(parse_args())
