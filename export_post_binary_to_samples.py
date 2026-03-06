import argparse
import os

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(
        "Export per-sample binary masks from a post-processed fused binary mosaic."
    )
    parser.add_argument("--test-root", type=str, default="./data/test_2")
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument(
        "--fused-binary",
        type=str,
        default="./data/test_2/fused_expert_pred/fused_pred_mask_post.tif",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/test_2/fused_expert_pred/pred_mask_post",
    )
    parser.add_argument(
        "--input-fg-threshold",
        type=float,
        default=0.5,
        help="Pixel > threshold is foreground after excluding nodata.",
    )
    parser.add_argument(
        "--output-fg-value",
        type=int,
        default=255,
        help="Foreground value written to sample masks.",
    )
    return parser.parse_args()


def _list_tifs(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".tif")])


def _offset_from_bounds(bounds, transform):
    min_x = float(transform.c)
    max_y = float(transform.f)
    res_x = float(transform.a)
    res_y = float(-transform.e)
    col_off = int(round((bounds.left - min_x) / res_x))
    row_off = int(round((max_y - bounds.top) / res_y))
    return row_off, col_off


def _save_like(path, arr2d, ref_profile, nodata=0):
    profile = dict(ref_profile)
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "zlevel": 6,
            "bigtiff": "IF_SAFER",
            "nodata": nodata,
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d.astype(np.uint8), 1)


def run(args):
    test_root = os.path.abspath(args.test_root)
    images_path = os.path.join(test_root, args.images_dir)
    fused_path = os.path.abspath(args.fused_binary)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(images_path):
        raise RuntimeError(f"Images directory not found: {images_path}")
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"Fused binary not found: {fused_path}")

    os.makedirs(out_dir, exist_ok=True)
    sample_names = _list_tifs(images_path)
    if len(sample_names) == 0:
        raise RuntimeError(f"No tif found in images dir: {images_path}")

    with rasterio.open(fused_path) as ds_fused:
        fused = ds_fused.read(1)
        fused_transform = ds_fused.transform
        fused_crs = ds_fused.crs
        fused_h = int(ds_fused.height)
        fused_w = int(ds_fused.width)
        fused_nodata = ds_fused.nodata

    exported = 0
    total_fg = 0
    for name in sample_names:
        sample_path = os.path.join(images_path, name)
        with rasterio.open(sample_path) as ds:
            h = int(ds.height)
            w = int(ds.width)
            bounds = ds.bounds
            sample_profile = ds.profile.copy()
            sample_crs = ds.crs

        if sample_crs != fused_crs:
            raise RuntimeError(f"CRS mismatch: sample={name}, sample_crs={sample_crs}, fused_crs={fused_crs}")

        out = np.zeros((h, w), dtype=np.uint8)
        r0, c0 = _offset_from_bounds(bounds, fused_transform)
        r1 = r0 + h
        c1 = c0 + w

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(fused_h, r1)
        cc1 = min(fused_w, c1)
        if rr0 < rr1 and cc0 < cc1:
            pr0 = rr0 - r0
            pc0 = cc0 - c0
            pr1 = pr0 + (rr1 - rr0)
            pc1 = pc0 + (cc1 - cc0)

            patch = fused[rr0:rr1, cc0:cc1]
            if fused_nodata is None:
                valid = np.ones_like(patch, dtype=bool)
            else:
                valid = patch != fused_nodata
            fg = valid & (patch.astype(np.float32) > float(args.input_fg_threshold))
            out[pr0:pr1, pc0:pc1][fg] = np.uint8(args.output_fg_value)

        out_path = os.path.join(out_dir, name)
        _save_like(out_path, out, sample_profile, nodata=0)
        exported += 1
        total_fg += int((out > 0).sum())

    print(f"[done] fused_binary={fused_path}")
    print(f"[done] sample_count={len(sample_names)}, exported={exported}")
    print(f"[done] out_dir={out_dir}")
    print(f"[done] total_fg_pixels={total_fg}")


if __name__ == "__main__":
    run(parse_args())
