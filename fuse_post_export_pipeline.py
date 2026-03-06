import argparse
import os
import subprocess
import sys
import zipfile

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(
        "Fuse multiple fused_prob folders, run post_process, and export per-sample masks."
    )
    parser.add_argument("--test-root", type=str, default="./data/test_2")
    parser.add_argument("--pred-dirs", nargs="+", required=True, help="Prediction folders under test-root.")
    parser.add_argument("--prob-name", type=str, default="fused_prob.tif")
    parser.add_argument("--valid-name", type=str, default="fused_valid_mask.tif")
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--min-area-step1", type=int, default=80)
    parser.add_argument("--min-area-step2", type=int, default=2000)
    parser.add_argument("--step3-prob-scale", type=float, default=35.0)
    parser.add_argument("--max-bg-area-step4", type=int, default=40)
    parser.add_argument("--zip-name", type=str, default="submission_pred_mask_post.zip")
    parser.add_argument(
        "--post-script",
        type=str,
        default="./post_process.py",
        help="Path to post-process script (the advanced post-processing logic).",
    )
    return parser.parse_args()


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


def _offset_from_bounds(bounds, transform):
    min_x = float(transform.c)
    max_y = float(transform.f)
    res_x = float(transform.a)
    res_y = float(-transform.e)
    col_off = int(round((bounds.left - min_x) / res_x))
    row_off = int(round((max_y - bounds.top) / res_y))
    return row_off, col_off


def _resolve_inputs(test_root, pred_dirs, prob_name, valid_name):
    prob_paths = []
    valid_paths = []
    for d in pred_dirs:
        folder = d if os.path.isabs(d) else os.path.join(test_root, d)
        p = os.path.join(folder, prob_name)
        v = os.path.join(folder, valid_name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing prob file: {p}")
        if not os.path.exists(v):
            raise FileNotFoundError(f"Missing valid mask file: {v}")
        prob_paths.append(p)
        valid_paths.append(v)
    return prob_paths, valid_paths


def _fuse_prob_with_valid(prob_paths, valid_paths):
    prob_sum = None
    valid_count = None
    ref_profile = None
    ref_transform = None
    ref_crs = None
    ref_shape = None

    for prob_path, valid_path in zip(prob_paths, valid_paths):
        with rasterio.open(prob_path) as ds_prob, rasterio.open(valid_path) as ds_valid:
            if ds_prob.shape != ds_valid.shape:
                raise RuntimeError(f"Shape mismatch within folder: {prob_path} vs {valid_path}")
            if ds_prob.transform != ds_valid.transform:
                raise RuntimeError(f"Transform mismatch within folder: {prob_path} vs {valid_path}")
            if ds_prob.crs != ds_valid.crs:
                raise RuntimeError(f"CRS mismatch within folder: {prob_path} vs {valid_path}")

            prob = ds_prob.read(1).astype(np.float32)
            valid = (ds_valid.read(1) > 0).astype(np.float32)

            if prob_sum is None:
                ref_profile = ds_prob.profile.copy()
                ref_transform = ds_prob.transform
                ref_crs = ds_prob.crs
                ref_shape = ds_prob.shape
                prob_sum = np.zeros(ref_shape, dtype=np.float32)
                valid_count = np.zeros(ref_shape, dtype=np.float32)
            else:
                if ds_prob.shape != ref_shape:
                    raise RuntimeError(f"Shape mismatch: {prob_path}")
                if ds_prob.transform != ref_transform:
                    raise RuntimeError(f"Transform mismatch: {prob_path}")
                if ds_prob.crs != ref_crs:
                    raise RuntimeError(f"CRS mismatch: {prob_path}")

            prob_sum += prob * valid
            valid_count += valid

    fused_prob = np.full(prob_sum.shape, np.float32(-9999.0), dtype=np.float32)
    valid = valid_count > 0.0
    fused_prob[valid] = prob_sum[valid] / valid_count[valid]
    fused_valid = (valid.astype(np.uint8) * 255)
    return fused_prob, fused_valid, ref_profile


def _run_post_process(
    python_exe,
    post_script,
    valid_mask_path,
    mean_prob_path,
    output_bin_path,
    output_prob_path,
    args,
):
    cmd = [
        python_exe,
        post_script,
        "--valid-mask",
        valid_mask_path,
        "--mean-prob",
        mean_prob_path,
        "--bin-threshold",
        str(float(args.threshold)),
        "--output-bin",
        output_bin_path,
        "--output-prob",
        output_prob_path,
        "--min-area-step1",
        str(int(args.min_area_step1)),
        "--min-area-step2",
        str(int(args.min_area_step2)),
        "--step3-prob-scale",
        str(float(args.step3_prob_scale)),
        "--max-bg-area-step4",
        str(int(args.max_bg_area_step4)),
    ]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _export_sample_masks(test_root, images_dir, fused_binary_path, out_pred_dir):
    images_path = os.path.join(test_root, images_dir)
    if not os.path.isdir(images_path):
        raise RuntimeError(f"Images directory not found: {images_path}")
    sample_names = sorted([f for f in os.listdir(images_path) if f.lower().endswith(".tif")])
    if len(sample_names) == 0:
        raise RuntimeError(f"No tif found in images dir: {images_path}")

    with rasterio.open(fused_binary_path) as ds_fused:
        fused = ds_fused.read(1)
        fused_transform = ds_fused.transform
        fused_crs = ds_fused.crs
        fused_h = int(ds_fused.height)
        fused_w = int(ds_fused.width)
        fused_nodata = ds_fused.nodata

    os.makedirs(out_pred_dir, exist_ok=True)
    exported = 0
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
            fg = valid & (patch > 0)
            out_patch = out[pr0:pr1, pc0:pc1]
            out_patch[fg] = 255
            out[pr0:pr1, pc0:pc1] = out_patch

        out_path = os.path.join(out_pred_dir, name)
        _save_like(out_path, out, sample_profile, dtype="uint8", nodata=0)
        exported += 1

    return sample_names, exported


def _zip_pred_mask(pred_dir, sample_names, zip_path):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sample_names:
            zf.write(os.path.join(pred_dir, name), arcname=name)


def run(args):
    test_root = os.path.abspath(args.test_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prob_paths, valid_paths = _resolve_inputs(
        test_root=test_root,
        pred_dirs=args.pred_dirs,
        prob_name=args.prob_name,
        valid_name=args.valid_name,
    )
    print("[input] folders:")
    for p in prob_paths:
        print(f"  - {os.path.dirname(p)}")

    fused_prob, fused_valid, ref_profile = _fuse_prob_with_valid(prob_paths, valid_paths)
    fused_prob_path = os.path.join(out_dir, "fused_prob_ensemble.tif")
    fused_valid_path = os.path.join(out_dir, "fused_valid_mask_ensemble.tif")
    _save_like(fused_prob_path, fused_prob, ref_profile, dtype="float32", nodata=np.float32(-9999.0))
    _save_like(fused_valid_path, fused_valid, ref_profile, dtype="uint8", nodata=0)

    post_script = args.post_script
    if not os.path.isabs(post_script):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, post_script)
        if os.path.exists(candidate):
            post_script = candidate
        else:
            post_script = os.path.abspath(post_script)
    if not os.path.exists(post_script):
        raise FileNotFoundError(f"Post-process script not found: {post_script}")

    post_bin_path = os.path.join(out_dir, "fused_pred_mask_post.tif")
    post_prob_path = os.path.join(out_dir, "fused_prob_post.tif")
    _run_post_process(
        python_exe=sys.executable,
        post_script=post_script,
        valid_mask_path=fused_valid_path,
        mean_prob_path=fused_prob_path,
        output_bin_path=post_bin_path,
        output_prob_path=post_prob_path,
        args=args,
    )

    pred_dir = os.path.join(out_dir, "pred_mask_post")
    sample_names, exported = _export_sample_masks(
        test_root=test_root,
        images_dir=args.images_dir,
        fused_binary_path=post_bin_path,
        out_pred_dir=pred_dir,
    )
    zip_path = os.path.join(out_dir, args.zip_name)
    _zip_pred_mask(pred_dir, sample_names, zip_path)

    print(f"[done] fused_prob={fused_prob_path}")
    print(f"[done] fused_valid={fused_valid_path}")
    print(f"[done] post_prob={post_prob_path}")
    print(f"[done] post_bin={post_bin_path}")
    print(f"[done] sample_exported={exported}, pred_dir={pred_dir}")
    print(f"[done] zip={zip_path}")


if __name__ == "__main__":
    run(parse_args())
