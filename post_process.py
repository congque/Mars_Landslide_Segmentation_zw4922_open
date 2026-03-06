import argparse
from collections import deque
from pathlib import Path

import numpy as np
import rasterio


def require_scipy():
    try:
        from scipy.ndimage import binary_dilation, find_objects, label
    except ImportError as exc:
        raise ImportError(
            "scipy is required. Install with: pip install scipy"
        ) from exc
    return binary_dilation, label, find_objects


def validate_alignment(ds_a, ds_b, path_a: Path, path_b: Path) -> None:
    if ds_a.width != ds_b.width or ds_a.height != ds_b.height:
        raise ValueError(
            f"Shape mismatch: {path_a} vs {path_b} "
            f"({ds_a.width}x{ds_a.height} != {ds_b.width}x{ds_b.height})"
        )
    if ds_a.transform != ds_b.transform:
        raise ValueError(f"Transform mismatch: {path_a} vs {path_b}")
    if ds_a.crs != ds_b.crs:
        raise ValueError(f"CRS mismatch: {path_a} vs {path_b}")


def filter_small_components(fg_mask: np.ndarray, min_area: int, label_func, structure: np.ndarray):
    labels, num = label_func(fg_mask, structure=structure)
    if num == 0:
        return fg_mask.copy(), 0, 0
    areas = np.bincount(labels.ravel())
    remove_ids = np.where((areas < min_area) & (np.arange(areas.size) > 0))[0]
    if remove_ids.size == 0:
        return fg_mask.copy(), 0, num
    out = fg_mask.copy()
    out[np.isin(labels, remove_ids)] = False
    return out, int(remove_ids.size), num


def filter_touching_invalid(
    fg_mask: np.ndarray,
    invalid_mask: np.ndarray,
    min_area: int,
    label_func,
    dilate_func,
    structure: np.ndarray,
):
    labels, num = label_func(fg_mask, structure=structure)
    if num == 0:
        return fg_mask.copy(), 0, 0

    invalid_or_adjacent = dilate_func(invalid_mask, structure=structure)
    touching_labels = np.unique(labels[invalid_or_adjacent & (labels > 0)])
    areas = np.bincount(labels.ravel())
    remove_ids = touching_labels[areas[touching_labels] < min_area]
    if remove_ids.size == 0:
        return fg_mask.copy(), 0, num

    out = fg_mask.copy()
    out[np.isin(labels, remove_ids)] = False
    return out, int(remove_ids.size), num


def filter_touching_invalid_by_prob_distance(
    fg_mask: np.ndarray,
    invalid_mask: np.ndarray,
    mean_prob: np.ndarray,
    prob_scale: float,
    dilate_func,
    label_func,
    find_objects_func,
    structure: np.ndarray,
):
    labels, num = label_func(fg_mask, structure=structure)
    if num == 0:
        return fg_mask.copy(), 0

    invalid_or_adjacent = dilate_func(invalid_mask, structure=structure)
    touching_component_ids = np.unique(labels[(labels > 0) & invalid_or_adjacent])
    if touching_component_ids.size == 0:
        return fg_mask.copy(), 0

    out = fg_mask.copy()
    removed_pixels = 0
    component_slices = find_objects_func(labels)

    for comp_id in touching_component_ids:
        comp_id = int(comp_id)
        sl = component_slices[comp_id - 1]
        if sl is None:
            continue

        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        pr0 = max(0, r0 - 1)
        pr1 = min(labels.shape[0], r1 + 1)
        pc0 = max(0, c0 - 1)
        pc1 = min(labels.shape[1], c1 + 1)
        psl = (slice(pr0, pr1), slice(pc0, pc1))

        labels_roi = labels[psl]
        component = labels_roi == comp_id
        if not np.any(component):
            continue

        invalid_roi = invalid_mask[psl]
        prob_roi = mean_prob[psl]
        boundary_pixels = component & invalid_or_adjacent[psl]
        if not np.any(boundary_pixels):
            continue

        n_touch_invalid = int((invalid_roi & dilate_func(component, structure=structure)).sum())
        threshold = float(n_touch_invalid) * float(prob_scale)

        visited = np.zeros_like(component, dtype=bool)
        removed_local = np.zeros_like(component, dtype=bool)
        q = deque()
        rr, cc = np.where(boundary_pixels)
        for r, c in zip(rr, cc):
            visited[r, c] = True
            q.append((r, c))

        csum = 0.0
        exceeded = False
        h, w = component.shape
        while q:
            r, c = q.popleft()
            removed_local[r, c] = True
            p = float(prob_roi[r, c])
            if np.isnan(p):
                p = 0.0
            csum += p
            if csum > threshold:
                exceeded = True
                break

            if r > 0 and component[r - 1, c] and not visited[r - 1, c]:
                visited[r - 1, c] = True
                q.append((r - 1, c))
            if r + 1 < h and component[r + 1, c] and not visited[r + 1, c]:
                visited[r + 1, c] = True
                q.append((r + 1, c))
            if c > 0 and component[r, c - 1] and not visited[r, c - 1]:
                visited[r, c - 1] = True
                q.append((r, c - 1))
            if c + 1 < w and component[r, c + 1] and not visited[r, c + 1]:
                visited[r, c + 1] = True
                q.append((r, c + 1))

        if not exceeded:
            removed_local |= visited

        if np.any(removed_local):
            rloc, cloc = np.where(removed_local)
            rglob = rloc + pr0
            cglob = cloc + pc0
            out[rglob, cglob] = False
            removed_pixels += int(removed_local.sum())

    return out, removed_pixels


def fill_small_background_components(
    fg_mask: np.ndarray,
    valid_mask: np.ndarray,
    max_area: int,
    label_func,
    structure: np.ndarray,
):
    bg_mask = (~fg_mask) & valid_mask
    labels, num = label_func(bg_mask, structure=structure)
    if num == 0:
        return fg_mask.copy(), 0, 0
    areas = np.bincount(labels.ravel())
    fill_ids = np.where((areas < max_area) & (np.arange(areas.size) > 0))[0]
    if fill_ids.size == 0:
        return fg_mask.copy(), 0, num
    out = fg_mask.copy()
    out[np.isin(labels, fill_ids)] = True
    return out, int(fill_ids.size), num


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split into 4 regions (x=w-400, y=1500), process separately, then merge. "
            "Output includes binary mask and post-processed probability GeoTIFFs "
            "with same spatial layout as mean_prob."
        )
    )
    parser.add_argument(
        "--fused-dir",
        type=Path,
        default=None,
        help=(
            "Directory produced by predict_expert_fused_geo.py. If set, input/output paths "
            "are auto-resolved from this folder."
        ),
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        default=Path("data") / "test_2" / "fused_expert_pred" / "fused_valid_mask.tif",
        help="Valid mask raster (0 means invalid).",
    )
    parser.add_argument(
        "--mean-prob",
        type=Path,
        default=Path("data") / "test_2" / "fused_expert_pred" / "fused_prob.tif",
        help="Mean probability raster for binarization and step 3.",
    )
    parser.add_argument(
        "--bin-threshold",
        type=float,
        default=0.60,
        help="Binarization threshold (>= threshold -> foreground).",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=Path("data") / "test_2" / "fused_expert_pred" / "fused_pred_mask_post.tif",
        help="Output binary GeoTIFF.",
    )
    parser.add_argument(
        "--output-prob",
        type=Path,
        default=Path("data") / "test_2" / "fused_expert_pred" / "fused_prob_post.tif",
        help="Output post-processed probability GeoTIFF.",
    )
    parser.add_argument("--min-area-step1", type=int, default=80)
    parser.add_argument("--min-area-step2", type=int, default=2000)
    parser.add_argument("--step3-prob-scale", type=float, default=35.0)
    parser.add_argument("--max-bg-area-step4", type=int, default=40)
    args = parser.parse_args()

    if args.fused_dir is not None:
        fused_dir = args.fused_dir
        args.valid_mask = fused_dir / "fused_valid_mask.tif"
        args.mean_prob = fused_dir / "fused_prob.tif"
        args.output_bin = fused_dir / "fused_pred_mask_post.tif"
        args.output_prob = fused_dir / "fused_prob_post.tif"

    if not args.valid_mask.exists():
        raise FileNotFoundError(f"Valid mask not found: {args.valid_mask}")
    if not args.mean_prob.exists():
        raise FileNotFoundError(f"Mean prob not found: {args.mean_prob}")

    dilate_func, label_func, find_objects_func = require_scipy()

    with (rasterio.open(args.valid_mask) as ds_valid, rasterio.open(args.mean_prob) as ds_prob):
        validate_alignment(ds_valid, ds_prob, args.valid_mask, args.mean_prob)
        valid_arr = ds_valid.read(1)
        mean_prob_arr = ds_prob.read(1).astype(np.float32)
        prob_nodata = ds_prob.nodata
        profile = ds_prob.profile.copy()

    if prob_nodata is not None:
        mean_prob_arr[mean_prob_arr == prob_nodata] = np.nan

    invalid_mask = valid_arr == 0
    valid_prob_mask = (~np.isnan(mean_prob_arr)) & (~invalid_mask)
    original_fg = (mean_prob_arr >= float(args.bin_threshold)) & valid_prob_mask
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    h, w = original_fg.shape
    x_cut = max(0, w - min(500, w))
    y_cut = min(1500, h)
    region_masks = [np.zeros_like(original_fg, dtype=bool) for _ in range(4)]
    region_masks[0][:y_cut, :x_cut] = True
    region_masks[1][:y_cut, x_cut:] = True
    region_masks[2][y_cut:, :x_cut] = True
    region_masks[3][y_cut:, x_cut:] = True

    fg_final = np.zeros_like(original_fg, dtype=bool)

    for region_idx, region_mask in enumerate(region_masks):
        fg0 = original_fg & region_mask
        fg1, _, _ = filter_small_components(fg0, args.min_area_step1, label_func, structure)
        fg2, _, _ = filter_touching_invalid(
            fg1, invalid_mask, args.min_area_step2, label_func, dilate_func, structure
        )
        region_scale = args.step3_prob_scale * 1.7 if region_idx == 1 else args.step3_prob_scale
        fg3, _ = filter_touching_invalid_by_prob_distance(
            fg2,
            invalid_mask,
            mean_prob_arr,
            region_scale,
            dilate_func,
            label_func,
            find_objects_func,
            structure,
        )
        fg4, _, _ = fill_small_background_components(
            fg3, valid_prob_mask & region_mask, args.max_bg_area_step4, label_func, structure
        )
        fg_final |= fg4

    out_nodata = 255
    out = np.zeros(mean_prob_arr.shape, dtype=np.uint8)
    out[fg_final] = 1
    out[~valid_prob_mask] = np.uint8(out_nodata)

    # Post-processed probability map:
    # background is clamped below threshold, foreground is clamped to threshold+.
    # This guarantees thresholding at bin_threshold reproduces fg_final exactly.
    prob_out = np.full(mean_prob_arr.shape, np.float32(-9999.0), dtype=np.float32)
    thr = float(args.bin_threshold)
    below_thr = np.nextafter(np.float32(thr), np.float32(-np.inf))
    above_thr = np.float32(thr)
    in_valid = valid_prob_mask
    prob_out[in_valid] = mean_prob_arr[in_valid]
    bg_valid = in_valid & (~fg_final)
    fg_valid = in_valid & fg_final
    prob_out[bg_valid] = np.minimum(prob_out[bg_valid], below_thr)
    prob_out[fg_valid] = np.maximum(prob_out[fg_valid], above_thr)

    # Keep geospatial layout identical to mean_prob (size, transform, CRS, extent).
    profile.update(dtype="uint8", count=1, compress="lzw", nodata=out_nodata)
    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_bin, "w", **profile) as dst:
        dst.write(out, 1)

    prob_profile = profile.copy()
    prob_profile.update(dtype="float32", nodata=np.float32(-9999.0))
    args.output_prob.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_prob, "w", **prob_profile) as dst:
        dst.write(prob_out, 1)

    print(f"Saved binary raster: {args.output_bin}")
    print(f"Saved post-processed probability raster: {args.output_prob}")
    print("Spatial layout matches mean_prob.tif (same size/transform/CRS).")


if __name__ == "__main__":
    main()
