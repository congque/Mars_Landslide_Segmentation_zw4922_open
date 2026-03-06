import os
import argparse
from collections import Counter

import numpy as np
import rasterio
from rasterio.transform import Affine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


def parse_args():
    parser = argparse.ArgumentParser("Build connected stitched crops from images_15d tiles.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="test_2",
        help="Dataset directory under project root, or an absolute path.",
    )
    return parser.parse_args()


def resolve_paths(dataset_dir: str):
    if os.path.isabs(dataset_dir):
        root_dir = dataset_dir
    else:
        root_dir = os.path.abspath(os.path.join(PROJECT_DIR, dataset_dir))

    img_dir = os.path.join(root_dir, "images_15d")
    out_dirs = {
        "LR": os.path.join(root_dir, "images_15d_128256"),
        "UD": os.path.join(root_dir, "images_15d_256128"),
        "CORNER": os.path.join(root_dir, "images_15d_256256"),
    }
    return root_dir, img_dir, out_dirs


def resolve_mask_paths(root_dir: str):
    mask_dir = os.path.join(root_dir, "masks")
    out_dirs = {
        "LR": os.path.join(root_dir, "masks_128256"),
        "UD": os.path.join(root_dir, "masks_256128"),
        "CORNER": os.path.join(root_dir, "masks_256256"),
    }
    return mask_dir, out_dirs


def resolve_train_root_for_test(test_root: str):
    parent = os.path.dirname(test_root)
    candidates = [
        os.path.join(parent, "train"),
        os.path.join(PROJECT_DIR, "train"),
        os.path.join(PROJECT_DIR, "data", "train"),
    ]
    for c in candidates:
        img15 = os.path.join(c, "images_15d")
        if os.path.isdir(img15):
            return c
    raise RuntimeError(
        f"dataset={test_root} 需要联动 train 进行裁剪，但未找到可用 train/images_15d。"
    )


def list_tifs(d: str):
    return sorted([f for f in os.listdir(d) if f.lower().endswith(".tif")])


def open_images(img_dir: str):
    names = list_tifs(img_dir)
    if len(names) == 0:
        raise RuntimeError(f"没有在目录中找到 tif: {img_dir}")
    return [(n, os.path.join(img_dir, n)) for n in names]


def open_train_image_mask_pairs(img_dir: str, mask_dir: str):
    img_names = list_tifs(img_dir)
    mask_names = set(list_tifs(mask_dir))
    common = [n for n in img_names if n in mask_names]
    if len(common) == 0:
        raise RuntimeError(f"没有找到 images_15d 与 masks 的同名 tif，img_dir={img_dir}, mask_dir={mask_dir}")
    image_pairs = [(n, os.path.join(img_dir, n)) for n in common]
    mask_pairs = [(n, os.path.join(mask_dir, n)) for n in common]
    return image_pairs, mask_pairs


def with_source(pairs, source: str):
    return [(n, p, source) for n, p in pairs]


def get_union_bounds(bounds_list):
    all_bounds = np.array(bounds_list)  # left,bottom,right,top
    min_x = all_bounds[:, 0].min()
    min_y = all_bounds[:, 1].min()
    max_x = all_bounds[:, 2].max()
    max_y = all_bounds[:, 3].max()
    return min_x, min_y, max_x, max_y


def compute_big_shape(min_x, min_y, max_x, max_y, res_x, res_y):
    width_big = int(np.ceil((max_x - min_x) / res_x))
    height_big = int(np.ceil((max_y - min_y) / res_y))
    return height_big, width_big


def offset_in_big(bounds, min_x, max_y, res_x, res_y):
    left, _, _, top = bounds
    col_off = int(round((left - min_x) / res_x))
    row_off = int(round((max_y - top) / res_y))
    return row_off, col_off


def all_valid(valid_mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h, c0 + w
    if r0 < 0 or c0 < 0 or r1 > valid_mask.shape[0] or c1 > valid_mask.shape[1]:
        return False
    return valid_mask[r0:r1, c0:c1].all()


def patch_transform_from_big(
    r0: int, c0: int, min_x: float, max_y: float, res_x: float, res_y: float
) -> Affine:
    x_left = min_x + c0 * res_x
    y_top = max_y - r0 * res_y
    return Affine(res_x, 0, x_left, 0, -res_y, y_top)


def save_patch_geotiff(path: str, patch: np.ndarray, crs, transform: Affine, nodata=None):
    if patch.ndim == 2:
        patch = patch[None, ...]
    bands, h, w = patch.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": bands,
        "dtype": patch.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(patch)


def atomic_save_patch(path: str, patch: np.ndarray, crs, transform: Affine, nodata=None):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    save_patch_geotiff(tmp, patch, crs, transform, nodata=nodata)
    os.replace(tmp, path)


def build_metadata(image_pairs):
    img_meta = []
    bounds_list = []
    res_x = None
    res_y = None
    crs = None
    img_bands = None
    data_dtype = None
    nodata = None

    for item in image_pairs:
        if len(item) == 2:
            name, p_img = item
            source = "unknown"
        else:
            name, p_img, source = item

        with rasterio.open(p_img) as ds_img:
            if res_x is None:
                res_x = ds_img.transform.a
                res_y = -ds_img.transform.e
                crs = ds_img.crs
                img_bands = ds_img.count
                data_dtype = ds_img.dtypes[0]
                nodata = ds_img.nodata
            bounds = (ds_img.bounds.left, ds_img.bounds.bottom, ds_img.bounds.right, ds_img.bounds.top)
            bounds_list.append(bounds)
            img_meta.append(
                {
                    "name": name,
                    "img_path": p_img,
                    "bounds": bounds,
                    "h": ds_img.height,
                    "w": ds_img.width,
                    "source": source,
                }
            )

    min_x, min_y, max_x, max_y = get_union_bounds(bounds_list)
    h_big, w_big = compute_big_shape(min_x, min_y, max_x, max_y, res_x, res_y)

    tile_infos = []
    for m in img_meta:
        r0, c0 = offset_in_big(m["bounds"], min_x, max_y, res_x, res_y)
        x = dict(m)
        x["r0"] = r0
        x["c0"] = c0
        tile_infos.append(x)

    return {
        "tile_infos": tile_infos,
        "min_x": min_x,
        "max_y": max_y,
        "res_x": res_x,
        "res_y": res_y,
        "h_big": h_big,
        "w_big": w_big,
        "crs": crs,
        "img_bands": img_bands,
        "data_dtype": data_dtype,
        "nodata": nodata,
    }


def build_valid_mask(meta):
    h_big = meta["h_big"]
    w_big = meta["w_big"]
    big_valid = np.zeros((h_big, w_big), dtype=bool)

    for t in meta["tile_infos"]:
        with rasterio.open(t["img_path"]) as ds_img:
            img_valid = ds_img.read_masks(1) > 0
        r0, c0, h, w = t["r0"], t["c0"], t["h"], t["w"]
        r1, c1 = min(r0 + h, h_big), min(c0 + w, w_big)
        ph, pw = r1 - r0, c1 - c0
        if ph <= 0 or pw <= 0:
            continue
        big_valid[r0:r1, c0:c1] |= img_valid[:ph, :pw]

    return big_valid


def build_connected_candidates(tile_infos, valid_mask, required_source=None):
    candidates = {"LR": [], "UD": [], "CORNER": []}
    pos2tile = {}
    for t in tile_infos:
        k = (t["r0"], t["c0"])
        if k not in pos2tile:
            pos2tile[k] = t
        else:
            # Prefer target tile when duplicated positions exist.
            if pos2tile[k].get("source") != "target" and t.get("source") == "target":
                pos2tile[k] = t

    for t in tile_infos:
        r, c, h, w = t["r0"], t["c0"], t["h"], t["w"]
        right_key = (r, c + w)
        down_key = (r + h, c)
        downright_key = (r + h, c + w)

        t_right = pos2tile.get(right_key)
        t_down = pos2tile.get(down_key)
        t_downright = pos2tile.get(downright_key)

        if t_right is not None and (t_right["h"], t_right["w"]) == (h, w):
            ph, pw = h, 2 * w
            srcs = {t.get("source"), t_right.get("source")}
            if (required_source is None or required_source in srcs) and all_valid(valid_mask, r, c, ph, pw):
                candidates["LR"].append(("LR", r, c, ph, pw))

        if t_down is not None and (t_down["h"], t_down["w"]) == (h, w):
            ph, pw = 2 * h, w
            srcs = {t.get("source"), t_down.get("source")}
            if (required_source is None or required_source in srcs) and all_valid(valid_mask, r, c, ph, pw):
                candidates["UD"].append(("UD", r, c, ph, pw))

        if (
            t_right is not None
            and t_down is not None
            and t_downright is not None
            and (t_right["h"], t_right["w"]) == (h, w)
            and (t_down["h"], t_down["w"]) == (h, w)
            and (t_downright["h"], t_downright["w"]) == (h, w)
        ):
            ph, pw = 2 * h, 2 * w
            srcs = {
                t.get("source"),
                t_right.get("source"),
                t_down.get("source"),
                t_downright.get("source"),
            }
            if (required_source is None or required_source in srcs) and all_valid(valid_mask, r, c, ph, pw):
                candidates["CORNER"].append(("CORNER", r, c, ph, pw))

    return candidates


def get_patch_array(tile_infos, r0, c0, ph, pw, bands, cache, out_dtype=np.float32):
    patch = np.zeros((bands, ph, pw), dtype=out_dtype)
    cover = np.zeros((ph, pw), dtype=bool)

    for t in tile_infos:
        tr0, tc0, th, tw = t["r0"], t["c0"], t["h"], t["w"]
        ir0 = max(r0, tr0)
        ic0 = max(c0, tc0)
        ir1 = min(r0 + ph, tr0 + th)
        ic1 = min(c0 + pw, tc0 + tw)
        if ir0 >= ir1 or ic0 >= ic1:
            continue

        path = t["img_path"]
        arr = cache.get(path)
        if arr is None:
            with rasterio.open(path) as ds:
                arr = ds.read().astype(out_dtype)
            cache[path] = arr
            if len(cache) > 512:
                cache.pop(next(iter(cache)))

        src_r0 = ir0 - tr0
        src_c0 = ic0 - tc0
        src_r1 = src_r0 + (ir1 - ir0)
        src_c1 = src_c0 + (ic1 - ic0)

        dst_r0 = ir0 - r0
        dst_c0 = ic0 - c0
        dst_r1 = dst_r0 + (ir1 - ir0)
        dst_c1 = dst_c0 + (ic1 - ic0)

        patch[:, dst_r0:dst_r1, dst_c0:dst_c1] = arr[:, src_r0:src_r1, src_c0:src_c1]
        cover[dst_r0:dst_r1, dst_c0:dst_c1] = True

    if not cover.all():
        raise RuntimeError("Patch extraction failed: uncovered pixels exist.")
    return patch


def cleanup_output_dir(out_dir: str, keep_names):
    keep_set = set(keep_names)
    for name in list_tifs(out_dir):
        if name not in keep_set:
            os.remove(os.path.join(out_dir, name))


def main():
    args = parse_args()
    root_dir, img_dir, out_dirs = resolve_paths(args.dataset_dir)
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(root_dir))
    is_train = dataset_name == "train"
    is_test = dataset_name in ("test", "test_1", "test_2")
    mask_pairs = None
    mask_out_dirs = None
    required_source = None
    if is_train:
        mask_dir, mask_out_dirs = resolve_mask_paths(root_dir)
        for d in mask_out_dirs.values():
            os.makedirs(d, exist_ok=True)
        image_pairs_base, mask_pairs = open_train_image_mask_pairs(img_dir, mask_dir)
        image_pairs = with_source(image_pairs_base, "train")
        print(
            f"Train mode: images_15d & masks synced by name, "
            f"pairs={len(image_pairs)}, mask_dir={mask_dir}"
        )
    elif is_test:
        test_pairs = with_source(open_images(img_dir), "target")
        train_root = resolve_train_root_for_test(root_dir)
        train_img_dir = os.path.join(train_root, "images_15d")
        train_pairs = with_source(open_images(train_img_dir), "train")
        image_pairs = test_pairs + train_pairs
        required_source = "target"
        print(
            f"Test mode: merge target+train for candidates, "
            f"target={len(test_pairs)}, train={len(train_pairs)}, train_root={train_root}"
        )
    else:
        image_pairs = with_source(open_images(img_dir), "target")
        required_source = None

    print(f"Dataset root: {root_dir}")
    print(f"Found {len(image_pairs)} image files.")

    meta = build_metadata(image_pairs)
    tile_infos = meta["tile_infos"]
    print(f"Big mosaic size: H={meta['h_big']}, W={meta['w_big']}")

    mask_meta = None
    if is_train and mask_pairs is not None:
        mask_meta = build_metadata(mask_pairs)
        if not (
            np.isclose(meta["res_x"], mask_meta["res_x"])
            and np.isclose(meta["res_y"], mask_meta["res_y"])
            and meta["h_big"] == mask_meta["h_big"]
            and meta["w_big"] == mask_meta["w_big"]
        ):
            raise RuntimeError("images_15d 与 masks 网格不一致，无法同步裁剪。")

        img_map = {t["name"]: t for t in meta["tile_infos"]}
        msk_map = {t["name"]: t for t in mask_meta["tile_infos"]}
        if set(img_map.keys()) != set(msk_map.keys()):
            raise RuntimeError("images_15d 与 masks 的同名样本集合不一致。")
        for name, it in img_map.items():
            mt = msk_map[name]
            if (it["r0"], it["c0"], it["h"], it["w"]) != (mt["r0"], mt["c0"], mt["h"], mt["w"]):
                raise RuntimeError(f"images_15d 与 masks 空间对齐不一致: {name}")

    shape_counter = Counter((t["h"], t["w"]) for t in tile_infos)
    (base_h, base_w), _ = shape_counter.most_common(1)[0]
    if len(shape_counter) > 1:
        print(f"[warn] multiple tile shapes detected: {dict(shape_counter)}; use base {(base_h, base_w)}")

    print(
        f"Crop shapes: LR=({base_h},{2 * base_w}), "
        f"UD=({2 * base_h},{base_w}), CORNER=({2 * base_h},{2 * base_w})"
    )

    valid_mask = build_valid_mask(meta)
    print("Valid coverage ratio:", float(valid_mask.mean()))

    candidates = build_connected_candidates(tile_infos, valid_mask, required_source=required_source)
    print(
        "Connected candidates: "
        f"LR={len(candidates['LR'])}, UD={len(candidates['UD'])}, CORNER={len(candidates['CORNER'])}"
    )

    image_cache = {}
    mask_cache = {}
    for typ in ("LR", "UD", "CORNER"):
        out_dir = out_dirs[typ]
        mask_out_dir = mask_out_dirs[typ] if (is_train and mask_out_dirs is not None) else None
        keep_names = []

        for idx, (_, r0, c0, ph, pw) in enumerate(candidates[typ]):
            out_name = f"{typ}_{idx:05d}.tif"
            keep_names.append(out_name)
            out_path = os.path.join(out_dir, out_name)

            patch = get_patch_array(
                tile_infos,
                r0,
                c0,
                ph,
                pw,
                bands=meta["img_bands"],
                cache=image_cache,
                out_dtype=np.float32,
            )
            tfm = patch_transform_from_big(
                r0, c0, meta["min_x"], meta["max_y"], meta["res_x"], meta["res_y"]
            )
            atomic_save_patch(out_path, patch.astype(np.float32), meta["crs"], tfm, nodata=None)

            if is_train and mask_meta is not None and mask_out_dir is not None:
                mask_out_path = os.path.join(mask_out_dir, out_name)
                mask_patch = get_patch_array(
                    mask_meta["tile_infos"],
                    r0,
                    c0,
                    ph,
                    pw,
                    bands=mask_meta["img_bands"],
                    cache=mask_cache,
                    out_dtype=np.dtype(mask_meta["data_dtype"]),
                )
                atomic_save_patch(
                    mask_out_path,
                    mask_patch,
                    mask_meta["crs"],
                    tfm,
                    nodata=mask_meta["nodata"],
                )

        cleanup_output_dir(out_dir, keep_names)
        if is_train and mask_out_dir is not None:
            cleanup_output_dir(mask_out_dir, keep_names)
            print(f"{typ} masks saved: {len(keep_names)} -> {mask_out_dir}")
        print(f"{typ} saved: {len(keep_names)} -> {out_dir}")


if __name__ == "__main__":
    main()
