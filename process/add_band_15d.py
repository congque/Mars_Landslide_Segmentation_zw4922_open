import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import Affine

def _mean_filter(x2d: np.ndarray, k: int) -> np.ndarray:
    """k×k 均值滤波，reflect padding。"""
    pad = k // 2
    xp = np.pad(x2d, ((pad, pad), (pad, pad)), mode="reflect")
    w = np.lib.stride_tricks.sliding_window_view(xp, (k, k))  # (H,W,k,k)
    return w.mean(axis=(-1, -2)).astype(np.float32)

def _local_shape_stats(x2d: np.ndarray, k: int, exclude_center: bool = True):
    """
    k×k 窗口统计（reflect padding）：
      diff: center - mean(neighbors)  # 凸/凹信号
      rng : max - min
      std : std(window)
    返回：diff, rng, std  (H,W) float32
    """
    pad = k // 2
    xp = np.pad(x2d, ((pad, pad), (pad, pad)), mode="reflect")
    w = np.lib.stride_tricks.sliding_window_view(xp, (k, k))  # (H,W,k,k)

    center = w[..., pad, pad]  # (H,W)

    if exclude_center:
        s = w.sum(axis=(-1, -2)) - center
        n = k * k - 1
        mean_nb = s / n
    else:
        mean_nb = w.mean(axis=(-1, -2))

    diff = center - mean_nb
    rng = w.max(axis=(-1, -2)) - w.min(axis=(-1, -2))
    std = w.std(axis=(-1, -2))

    return diff.astype(np.float32), rng.astype(np.float32), std.astype(np.float32)

def _aspect_cos_sin(dem: np.ndarray, transform: Affine, eps: float = 1e-6):
    """
    aspect 定义：0=北，顺时针。返回 cos/sin(aspect)。
    """
    dx = float(transform.a)
    dy = float(abs(transform.e))
    if dx <= 0 or dy <= 0:
        raise ValueError(f"Bad pixel size from transform: dx={dx}, dy={dy}")

    d_dem_dy, d_dem_dx = np.gradient(dem, dy, dx)

    aspect = np.arctan2(d_dem_dx, -d_dem_dy)
    aspect = np.mod(aspect, 2 * np.pi)

    cos_a = np.cos(aspect).astype(np.float32)
    sin_a = np.sin(aspect).astype(np.float32)

    g2 = d_dem_dx * d_dem_dx + d_dem_dy * d_dem_dy
    flat = g2 < (eps * eps)
    cos_a[flat] = 0.0
    sin_a[flat] = 0.0
    return cos_a, sin_a

def _local_range(x2d: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x2d, ((pad, pad), (pad, pad)), mode="reflect")
    w = np.lib.stride_tricks.sliding_window_view(xp, (k, k))
    return (w.max(axis=(-1, -2)) - w.min(axis=(-1, -2))).astype(np.float32)

def add_band(tif_data_chw: np.ndarray, transform: Affine, dem_band: int = 2) -> np.ndarray:
    """
    (1) 先对 DEM 做 3×3 均值平滑，再计算所有地形因子
    (2) 坡向用 5×5 尺度更稳定：对 dem_s 再做 5×5 均值平滑后求梯度
    输出：(C+8,H,W)
    """
    if tif_data_chw.ndim != 3:
        raise ValueError(f"Expect (C,H,W), got {tif_data_chw.shape}")
    C, H, W = tif_data_chw.shape
    if not (0 <= dem_band < C):
        raise ValueError(f"dem_band={dem_band} out of range for C={C}")

    # 原始 DEM
    dem = tif_data_chw[dem_band].astype(np.float32)

    # (1) 3×3 均值平滑（所有地形因子之前）
    dem_s = _mean_filter(dem, 3)

    # 用 dem_s 计算原来的 3×3 / 7×7 统计特征
    b1, b2, b3 = _local_shape_stats(dem_s, 5, exclude_center=True)
    b4, b5, b6 = _local_shape_stats(dem_s, 7, exclude_center=True)

    # (2) 坡向稳定性：先用 dem_s 算 aspect cos/sin，再在 5×5 里算 std
    # 如果你还想先对 dem_s 再做 5×5 平滑再算坡向，也可以，把 dem_s 换成 _mean_filter(dem_s, 5)
    cos_a, sin_a = _aspect_cos_sin(dem_s, transform)
    std_cos = _local_range(cos_a, 5)
    std_sin = _local_range(sin_a, 5)

    new_bands = np.stack([b1, b2, b3, b4, b5, b6, std_cos, std_sin], axis=0).astype(np.float32)
    out = np.concatenate([tif_data_chw.astype(np.float32), new_bands], axis=0)
    return out

def load_tifs_rasterio_geotiff(tif_dir: str, tif_dir_new: str, dem_band: int = 2):
    os.makedirs(tif_dir_new, exist_ok=True)

    tif_files = sorted([
        f for f in os.listdir(tif_dir)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])

    for tif_name in tif_files:
        in_path = os.path.join(tif_dir, tif_name)
        out_path = os.path.join(tif_dir_new, tif_name)

        try:
            with rasterio.open(in_path) as src:
                data = src.read().astype(np.float32)  # (C,H,W)

                data_enh = add_band(data, src.transform, dem_band=dem_band)  # (C+8,H,W)

                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=data_enh.shape[0],
                    dtype=rasterio.float32,
                    nodata=None,
                    compress="deflate",
                    predictor=3,
                    tiled=True,
                    blockxsize=min(128, src.width),
                    blockysize=min(128, src.height),
                )

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data_enh)

        except Exception as e:
            print(f"[WARNING] Failed on {in_path}: {e}")


def parse_args():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.abspath(os.path.join(base_dir, "..", "data", "test_2", "images"))
    default_out = os.path.abspath(os.path.join(base_dir, "..", "data", "test_2", "images_15d"))

    parser = argparse.ArgumentParser("Add 8 terrain-derived bands to tif tiles.")
    parser.add_argument("--input-dir", type=str, default=default_in, help="Input tif directory.")
    parser.add_argument("--output-dir", type=str, default=default_out, help="Output tif directory.")
    parser.add_argument("--dem-band", type=int, default=2, help="DEM band index (0-based).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_tifs_rasterio_geotiff(args.input_dir, args.output_dir, dem_band=args.dem_band)
