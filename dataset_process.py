import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import tifffile as tiff
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

CHANNEL_MAXS_15 = [14.39, 5.93, 6143.00, 5.42, 5.92, 6.14, 6.34, 6.45, 27.04, 14.29, 8.41, 32.49, 16.73, 1.99, 1.99]
CHANNEL_MINS_15 = [8.00, 0.74, -5880.00, 4.55, 3.36, 2.62, 2.62, -6.12, 3.64, 1.90, -7.91, 4.58, 2.29, 0.03, 0.08]

def _list_tif_names(dir_path):
    return sorted([f for f in os.listdir(dir_path) if f.endswith('.tif')])

def _build_paired_files(input_dir, label_dir):
    if isinstance(input_dir, str):
        input_dirs = [input_dir]
        label_dirs = [label_dir]
    else:
        if len(input_dir) != len(label_dir):
            raise ValueError("input_dir 和 label_dir 列表长度不一致。")
        input_dirs = list(input_dir)
        label_dirs = list(label_dir)

    file_names = []
    label_files = []

    for idir, ldir in zip(input_dirs, label_dirs):
        input_names = set(_list_tif_names(idir))
        label_names = set(_list_tif_names(ldir))

        missing_label = sorted(input_names - label_names)
        missing_input = sorted(label_names - input_names)
        if missing_label or missing_input:
            msg = (
                f"输入和标签文件名不一致：{idir} vs {ldir}; "
                f"输入缺失{len(missing_input)}个, 标签缺失{len(missing_label)}个。"
            )
            raise RuntimeError(msg)

        common = sorted(input_names & label_names)
        if len(common) == 0:
            raise RuntimeError(f"目录 {idir} 与 {ldir} 没有同名 tif。")

        file_names.extend([os.path.join(idir, f) for f in common])
        label_files.extend([os.path.join(ldir, f) for f in common])

    return file_names, label_files

def _parse_stitch_bins(stitch, num_bins=7):
    if stitch is None or stitch is False or stitch == "":
        return set()
    if isinstance(stitch, bool):
        return set()

    bins = []
    if isinstance(stitch, int):
        bins = [stitch]
    elif isinstance(stitch, (list, tuple, set, np.ndarray)):
        bins = [int(v) for v in stitch]
    elif isinstance(stitch, str):
        tokens = [tok for tok in re.split(r"[,\s;，]+", stitch.strip()) if tok]
        for tok in tokens:
            if "-" in tok:
                parts = tok.split("-", 1)
                if len(parts) != 2:
                    raise ValueError(f"stitch 格式错误: {tok}")
                start = int(parts[0])
                end = int(parts[1])
                lo, hi = min(start, end), max(start, end)
                bins.extend(list(range(lo, hi + 1)))
            else:
                bins.append(int(tok))
    else:
        raise TypeError("stitch 仅支持 bool/int/str/list/tuple/set。")

    invalid = sorted(set([b for b in bins if b < 0 or b >= num_bins]))
    if invalid:
        raise ValueError(f"stitch 列编号越界: {invalid}，合法范围是 [0, {num_bins - 1}]。")

    return set(bins)

def _center_x_from_tif(path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        return (b.left + b.right) / 2.0

def _apply_stitch_filter(file_names, label_files, stitch, num_bins=7):
    drop_bins = _parse_stitch_bins(stitch, num_bins=num_bins)
    if not drop_bins:
        return file_names, label_files

    center_xs = np.array([_center_x_from_tif(p) for p in file_names], dtype=np.float64)
    x_min = float(center_xs.min())
    x_max = float(center_xs.max())

    if np.isclose(x_max, x_min):
        bin_ids = np.zeros(len(center_xs), dtype=np.int64)
    else:
        scaled = (center_xs - x_min) / (x_max - x_min)
        bin_ids = np.floor(scaled * num_bins).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    keep_mask = np.array([b not in drop_bins for b in bin_ids], dtype=bool)
    kept_file_names = [p for p, k in zip(file_names, keep_mask) if k]
    kept_label_files = [p for p, k in zip(label_files, keep_mask) if k]
    removed = int((~keep_mask).sum())

    if len(kept_file_names) == 0:
        raise RuntimeError(f"stitch={sorted(drop_bins)} 过滤后无样本，请调整要删除的列。")

    print(
        f"[stitch] num_bins={num_bins}, drop={sorted(drop_bins)}, "
        f"x_range=({x_min:.3f}, {x_max:.3f}), removed={removed}, kept={len(kept_file_names)}"
    )
    return kept_file_names, kept_label_files

def _prepare_dataset_files(input_dir, label_dir, stitch, num_bins=7):
    file_names, label_files = _build_paired_files(input_dir, label_dir)
    return _apply_stitch_filter(file_names, label_files, stitch=stitch, num_bins=num_bins)

def gamma_transform(img, gamma_range=(0.87, 1.13)):
    C = img.shape[0]
    gammas = np.random.uniform(gamma_range[0], gamma_range[1], size=(C, 1, 1))
    img = np.power(img, gammas)
    return np.clip(img, 0, 1).copy()

def random_brightness_contrast(img, brightness=0.1, contrast=0.1):
    C = img.shape[0]
    alpha = 1.0 + np.random.uniform(-contrast, contrast, size=(C, 1, 1))
    beta  = np.random.uniform(-brightness, brightness, size=(C, 1, 1))
    img = alpha * img + beta
    return np.clip(img, 0, 1).copy()

def geom_aug_chw(input_img, label_img):
    t = np.random.randint(0, 8)

    # 0~3: 纯旋转
    if t < 4:
        k = t
        if k != 0:
            input_img = np.rot90(input_img, k=k, axes=(1, 2)).copy()
            label_img = np.rot90(label_img, k=k, axes=(1, 2)).copy()

    # 4~7: 先左右翻转，再旋转
    else:
        # 左右翻转（W 维）
        input_img = input_img[:, :, ::-1]
        label_img = label_img[:, :, ::-1]

        k = t - 4
        if k != 0:
            input_img = np.rot90(input_img, k=k, axes=(1, 2)).copy()
            label_img = np.rot90(label_img, k=k, axes=(1, 2)).copy()

        input_img = input_img.copy()
        label_img = label_img.copy()

    return input_img, label_img

def geom_aug_chw_d4(input_img, label_img):
    t = np.random.randint(0, 4)

    if t == 1:  # 左右翻转
        input_img = input_img[:, :, ::-1]
        label_img = label_img[:, :, ::-1]

    elif t == 2:  # 上下翻转
        input_img = input_img[:, ::-1, :]
        label_img = label_img[:, ::-1, :]

    elif t == 3:  # 旋转180
        input_img = np.rot90(input_img, k=2, axes=(1, 2))
        label_img = np.rot90(label_img, k=2, axes=(1, 2))

    return input_img.copy(), label_img.copy()

def compute_boundry(label):

    if not isinstance(label, torch.Tensor):
        label = torch.from_numpy(label)

    label = label.float()

    label = label.unsqueeze(0)

    max_pool = F.max_pool2d(label, kernel_size=7, stride=1, padding=3)
    min_pool = -F.max_pool2d(-label, kernel_size=7, stride=1, padding=3)

    boundary = (max_pool != min_pool).float()
    boundary = boundary.squeeze(0)

    return boundary.numpy()

# Train
class TifSegDataset_15d(Dataset):

    def __init__(self, input_dir, label_dir, boundary = False, band = 15, stitch = False, spatial_size=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.band = band
        self.boundary = boundary
        self.stitch = stitch
        _ = spatial_size  # kept in signature for backward compatibility

        self.file_names, self.label_files = _prepare_dataset_files(
            input_dir=input_dir,
            label_dir=label_dir,
            stitch=stitch,
            num_bins=7
        )

        assert len(self.file_names) == len(self.label_files), "输入和标签文件数量不匹配！"

    def __len__(self):
        return len(self.file_names)

    def stitch(self):
        pass

    def __getitem__(self, idx):

        # -------------------------
        # 读取（HWC）
        # -------------------------
        input_img = tiff.imread(self.file_names[idx])  # (H,W,C)
        input_img[:, :, 1][input_img[:, :, 1] < 0] = 0

        label_img = tiff.imread(self.label_files[idx])
        label_img = (label_img > 0.5).astype(np.float32)


        dem = input_img[:, :, 2].astype(np.float32)
        m1 = float(dem.mean())
        s1 = float(dem.std())
        if s1 < 1e-6:
            b3_z_raw = np.zeros_like(dem, dtype=np.float32)
        else:
            b3_z_raw = (dem - m1) / (s1 + 1e-6)   # (H,W)

        q25 = np.percentile(dem, 25)
        q75 = np.percentile(dem, 75)
        iqr = q75 - q25 + 1e-6
        dem_robust = (dem - np.median(dem)) / iqr

        # -------------------------
        # 你原来的 STD/minmax/clip（只对前15个原始通道做）
        # -------------------------
        if self.band == 15:
            for c in range(self.band):
                if c in [0, 1, 8, 9, 11, 12]:
                    input_img[:, :, c] = np.sqrt(input_img[:, :, c] + 0.0001)
                elif c in [3, 4, 5, 6]:
                    input_img[:, :, c] = np.pow(input_img[:, :, c] + 0.0001, 1/3)
                elif c in [7, 10]:
                    input_img[:, :, c] = np.sign(input_img[:, :, c]) * np.sqrt(np.abs(input_img[:, :, c]) + 0.0001)

                input_img[:, :, c] = (input_img[:, :, c] - CHANNEL_MINS_15[c]) / (CHANNEL_MAXS_15[c] - CHANNEL_MINS_15[c])
                input_img[:, :, c] = np.clip(input_img[:, :, c], 0, 1)

        # -------------------------
        # HWC → CHW
        # -------------------------
        input_img = np.transpose(input_img, (2, 0, 1)).astype(np.float32)

        label_img = np.expand_dims(label_img, axis=-1)
        label_img = np.transpose(label_img, (2, 0, 1)).astype(np.float32)

        # -------------------------
        # append 16th + 17th（输出17通道）
        # -------------------------
        input_img = np.concatenate(
            [input_img, b3_z_raw[None, :, :], dem_robust[None, :, :]],
            axis=0
        )  # (17,H,W)

        if self.boundary:
            boundry_img = compute_boundry(label_img)
            return (
                torch.from_numpy(input_img).float(),
                torch.from_numpy(label_img).float(),
                torch.from_numpy(boundry_img).float()
            )

        return torch.from_numpy(input_img).float(), torch.from_numpy(label_img).float()


class TifSegDataset_15d_on_the_fly(Dataset):

    def __init__(self, input_dir, label_dir, boundary=False, band=15, stitch=False):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.band = band
        self.boundary = boundary
        self.stitch = stitch

        self.file_names, self.label_files = _prepare_dataset_files(
            input_dir=input_dir,
            label_dir=label_dir,
            stitch=stitch,
            num_bins=7
        )

        assert len(self.file_names) == len(self.label_files), "输入和标签文件数量不匹配！"

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        # -------------------------
        # 读取（HWC）
        # -------------------------
        input_img = tiff.imread(self.file_names[idx])  # (H,W,C)
        input_img[:, :, 1][input_img[:, :, 1] < 0] = 0

        label_img = tiff.imread(self.label_files[idx])
        label_img = (label_img > 0.5).astype(np.float32)

        # =========================================================
        # DEM (band index 2) 原始域处理：delta + 两个派生通道（最后输出17通道）
        #   - 通道16: b3_z_raw  = (dem - mean)/std
        #   - 通道17: dem_med_z = ((dem - median) - mean)/std
        # 注意：这两条通道都在“原始值域”（minmax 前）计算
        # =========================================================
        delta = np.random.uniform(-200.0, 200.0)
        dem = input_img[:, :, 2].astype(np.float32) + delta
        input_img[:, :, 2] = dem  # 写回去：后续 minmax 用的是扰动后的 DEM

        m1 = float(dem.mean())
        s1 = float(dem.std())
        if s1 < 1e-6:
            b3_z_raw = np.zeros_like(dem, dtype=np.float32)
        else:
            b3_z_raw = (dem - m1) / (s1 + 1e-6)   # (H,W)

        q25 = np.percentile(dem, 25)
        q75 = np.percentile(dem, 75)
        iqr = q75 - q25 + 1e-6
        dem_robust = (dem - np.median(dem)) / iqr

        # -------------------------
        # 你原来的 STD/minmax/clip（只对前15个原始通道做）
        # -------------------------
        if self.band == 15:
            for c in range(self.band):
                if c in [0, 1, 8, 9, 11, 12]:
                    input_img[:, :, c] = np.sqrt(input_img[:, :, c] + 0.0001)
                elif c in [3, 4, 5, 6]:
                    input_img[:, :, c] = np.pow(input_img[:, :, c] + 0.0001, 1/3)
                elif c in [7, 10]:
                    input_img[:, :, c] = np.sign(input_img[:, :, c]) * np.sqrt(np.abs(input_img[:, :, c]) + 0.0001)

                input_img[:, :, c] = (input_img[:, :, c] - CHANNEL_MINS_15[c]) / (CHANNEL_MAXS_15[c] - CHANNEL_MINS_15[c])
                input_img[:, :, c] = np.clip(input_img[:, :, c], 0, 1)

        # -------------------------
        # HWC → CHW
        # -------------------------
        input_img = np.transpose(input_img, (2, 0, 1)).astype(np.float32)

        label_img = np.expand_dims(label_img, axis=-1)
        label_img = np.transpose(label_img, (2, 0, 1)).astype(np.float32)

        # -------------------------
        # append 16th + 17th（输出17通道）
        # -------------------------
        input_img = np.concatenate(
            [input_img, b3_z_raw[None, :, :], dem_robust[None, :, :]],
            axis=0
        )  # (17,H,W)

        # ===== on the fly geom aug（同步增强）=====
        input_img, label_img = geom_aug_chw(input_img, label_img)

        # ===== on the fly value aug（只增强 input 的前15通道；不动新增DEM派生通道）=====
        if np.random.rand() < 0.8:
            tmp = input_img[:15]
            tmp = gamma_transform(tmp)
            input_img[:15] = tmp

        if np.random.rand() < 0.8:
            tmp = input_img[:15]
            tmp = random_brightness_contrast(tmp)
            input_img[:15] = tmp

        if self.boundary:
            boundry_img = compute_boundry(label_img)
            return (
                torch.from_numpy(input_img).float(),
                torch.from_numpy(label_img).float(),
                torch.from_numpy(boundry_img).float()
            )

        return torch.from_numpy(input_img).float(), torch.from_numpy(label_img).float()


class TifSegDataset_15d_on_the_fly_4d(Dataset):

    def __init__(self, input_dir, label_dir, boundary=False, band=15, stitch=False):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.band = band
        self.boundary = boundary
        self.stitch = stitch

        self.file_names, self.label_files = _prepare_dataset_files(
            input_dir=input_dir,
            label_dir=label_dir,
            stitch=stitch,
            num_bins=7
        )

        assert len(self.file_names) == len(self.label_files), "输入和标签文件数量不匹配！"

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        # -------------------------
        # 读取（HWC）
        # -------------------------
        input_img = tiff.imread(self.file_names[idx])  # (H,W,C)
        input_img[:, :, 1][input_img[:, :, 1] < 0] = 0

        label_img = tiff.imread(self.label_files[idx])
        label_img = (label_img > 0.5).astype(np.float32)

        # =========================================================
        # DEM (band index 2) 原始域处理：delta + 两个派生通道（最后输出17通道）
        #   - 通道16: b3_z_raw  = (dem - mean)/std
        #   - 通道17: dem_med_z = ((dem - median) - mean)/std
        # 注意：这两条通道都在“原始值域”（minmax 前）计算
        # =========================================================
        delta = np.random.uniform(-200.0, 200.0)
        dem = input_img[:, :, 2].astype(np.float32) + delta
        input_img[:, :, 2] = dem  # 写回去：后续 minmax 用的是扰动后的 DEM

        m1 = float(dem.mean())
        s1 = float(dem.std())
        if s1 < 1e-6:
            b3_z_raw = np.zeros_like(dem, dtype=np.float32)
        else:
            b3_z_raw = (dem - m1) / (s1 + 1e-6)   # (H,W)

        q25 = np.percentile(dem, 25)
        q75 = np.percentile(dem, 75)
        iqr = q75 - q25 + 1e-6
        dem_robust = (dem - np.median(dem)) / iqr

        # -------------------------
        # 你原来的 STD/minmax/clip（只对前15个原始通道做）
        # -------------------------
        if self.band == 15:
            for c in range(self.band):
                if c in [0, 1, 8, 9, 11, 12]:
                    input_img[:, :, c] = np.sqrt(input_img[:, :, c] + 0.0001)
                elif c in [3, 4, 5, 6]:
                    input_img[:, :, c] = np.pow(input_img[:, :, c] + 0.0001, 1/3)
                elif c in [7, 10]:
                    input_img[:, :, c] = np.sign(input_img[:, :, c]) * np.sqrt(np.abs(input_img[:, :, c]) + 0.0001)

                input_img[:, :, c] = (input_img[:, :, c] - CHANNEL_MINS_15[c]) / (CHANNEL_MAXS_15[c] - CHANNEL_MINS_15[c])
                input_img[:, :, c] = np.clip(input_img[:, :, c], 0, 1)

        # -------------------------
        # HWC → CHW
        # -------------------------
        input_img = np.transpose(input_img, (2, 0, 1)).astype(np.float32)

        label_img = np.expand_dims(label_img, axis=-1)
        label_img = np.transpose(label_img, (2, 0, 1)).astype(np.float32)

        # -------------------------
        # append 16th + 17th（输出17通道）
        # -------------------------
        input_img = np.concatenate(
            [input_img, b3_z_raw[None, :, :], dem_robust[None, :, :]],
            axis=0
        )  # (17,H,W)

        # ===== on the fly geom aug（同步增强）=====
        input_img, label_img = geom_aug_chw_d4(input_img, label_img)

        # ===== on the fly value aug（只增强 input 的前15通道；不动新增DEM派生通道）=====
        if np.random.rand() < 0.8:
            tmp = input_img[:15]
            tmp = gamma_transform(tmp)
            input_img[:15] = tmp

        if np.random.rand() < 0.8:
            tmp = input_img[:15]
            tmp = random_brightness_contrast(tmp)
            input_img[:15] = tmp

        if self.boundary:
            boundry_img = compute_boundry(label_img)
            return (
                torch.from_numpy(input_img).float(),
                torch.from_numpy(label_img).float(),
                torch.from_numpy(boundry_img).float()
            )

        return torch.from_numpy(input_img).float(), torch.from_numpy(label_img).float()


# Predict
class TifSegDataset_15d_predict(Dataset):

    def __init__(self, input_dir, band):
        self.input_dir = input_dir
        self.band = band
        self.file_names = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        input_path = os.path.join(self.input_dir, self.file_names[idx])
        input_img = tiff.imread(input_path)  # shape: (128, 128, 13)
        input_img[:, :, 1][input_img[:, :, 1]<0] = 0

        # STD
        if self.band == 15:
            for c in range(self.band):
                if c in [0,1,8,9,11,12]:
                    input_img[:, :, c] = np.sqrt(input_img[:, :, c]+0.0001)
                elif c in [3,4,5,6]:
                    input_img[:, :, c] = np.pow(input_img[:, :, c]+0.0001,1/3)
                elif c in [7,10]:
                    input_img[:, :, c] = np.sign(input_img[:, :, c])*np.sqrt(np.abs(input_img[:, :, c])+0.0001)
                input_img[:, :, c] = (input_img[:, :, c] - CHANNEL_MINS_15[c]) / (CHANNEL_MAXS_15[c] - CHANNEL_MINS_15[c])
                input_img[:, :, c] = np.clip(input_img[:, :, c], 0, 1)
        
        input_img = np.transpose(input_img, (2, 0, 1)).astype(np.float32)

        return torch.from_numpy(input_img).float(), self.file_names[idx]


if __name__ == "__main__":
    input_dir = "./data/train/images_crop_15d"
