import argparse
import os
from pathlib import Path
import zipfile


def collect_mask_files(mask_dir: Path):
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(mask_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    return files


def build_args():
    parser = argparse.ArgumentParser(description="Zip final predicted masks only.")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="./data/test_2/fused_expert_pred/pred_mask",
        help="Directory containing final mask files.",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default="",
        help="Output zip path. Default: <mask_dir>/../pred_mask.zip",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Store only file names in zip (recommended for submission).",
    )
    return parser.parse_args()


def main():
    args = build_args()
    mask_dir = Path(args.mask_dir).resolve()
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    files = collect_mask_files(mask_dir)
    if len(files) == 0:
        raise RuntimeError(f"No mask files found in: {mask_dir}")

    if args.zip_path.strip():
        zip_path = Path(args.zip_path).resolve()
    else:
        zip_path = (mask_dir.parent / "pred_mask.zip").resolve()

    os.makedirs(zip_path.parent, exist_ok=True)

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for p in files:
            arcname = p.name if args.flatten else str(p.relative_to(mask_dir.parent))
            zf.write(p, arcname=arcname)

    size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved zip: {zip_path}")
    print(f"Files packed: {len(files)}")
    print(f"Zip size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
