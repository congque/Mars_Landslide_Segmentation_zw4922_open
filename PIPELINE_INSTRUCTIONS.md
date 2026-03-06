# Mars Open End-to-End Instructions

This document describes the full pipeline from raw data to final submission masks.

## 0) Required folder layout

Under project root:

```text
data/
  train/
    images/
    masks/
  test_2/
    images/
```

## 1) Environment

```bash
cd /gpfsnyu/home/zw4922/projs/mars_open
source /gpfsnyu/packages/anaconda3/22py10/etc/profile.d/conda.sh
conda activate zw4922_312
```

## 2) Add terrain bands (7 -> 15)

Script: `process/add_band_15d.py`

```bash
python process/add_band_15d.py --input-dir ./data/train/images  --output-dir ./data/train/images_15d
python process/add_band_15d.py --input-dir ./data/test_2/images --output-dir ./data/test_2/images_15d
```

## 3) Build stitched crops

Script: `process/mid_crop.py`

```bash
python process/mid_crop.py --dataset-dir data/train
python process/mid_crop.py --dataset-dir data/test_2
```

After this step, expected folders include:

```text
data/train/images_15d_128256
data/train/images_15d_256128
data/train/images_15d_256256
data/train/masks_128256
data/train/masks_256128
data/train/masks_256256
data/test_2/images_15d_128256
data/test_2/images_15d_256128
data/test_2/images_15d_256256
```

## 4) Train 4 experts for one backbone

Script: `train_expert_full.py`

Example backbone: `uconvnextb`

```bash
MODEL=uconvnextb
SAVE=./pths_expert_${MODEL}
mkdir -p ${SAVE}

python train_expert_full.py --expert 128128 --model ${MODEL} --epochs 147 --seed 3400 --save-dir ${SAVE}
python train_expert_full.py --expert 128256 --model ${MODEL} --epochs 147 --seed 3401 --save-dir ${SAVE}
python train_expert_full.py --expert 256128 --model ${MODEL} --epochs 147 --seed 3402 --save-dir ${SAVE}
python train_expert_full.py --expert 256256 --model ${MODEL} --epochs 147 --seed 3403 --save-dir ${SAVE}
```

The script saves files named:

```text
expert_<expert>_<model>_full.pth
```

## 5) Run prediction for one model

Script: `predict_expert_fused_geo.py`

```bash
python predict_expert_fused_geo.py \
  --test-root ./data/test_2 \
  --pths-dir ./pths_expert_uconvnextb \
  --model uconvnextb \
  --save-dir ./data/test_2/fused_expert_pred_uconvnextb \
  --no-export-sample-mask
```

Output folder contains:

```text
fused_prob.tif
fused_valid_mask.tif
fused_pred_mask.tif
```

Repeat step 4-5 for more backbones/runs if you want ensemble fusion.

## 6) Ensemble fusion + post-process + export sample masks

Script: `fuse_post_export_pipeline.py`

This script does all of the following:
- read multiple prediction folders
- fuse probability maps (valid-pixel weighted)
- run `process/post_process.py` with threshold and post params
- export per-sample binary masks under `pred_mask_post/`
- create a submission zip

```bash
python fuse_post_export_pipeline.py \
  --test-root ./data/test_2 \
  --pred-dirs fused_expert_pred_uconvnextb fused_expert_pred_uconvnexts fused_expert_pred_ueffb4 fused_expert_pred_umshd \
  --out-dir ./data/test_2/fused_ensemble_post \
  --threshold 0.60
```

If you only have one model output:

```bash
python fuse_post_export_pipeline.py \
  --test-root ./data/test_2 \
  --pred-dirs fused_expert_pred_uconvnextb \
  --out-dir ./data/test_2/fused_ensemble_post \
  --threshold 0.60
```

## 7) Final outputs

In `--out-dir`:

```text
fused_prob_ensemble.tif
fused_valid_mask_ensemble.tif
fused_prob_post.tif
fused_pred_mask_post.tif
pred_mask_post/*.tif
submission_pred_mask_post.zip
```

`submission_pred_mask_post.zip` is the final submission package.
