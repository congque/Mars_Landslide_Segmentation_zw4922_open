Official code release of Team New York Martians for the PBVS 2026 Mars Landslide Segmentation Challenge.

## Overview
Our solution is a spatial-continuity-aware segmentation pipeline with four expert models for different crop scales, followed by geospatial fusion and post-processing.

## Repository Structure
- `process/`: preprocessing scripts
- `process/post_process.py`: advanced post-processing on fused probability maps
- `train_expert_full.py`: train expert models
- `predict_expert_fused_geo.py`: inference and geospatial fusion
- `fuse_post_export_pipeline.py`: ensemble, post-processing, and submission export

## Setup
See `requirements.txt` and `PIPELINE_INSTRUCTIONS.md`.

## Team
New York Martians  
New York University
