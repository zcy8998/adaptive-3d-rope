# Adaptive 3D-RoPE: Physics-Aligned Rotary Positional Encoding for Wireless Foundation Models

This repository is the streamlined open-source release for the Adaptive 3D-RoPE project. It keeps the core code needed to train, evaluate, and few-shot fine-tune CSI foundation models, while removing paper-only plotting and visualization utilities.

## Method Overview

The codebase provides one unified CSI MAE model with configurable positional encoding:

- `rope_mode=none`: standard attention without RoPE
- `rope_mode=learnable`: learnable mixed 3D-RoPE
- `rope_mode=fixed`: fixed separable 3D-RoPE
- `rope_mode=adaptive`: Adaptive 3D-RoPE with a dynamic controller

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

## Expected Data Layout

The loaders expect:

```text
DATA_DIR/
  D1/
    train_data.mat
    val_data.mat
    test_data.mat
    config.mat
  D2/
    ...
```

Each `.mat` file should contain `H_train`, `H_val`, or `H_test`.

## Core Commands

Train:

```bash
python cli.py train \
  --model csi_mae_base \
  --rope_mode adaptive \
  --encoder_pe complex_rotation \
  --decoder_pe sincos_3d \
  --dataset D1,D2,D3,D4 \
  --data_dir /path/to/csidata \
  --mask_type random \
  --mask_ratio 0.85 \
  --batch_size 32 \
  --epochs 150 \
  --output_dir ./outputs/train
```

Evaluate:

```bash
python cli.py eval \
  --resume /path/to/checkpoint.pth \
  --model csi_mae_base \
  --rope_mode adaptive \
  --encoder_pe complex_rotation \
  --decoder_pe sincos_3d \
  --dataset D1,D2,D3,D4 \
  --data_dir /path/to/csidata \
  --mask_type random \
  --mask_ratio 0.85 \
  --output_dir ./outputs/eval
```

Few-shot fine-tune:

```bash
python cli.py finetune \
  --finetune /path/to/source_checkpoint.pth \
  --model csi_mae_base \
  --rope_mode adaptive \
  --encoder_pe complex_rotation \
  --decoder_pe sincos_3d \
  --dataset D17 \
  --data_dir /path/to/csidata \
  --data_num 0.1 \
  --batch_size 32 \
  --epochs 20 \
  --output_dir ./outputs/finetune
```

`finetune` can also start from scratch if `--finetune` is omitted, but the recommended workflow is to initialize from a source-domain checkpoint.

## Key Arguments

- `--model`: `csi_mae_base`, `csi_mae_small`, `csi_mae_tiny`
- `--rope_mode`: `none`, `learnable`, `fixed`, `adaptive`
- `--encoder_pe`: `complex_rotation`, `complex_rotation_3d`, `trivial`, `sincos`, `sincos_2d`, `sincos_3d`, `sincos_1d_3`, `none`
- `--decoder_pe`: same choices as `--encoder_pe`
- `--rope_theta`: single value like `10` or triplet like `10,100,1000`
- `--rope_use_ape`: enable absolute positional encoding alongside RoPE
- `--use_phys_coord`: load dataset-level physical metadata when available

## Outputs

- Training writes checkpoints to `output_dir/checkpoint-*.pth`
- The final training checkpoint is `output_dir/checkpoint-final.pth`
- Training metrics are appended to `output_dir/train_metrics.csv`
- Evaluation writes `output_dir/eval_metrics.json` and `output_dir/eval_summary.txt`

## License

This project is released under the MIT License. See `LICENSE`.

## Citation

