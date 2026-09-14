# Adaptive 3D-RoPE

Minimal reference implementation of Adaptive 3D-RoPE for masked 3D CSI
reconstruction. The public interface is locked to the released pure-RoPE,
Std-only controller configuration:

- `encoder_pe=none`, `decoder_pe=none`
- `controller_token_groups=token_std`
- `controller_architecture=token_only`
- `controller_decoder_token_scope=visible_only`

![Adaptive 3D-RoPE framework](assets/framework.png)

## Install

Use Python 3.11. Install the PyTorch wheel appropriate for the local CPU/CUDA
platform, then install the remaining pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

CSI tensors are external to this repository. The CLI expects one directory per
dataset, each containing MATLAB v7.3/HDF5 tensors and its physical
configuration:

```text
DATA_ROOT/
  D1/
    train_data.mat  # H_train
    val_data.mat    # H_val
    test_data.mat   # H_test
    config.mat
```

Each CSI tensor has shape `(U, K, T, B)` and all spatial dimensions must be
divisible by the patch size (4). Use the same layout for D2, D3, and additional
datasets.

## Train And Evaluate

The release configuration is encoded by the CLI; no controller or positional
arguments are required.

```bash
python cli.py train \
  --dataset D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16 \
  --data_dir DATA_ROOT --mask_type all --mask_ratio 0.75 \
  --batch_size 64 --epochs 150 --num_workers 8 --seed 42 --snr_db 20 \
  --output_dir OUTPUT_ROOT/std_only_pretrain

python cli.py eval \
  --dataset D1 --data_dir DATA_ROOT --mask_type random --mask_ratio 0.85 \
  --eval_split test --batch_size 8 --num_workers 0 --seed 42 --snr_db 20 \
  --eval_mask_seed 42 --strict_load \
  --resume CHECKPOINT_PATH/checkpoint-149.pth \
  --output_dir OUTPUT_ROOT/d1_eval
```

`configs/std_only_public.example.yaml` records the same release settings for
experiment tracking.

## Weights

Weights are not tracked in Git and are not uploaded to Hugging Face in this
release. The planned public HF model package contains only:

```text
proposed_std_only_epoch149/checkpoint-149.pth
config.yaml
README.md
```

The expected SHA-256 for `checkpoint-149.pth` is:

```text
9c3227ee446c316ab564bc481c33d94b7ea13b7d2727c5af14beda36f6e5650b
```

Verify any downloaded checkpoint before evaluation:

```bash
sha256sum CHECKPOINT_PATH/checkpoint-149.pth
```

Raw and processed CSI data are not distributed through this repository or the
future HF model repository. Their source licenses and redistribution conditions
must be reviewed before a separate HF dataset repository is created.

## Citation

```bibtex
@article{zhang2026adaptive,
  title={Adaptive 3D-RoPE: Channel-Driven Rotary Positional Embedding for Wireless Foundation Models},
  author={Zhang, Chenyu and Lyu, Xinchen and Ren, Chenshan and Hou, Yanzhao and Zhang, Xuefei and Liu, Shuhan and Cui, Qimei},
  year={2026}
}
```

## License

The source code is released under Apache-2.0. See `LICENSE` and `NOTICE`.
