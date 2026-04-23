#!/usr/bin/env bash

# D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24

CUDA_VISIBLE_DEVICES=0 python -u cli.py eval \
--dataset D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24 \
--seed 42 \
--batch_size 16 \
--num_workers 8 \
--mask_type all \
--mask_ratio 0.5 \
--model csi_mae_base \
--rope_mode adaptive \
--encoder_pe complex_rotation \
--decoder_pe sincos_3d \
--rope_theta 10 \
--resume "/home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_adaptive/checkpoint-149.pth" \
--data_dir "/home/zhangchenyu/data/csidata/extraploation/freq_v2" \
--output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/test_freq \
--report_memory

# Fixed 3D-RoPE example:
# CUDA_VISIBLE_DEVICES=0 python -u cli.py eval \
# --dataset D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24 \
# --seed 42 \
# --batch_size 16 \
# --num_workers 8 \
# --mask_type all \
# --mask_ratio 0.5 \
# --model csi_mae_base \
# --rope_mode fixed \
# --encoder_pe none \
# --decoder_pe sincos_3d \
# --rope_theta 10000 \
# --resume "/home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_fixed/checkpoint-149.pth" \
# --data_dir "/home/zhangchenyu/data/csidata/extraploation/freq_v2" \
# --output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/test_freq_fixed \
# --report_memory
