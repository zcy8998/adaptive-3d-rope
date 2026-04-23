#!/usr/bin/env bash

# D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29415 cli.py train \
--dataset D1,D2 \
--seed 42 \
--batch_size 64 --accum_iter 1 --lr 0.0008 \
--epochs 150 --warmup_epochs 10 --num_workers 16 \
--mask_type all \
--mask_ratio 0.5 \
--model csi_mae_base \
--rope_mode adaptive \
--encoder_pe none \
--decoder_pe sincos_3d \
--rope_theta 10 \
--data_dir "/home/zhangchenyu/data/csidata/train" \
--output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_adaptive \
--log_dir /home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_adaptive

# Fixed 3D-RoPE example:
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29416 cli.py train \
# --dataset D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16 \
# --seed 42 \
# --batch_size 64 --accum_iter 1 --lr 0.0008 \
# --epochs 150 --warmup_epochs 10 --num_workers 16 \
# --mask_type all \
# --mask_ratio 0.5 \
# --model csi_mae_base \
# --rope_mode fixed \
# --encoder_pe none \
# --decoder_pe sincos_3d \
# --rope_theta 10000 \
# --data_dir "/home/zhangchenyu/data/csidata/train" \
# --output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_fixed \
# --log_dir /home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_fixed
