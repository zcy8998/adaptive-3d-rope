#!/usr/bin/env bash

# Few-shot / transfer example

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29425 cli.py finetune \
--dataset D17 \
--seed 42 \
--batch_size 16 --accum_iter 1 --lr 0.0008 \
--epochs 20 --warmup_epochs 0 --num_workers 8 \
--mask_type all \
--mask_ratio 0.5 \
--model csi_mae_base \
--rope_mode adaptive \
--encoder_pe complex_rotation \
--decoder_pe sincos_3d \
--rope_theta 10 \
--finetune "/home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_adaptive/checkpoint-149.pth" \
--data_dir "/home/zhangchenyu/data/csidata/train" \
--data_num 0.1 \
--output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/finetune_D17_10pct \
--log_dir /home/zhangchenyu/experiments/adaptive_3d_rope/finetune_D17_10pct

# Fixed 3D-RoPE example:
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29426 cli.py finetune \
# --dataset D17 \
# --seed 42 \
# --batch_size 16 --accum_iter 1 --lr 0.0008 \
# --epochs 20 --warmup_epochs 0 --num_workers 8 \
# --mask_type all \
# --mask_ratio 0.5 \
# --model csi_mae_base \
# --rope_mode fixed \
# --encoder_pe none \
# --decoder_pe sincos_3d \
# --rope_theta 10000 \
# --finetune "/home/zhangchenyu/experiments/adaptive_3d_rope/train_16dataset_fixed/checkpoint-149.pth" \
# --data_dir "/home/zhangchenyu/data/csidata/train" \
# --data_num 0.1 \
# --output_dir /home/zhangchenyu/experiments/adaptive_3d_rope/finetune_D17_10pct_fixed \
# --log_dir /home/zhangchenyu/experiments/adaptive_3d_rope/finetune_D17_10pct_fixed
