"""Argument parsing for the public Adaptive 3D-RoPE release."""
from __future__ import annotations
import argparse
import os
from util.profiles import apply_profile, profile_names

POSITIONAL_ENCODING_ALIASES = {"sincos": "SinCos", "sincos_3d": "SinCos_3D", "none": "None"}
MASK_CHOICES = ("random", "temporal", "freq", "antenna", "all")
_CORE_COMMANDS = {"train", "eval", "finetune", "calibrate_controller"}

def _add_profile_arg(parser, required=True):
    parser.add_argument("--profile", choices=profile_names(), required=required,
                        help="Published model profile; fixes architecture and positional encoding.")

def _add_data_args(parser):
    parser.add_argument("--dataset", default="D1", help="Comma-separated dataset names.")
    parser.add_argument("--data_dir", required=True, help="Directory containing D*/ split files.")
    parser.add_argument("--data_num", default=1.0, type=float)
    parser.add_argument("--mask_type", default="random", choices=MASK_CHOICES)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--snr_db", default=20.0, type=float)
    parser.add_argument("--use_phys_coord", action="store_true", default=False)

def _add_runtime_args(parser):
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int); parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem"); parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local-rank", dest="local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--dist_on_itp", action="store_true"); parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--fast_inference", action="store_true"); parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")

def _add_optimization_args(parser):
    parser.add_argument("--batch_size", default=32, type=int); parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--accum_iter", default=1, type=int); parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lr", default=None, type=float); parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float); parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--controller_lr_multiplier", default=0.1, type=float); parser.add_argument("--controller_calibration_batches", default=8, type=int)

def _add_checkpoint_args(parser, include_finetune):
    parser.add_argument("--resume", default="")
    if include_finetune: parser.add_argument("--finetune", default="")
    parser.add_argument("--start_epoch", default=0, type=int); parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--output_dir", default="./outputs"); parser.add_argument("--log_dir", default=None); parser.add_argument("--strict_load", action="store_true")

def _add_deepmimo_args(parser, training):
    _add_profile_arg(parser, required=False); _add_runtime_args(parser)
    parser.add_argument("--task", required=True, choices=("beam_management", "csi_feedback", "los_classification"))
    parser.add_argument("--deepmimo_root", required=True); parser.add_argument("--deepmimo_scenario", required=True); parser.add_argument("--deepmimo_cache_dir", default="")
    parser.add_argument("--pretrained", default=""); parser.add_argument("--baseline", default="ours", choices=("ours", "mlp", "cnn", "lstm", "beam_codebook", "csinet", "csinet_lstm", "transnet"))
    parser.add_argument("--adaptation_mode", default="adapter", choices=("head", "adapter", "full")); parser.add_argument("--data_num", type=float, default=1.0)
    parser.add_argument("--set_a_size", type=int, default=256); parser.add_argument("--set_b_size", type=int, default=16); parser.add_argument("--csi_compression_ratio", type=int, default=16)
    parser.add_argument("--csi_codec_architecture", default="token_lowrank", choices=("token_lowrank", "dense_legacy")); parser.add_argument("--csi_codec_hidden_dim", type=int, default=64)
    parser.add_argument("--adapter_bottleneck_ratio", type=float, default=0.125); parser.add_argument("--adapter_depth", type=int, default=1); parser.add_argument("--beam_rsrp_loss_weight", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32); parser.add_argument("--accum_iter", type=int, default=1); parser.add_argument("--output_dir", required=True); parser.add_argument("--resume", default="")
    if training:
        parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--task_lr", type=float, default=1e-4); parser.add_argument("--encoder_lr", type=float, default=1e-5); parser.add_argument("--weight_decay", type=float, default=0.0); parser.add_argument("--early_stopping_patience", type=int, default=0)
    else: parser.add_argument("--eval_split", default="test", choices=("val", "test"))

def build_parser():
    parser = argparse.ArgumentParser(description="Adaptive 3D-RoPE training and evaluation")
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train", help="Pretrain a published positional-encoding profile"); _add_profile_arg(train); _add_data_args(train); _add_runtime_args(train); _add_optimization_args(train); _add_checkpoint_args(train, True); train.set_defaults(output_dir="./outputs/train")
    evaluate = sub.add_parser("eval", help="Evaluate a published checkpoint profile"); _add_profile_arg(evaluate); _add_data_args(evaluate); _add_runtime_args(evaluate); evaluate.add_argument("--batch_size", default=1, type=int); evaluate.add_argument("--resume", required=True); evaluate.add_argument("--strict_load", action="store_true", default=True); evaluate.add_argument("--output_dir", default="./outputs/eval"); evaluate.add_argument("--log_dir", default=None); evaluate.add_argument("--eval_split", default="test", choices=("val", "test")); evaluate.add_argument("--max_eval_batches", default=0, type=int); evaluate.add_argument("--eval_mask_seed", default=42, type=int); evaluate.add_argument("--save_sample_nmse", action="store_true")
    finetune = sub.add_parser("finetune", help="Fine-tune a published checkpoint profile"); _add_profile_arg(finetune); _add_data_args(finetune); _add_runtime_args(finetune); _add_optimization_args(finetune); _add_checkpoint_args(finetune, True); finetune.set_defaults(output_dir="./outputs/finetune", blr=1e-4, warmup_epochs=2, epochs=20, finetune_mode="full")
    calibrate = sub.add_parser("calibrate_controller", help="Recalibrate a published adaptive profile"); _add_profile_arg(calibrate); _add_data_args(calibrate); _add_runtime_args(calibrate); calibrate.add_argument("--batch_size", default=64, type=int); calibrate.add_argument("--controller_calibration_batches", default=8, type=int); calibrate.add_argument("--preserve_descriptor_statistics", action="store_true"); calibrate.add_argument("--resume", required=True); calibrate.add_argument("--output_dir", required=True); calibrate.add_argument("--log_dir", default=None)
    _add_deepmimo_args(sub.add_parser("deepmimo_train", help="Train a DeepMIMO transfer model or baseline"), True); _add_deepmimo_args(sub.add_parser("deepmimo_eval", help="Evaluate a DeepMIMO transfer model or baseline"), False)
    return parser

def _set_runtime_defaults(args):
    for key, value in {"rope_probe_train_axis":"none", "rope_oracle_scale_axis":"none", "rope_oracle_checkpoint":"", "frozen_probe_eval_mode":False, "controller_snapshot_epochs":"", "report_memory":False, "save_controller_diagnostics":False, "profile_flops":False, "reconstruction_sample_index":-1, "reconstruction_dataset_name":"", "reconstruction_mask_type":"random", "reconstruction_output":"", "save_spatial_rms_output":"", "spatial_rms_dataset_name":"", "spatial_rms_mask_type":"random"}.items(): setattr(args, key, value)

def validate_args(parser, args):
    if args.command is None: parser.print_help(); return False
    if args.command in _CORE_COMMANDS: apply_profile(args); _set_runtime_defaults(args)
    elif args.command.startswith("deepmimo_"):
        if args.profile: apply_profile(args); _set_runtime_defaults(args)
        elif args.baseline == "ours": parser.error("DeepMIMO --baseline ours requires a published --profile.")
    if args.command in {"train", "finetune"} and args.resume and args.finetune: parser.error("--resume and --finetune are mutually exclusive.")
    if args.command == "calibrate_controller" and args.profile != "proposed_std_only": parser.error("calibrate_controller is only defined for --profile proposed_std_only.")
    if args.command.startswith("deepmimo_") and args.baseline == "ours" and not args.pretrained: parser.error("DeepMIMO --baseline ours requires --pretrained.")
    return True
