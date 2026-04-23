import argparse
import os


MODEL_CHOICES = ("csi_mae_base", "csi_mae_small", "csi_mae_tiny")
ROPE_MODE_CHOICES = ("none", "learnable", "fixed", "adaptive")
POSITIONAL_ENCODING_CHOICES = (
    "complex_rotation",
    "complex_rotation_3d",
    "trivial",
    "sincos",
    "sincos_2d",
    "sincos_3d",
    "sincos_1d_3",
    "none",
)
MASK_CHOICES = ("random", "temporal", "freq", "all")

POSITIONAL_ENCODING_ALIASES = {
    "complex_rotation": "ComplexRotation",
    "complex_rotation_3d": "ComplexRotation_3D",
    "trivial": "trivial",
    "sincos": "SinCos",
    "sincos_2d": "SinCos_2D",
    "sincos_3d": "SinCos_3D",
    "sincos_1d_3": "SinCos_1D_3",
    "none": "None",
}


def _add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", default="csi_mae_base", choices=MODEL_CHOICES)
    parser.add_argument("--rope_mode", default="adaptive", choices=ROPE_MODE_CHOICES)
    parser.add_argument(
        "--encoder_pe",
        default="complex_rotation",
        choices=POSITIONAL_ENCODING_CHOICES,
    )
    parser.add_argument(
        "--decoder_pe",
        default="sincos_3d",
        choices=POSITIONAL_ENCODING_CHOICES,
    )
    parser.add_argument(
        "--rope_theta",
        default="10",
        type=str,
        help="Single theta value or a comma-separated triplet such as 10,100,1000.",
    )
    parser.add_argument("--rope_use_ape", action="store_true", default=False)
    parser.add_argument("--cls_token", action="store_true", default=False)


def _add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default="D1", type=str)
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--data_num", default=1.0, type=float)
    parser.add_argument("--mask_type", default="random", choices=MASK_CHOICES)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--use_phys_coord", action="store_true", default=False)


def _add_runtime_args(parser: argparse.ArgumentParser):
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument(
        "--local-rank", dest="local_rank", default=os.getenv("LOCAL_RANK", 0), type=int
    )
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")


def _add_optimization_args(parser: argparse.ArgumentParser):
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)


def _add_checkpoint_args(parser: argparse.ArgumentParser, include_finetune: bool):
    parser.add_argument("--resume", default="", type=str)
    if include_finetune:
        parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--log_dir", default=None, type=str)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Adaptive 3D-RoPE training and evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Pretrain the CSI foundation model")
    _add_model_args(train_parser)
    _add_data_args(train_parser)
    _add_runtime_args(train_parser)
    _add_optimization_args(train_parser)
    _add_checkpoint_args(train_parser, include_finetune=True)
    train_parser.set_defaults(output_dir="./outputs/train")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained checkpoint")
    _add_model_args(eval_parser)
    _add_data_args(eval_parser)
    _add_runtime_args(eval_parser)
    eval_parser.add_argument("--batch_size", default=1, type=int)
    eval_parser.add_argument("--resume", default="", type=str)
    eval_parser.add_argument("--output_dir", default="./outputs/eval", type=str)
    eval_parser.add_argument("--log_dir", default=None, type=str)
    eval_parser.add_argument("--report_memory", action="store_true", default=False)

    finetune_parser = subparsers.add_parser(
        "finetune", help="Few-shot fine-tuning from source-domain checkpoints"
    )
    _add_model_args(finetune_parser)
    _add_data_args(finetune_parser)
    _add_runtime_args(finetune_parser)
    _add_optimization_args(finetune_parser)
    _add_checkpoint_args(finetune_parser, include_finetune=True)
    finetune_parser.set_defaults(
        output_dir="./outputs/finetune",
        blr=1e-4,
        warmup_epochs=2,
        epochs=20,
    )

    return parser


def validate_args(parser: argparse.ArgumentParser, args):
    if args.command is None:
        parser.print_help()
        return False

    if args.command == "eval" and not args.resume:
        parser.error("`eval` requires `--resume`.")

    if args.command in {"train", "finetune"} and args.resume and args.finetune:
        parser.error("`--resume` and `--finetune` are mutually exclusive.")

    return True
