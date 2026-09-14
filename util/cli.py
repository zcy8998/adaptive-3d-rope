import argparse
import os


POSITIONAL_ENCODING_ALIASES = {"none": "None"}


def _add_release_defaults(parser: argparse.ArgumentParser) -> None:
    """Lock the public CLI to the released Std-only Adaptive 3D-RoPE model."""
    parser.set_defaults(
        model="csi_mae_base",
        rope_mode="adaptive",
        rope_axes="3d",
        attention_backbone="global",
        encoder_pe="none",
        decoder_pe="none",
        rope_theta="10",
        rope_frequency_scale="1,1,1",
        rope_use_ape=False,
        cls_token=False,
        controller_mode="compact",
        controller_ablation="none",
        controller_max_scale=5.0,
        controller_features="",
        controller_lags="1,2,4",
        controller_validity_mode="axis_mean",
        controller_token_groups="token_std",
        controller_scale_granularity="axis_head",
        controller_token_normalization="frozen",
        controller_descriptor_hidden_dim=128,
        controller_token_hidden_dim=64,
        controller_raw_descriptor_groups="",
        controller_context_mode="sample",
        controller_token_pool="channel",
        controller_decoder_token_scope="visible_only",
        controller_architecture="token_only",
        controller_fusion_hidden_dim=64,
        adaptive_scope="encoder_decoder",
        controller_boundary_regularization=0.0,
        controller_neutralize_groups="",
        controller_shuffle_groups="",
        controller_shuffle_offset=1,
        transfer_adapter_ratio=0.0,
        rope_probe_train_axis="none",
        rope_oracle_scale_axis="none",
        rope_oracle_checkpoint="",
        frozen_probe_eval_mode=False,
        controller_snapshot_epochs="",
        report_memory=False,
        save_controller_diagnostics=False,
        profile_flops=False,
        reconstruction_sample_index=-1,
        reconstruction_dataset_name="",
        reconstruction_mask_type="random",
        reconstruction_output="",
        save_spatial_rms_output="",
        spatial_rms_dataset_name="",
        spatial_rms_mask_type="random",
        save_sample_nmse=False,
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="D1", help="Comma-separated dataset names.")
    parser.add_argument("--data_dir", required=True, help="Directory containing D*/ split files.")
    parser.add_argument("--data_num", default=1.0, type=float)
    parser.add_argument(
        "--mask_type", default="random", choices=("random", "temporal", "freq", "antenna", "all")
    )
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--snr_db", default=20.0, type=float)
    parser.add_argument("--use_phys_coord", action="store_true", default=False)


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local-rank", dest="local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--fast_inference", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument(
        "--compile_mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead"
    )


def _add_optimization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=1e-3, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--controller_lr_multiplier", default=0.1, type=float)
    parser.add_argument("--controller_calibration_batches", default=8, type=int)


def _add_checkpoint_args(parser: argparse.ArgumentParser, include_finetune: bool) -> None:
    parser.add_argument("--resume", default="")
    if include_finetune:
        parser.add_argument("--finetune", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--strict_load", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive 3D-RoPE core training and evaluation")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train the released Std-only model")
    _add_release_defaults(train)
    _add_data_args(train)
    _add_runtime_args(train)
    _add_optimization_args(train)
    _add_checkpoint_args(train, include_finetune=True)
    train.set_defaults(output_dir="./outputs/train")

    evaluate = subparsers.add_parser("eval", help="Evaluate a released-model checkpoint")
    _add_release_defaults(evaluate)
    _add_data_args(evaluate)
    _add_runtime_args(evaluate)
    evaluate.add_argument("--batch_size", default=1, type=int)
    evaluate.add_argument("--resume", required=True)
    evaluate.add_argument("--strict_load", action="store_true")
    evaluate.add_argument("--output_dir", default="./outputs/eval")
    evaluate.add_argument("--log_dir", default=None)
    evaluate.add_argument("--eval_split", default="test", choices=("val", "test"))
    evaluate.add_argument("--max_eval_batches", default=0, type=int)
    evaluate.add_argument("--eval_mask_seed", default=42, type=int)
    evaluate.add_argument("--save_sample_nmse", action="store_true")

    finetune = subparsers.add_parser("finetune", help="Fine-tune a released-model checkpoint")
    _add_release_defaults(finetune)
    _add_data_args(finetune)
    _add_runtime_args(finetune)
    _add_optimization_args(finetune)
    _add_checkpoint_args(finetune, include_finetune=True)
    finetune.set_defaults(output_dir="./outputs/finetune", blr=1e-4, warmup_epochs=2, epochs=20, finetune_mode="full")

    calibrate = subparsers.add_parser("calibrate_controller", help="Recalibrate frozen controller statistics")
    _add_release_defaults(calibrate)
    _add_data_args(calibrate)
    _add_runtime_args(calibrate)
    calibrate.add_argument("--batch_size", default=64, type=int)
    calibrate.add_argument("--controller_calibration_batches", default=8, type=int)
    calibrate.add_argument("--preserve_descriptor_statistics", action="store_true")
    calibrate.add_argument("--resume", required=True)
    calibrate.add_argument("--output_dir", required=True)
    calibrate.add_argument("--log_dir", default=None)

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    if args.command is None:
        parser.print_help()
        return False
    if args.command in {"train", "finetune"} and args.resume and args.finetune:
        parser.error("--resume and --finetune are mutually exclusive.")
    return True
