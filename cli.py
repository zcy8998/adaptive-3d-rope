from pathlib import Path
import torch.distributed as dist
from util.cli import build_parser, validate_args
from util.runtime import run_calibrate_controller, run_eval, run_finetune, run_train

def main() -> int:
    parser = build_parser(); args = parser.parse_args()
    if not validate_args(parser, args): return 0
    if getattr(args, "output_dir", None): Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if getattr(args, "log_dir", None): Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    handlers = {"train": run_train, "eval": run_eval, "finetune": run_finetune, "calibrate_controller": run_calibrate_controller}
    if args.command.startswith("deepmimo_"):
        from util.downstream_runtime import run_downstream_eval, run_downstream_train
        handlers.update({"deepmimo_train": run_downstream_train, "deepmimo_eval": run_downstream_eval})
    try: handlers[args.command](args)
    finally:
        if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()
    return 0

if __name__ == "__main__": raise SystemExit(main())
