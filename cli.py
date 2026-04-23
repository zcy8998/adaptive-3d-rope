from pathlib import Path

from util.cli import build_parser, validate_args


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not validate_args(parser, args):
        return 0

    if getattr(args, "output_dir", None):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if getattr(args, "log_dir", None):
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    from util.runtime import run_eval, run_finetune, run_train

    command_handlers = {
        "train": run_train,
        "eval": run_eval,
        "finetune": run_finetune,
    }
    handler = command_handlers[args.command]
    handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
