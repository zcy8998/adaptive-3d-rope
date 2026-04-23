import argparse
import csv
import datetime
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models.csi_mae as csi_mae
import timm_utils.optim.optim_factory as optim_factory
import util.misc as misc
from engine_pretrain import train_one_epoch_3mask, train_one_epoch_csi
from util.cli import POSITIONAL_ENCODING_ALIASES
from util.data import CSIDataset, data_load_main
from util.misc import NativeScalerWithGradNormCount as NativeScaler


TRAIN_METRIC_FIELDS = (
    "epoch",
    "split",
    "mask_type",
    "train_loss",
    "lr",
    "avg_nmse_linear",
    "avg_nmse_db",
)


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _ensure_dir(path):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_device(args):
    requested = str(getattr(args, "device", "cuda")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        if getattr(args, "distributed", False):
            return torch.device(f"cuda:{args.gpu}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _set_seed(args):
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True


def _parse_rope_theta(value):
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 3:
            return tuple(float(part) for part in parts)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(float(part) for part in value)
    raise ValueError(f"Invalid rope_theta value: {value}")


def _mask_plan(mask_type, mask_ratio):
    if mask_type == "all":
        return {"random": 0.85, "temporal": 0.5, "freq": 0.5}
    return {mask_type: mask_ratio}


def _metric_csv_path(args):
    return Path(args.output_dir) / "train_metrics.csv"


def _append_metric_rows(csv_path: Path, rows):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAIN_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in TRAIN_METRIC_FIELDS}
            )


def _write_eval_outputs(output_dir: Path, metrics):
    _ensure_dir(output_dir)
    metrics_path = output_dir / "eval_metrics.json"
    summary_path = output_dir / "eval_summary.txt"

    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    lines = [
        f"Checkpoint: {metrics['checkpoint']}",
        f"RoPE mode: {metrics['rope_mode']}",
        f"Overall NMSE (linear): {metrics['overall_nmse_linear']:.7f}",
        f"Overall NMSE (dB): {metrics['overall_nmse_db']:.7f}",
    ]
    for mask_name, mask_metrics in metrics["mask_results"].items():
        lines.append(
            f"{mask_name}: avg_nmse_linear={mask_metrics['avg_nmse_linear']:.7f}, "
            f"avg_nmse_db={mask_metrics['avg_nmse_db']:.7f}"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def _build_model(args, device):
    model_ctor = csi_mae.__dict__[args.model]
    return model_ctor(
        cls_embed=args.cls_token,
        pos_emb_type=POSITIONAL_ENCODING_ALIASES[args.encoder_pe],
        decoder_pos_emb_type=POSITIONAL_ENCODING_ALIASES[args.decoder_pe],
        rope_mode=args.rope_mode,
        rope_theta=_parse_rope_theta(args.rope_theta),
        use_ape=args.rope_use_ape,
        device=device,
    )


def _configure_optimizer(args, model_without_ddp, effective_batch_size):
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    param_groups = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer, NativeScaler()


def _load_checkpoint(path, model_without_ddp, optimizer=None, loss_scaler=None, load_state=False):
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    message = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    print(f"Loaded checkpoint from: {path}")
    print(message)
    if load_state and optimizer is not None and loss_scaler is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _maybe_resume_or_finetune(args, model_without_ddp, optimizer, loss_scaler):
    if getattr(args, "resume", ""):
        checkpoint = _load_checkpoint(
            args.resume,
            model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            load_state=True,
        )
        if "epoch" in checkpoint:
            args.start_epoch = int(checkpoint["epoch"]) + 1
    elif getattr(args, "finetune", ""):
        _load_checkpoint(args.finetune, model_without_ddp)


def _build_train_loaders(args):
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    data_loaders = []
    for dataset_name in args.dataset.split(","):
        dataset_train = CSIDataset(
            dataset=dataset_name,
            world_size=num_tasks,
            rank=global_rank,
            dataset_type="train",
            data_dir=args.data_dir,
            data_num=args.data_num,
            return_phys_meta=args.use_phys_coord,
        )
        if args.distributed:
            sampler = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset_train)
        loader = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=False,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loaders.append(loader)
    return data_loaders


def _build_subset(dataset, data_num, seed):
    total = len(dataset)
    if data_num is None or data_num == 1.0:
        return dataset, total

    if 0 < data_num < 1:
        keep = max(1, int(total * data_num))
    elif data_num > 1:
        keep = min(total, int(data_num))
    else:
        raise ValueError(f"data_num should be positive, got {data_num}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)[:keep].tolist()
    return torch.utils.data.Subset(dataset, indices), keep


def _build_finetune_loader(args):
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    dataset_train = CSIDataset(
        dataset=args.dataset,
        world_size=num_tasks,
        rank=global_rank,
        dataset_type="train",
        data_dir=args.data_dir,
        return_phys_meta=args.use_phys_coord,
    )
    dataset_train, num_train = _build_subset(dataset_train, args.data_num, args.seed)
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset_train)
    loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=False,
        sampler=sampler,
        batch_size=min(args.batch_size, max(1, num_train)),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return loader, num_train


def _build_eval_loaders(args, split):
    return data_load_main(args, dataset_type=split, test_type="normal")


def _summarize_nmse(samples, pred, mask, token_length):
    target = torch.cat([samples.real, samples.imag], dim=-1)
    batch_size, seq_len, _ = target.shape
    col_indices = torch.arange(seq_len, device=target.device).expand(batch_size, seq_len)
    mask_in_length = col_indices < token_length[:, None]
    mask_nmse = mask.bool() & mask_in_length

    sample_nmse = []
    for index in range(batch_size):
        current_mask = mask_nmse[index]
        if not torch.any(current_mask):
            continue
        current_pred = pred[index][current_mask]
        current_target = target[index][current_mask]
        mse = torch.mean(torch.abs(current_target - current_pred) ** 2).item()
        power = torch.mean(torch.abs(current_target) ** 2).item()
        sample_nmse.append(mse / max(power, 1e-10))
    return sample_nmse


def evaluate_model(args, model, device, eval_loaders, split):
    mask_results = {}
    flattened_dataset_results = []
    overall_linear = []
    mask_plan = _mask_plan(args.mask_type, args.mask_ratio)

    model.eval()
    for mask_name, mask_ratio in mask_plan.items():
        dataset_results = []
        dataset_nmse_linear = []

        for dataset_loader in eval_loaders:
            dataset_name = dataset_loader.dataset.get_dataset_name()
            per_sample_nmse = []
            losses = []
            total_inference_time = 0.0
            num_batches = 0

            report_memory = getattr(args, "report_memory", False)
            if device.type == "cuda" and report_memory:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()

            with torch.no_grad():
                for batch in dataset_loader:
                    if len(batch) == 4:
                        samples, token_length, input_size, phys_meta = batch
                        phys_meta = phys_meta.to(device, non_blocking=True)
                    else:
                        samples, token_length, input_size = batch
                        phys_meta = None

                    samples = samples.to(device, non_blocking=True)
                    token_length = token_length.to(device)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with _autocast_context(device):
                        loss, pred, mask = model(
                            samples,
                            token_length,
                            input_size=input_size,
                            mask_ratio=mask_ratio,
                            mask_strategy=mask_name,
                            phys_meta=phys_meta,
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    total_inference_time += time.perf_counter() - start
                    num_batches += 1

                    losses.append(float(loss.item()))
                    per_sample_nmse.extend(
                        _summarize_nmse(samples, pred, mask, token_length)
                    )

            avg_nmse_linear = float(np.mean(per_sample_nmse)) if per_sample_nmse else float("inf")
            avg_nmse_db = float(10 * np.log10(np.clip(avg_nmse_linear, 1e-10, None)))
            avg_loss = float(np.mean(losses)) if losses else float("nan")
            avg_inference_ms = (total_inference_time / max(1, num_batches)) * 1000.0
            peak_memory_mb = None
            if device.type == "cuda" and report_memory:
                peak_memory_mb = float(
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                )

            result = {
                "dataset_name": dataset_name,
                "split": split,
                "mask_type": mask_name,
                "mask_ratio": mask_ratio,
                "loss": avg_loss,
                "nmse_linear": avg_nmse_linear,
                "nmse_db": avg_nmse_db,
                "avg_inference_ms": avg_inference_ms,
                "peak_memory_mb": peak_memory_mb,
            }
            dataset_results.append(result)
            flattened_dataset_results.append(result)
            dataset_nmse_linear.append(avg_nmse_linear)

            print(
                f"[{split}] {dataset_name} | mask={mask_name} | "
                f"loss={avg_loss:.7f} | nmse_linear={avg_nmse_linear:.7f} | "
                f"nmse_db={avg_nmse_db:.7f}"
            )

        avg_nmse_linear = float(np.mean(dataset_nmse_linear)) if dataset_nmse_linear else float("inf")
        avg_nmse_db = float(10 * np.log10(np.clip(avg_nmse_linear, 1e-10, None)))
        overall_linear.append(avg_nmse_linear)
        mask_results[mask_name] = {
            "mask_ratio": mask_ratio,
            "dataset_results": dataset_results,
            "avg_nmse_linear": avg_nmse_linear,
            "avg_nmse_db": avg_nmse_db,
        }
        print(
            f"[{split}] summary | mask={mask_name} | "
            f"avg_nmse_linear={avg_nmse_linear:.7f} | avg_nmse_db={avg_nmse_db:.7f}"
        )

    overall_nmse_linear = float(np.mean(overall_linear)) if overall_linear else float("inf")
    overall_nmse_db = float(10 * np.log10(np.clip(overall_nmse_linear, 1e-10, None)))
    return {
        "checkpoint": args.resume or getattr(args, "finetune", ""),
        "rope_mode": args.rope_mode,
        "mask_results": mask_results,
        "dataset_results": flattened_dataset_results,
        "overall_nmse_linear": overall_nmse_linear,
        "overall_nmse_db": overall_nmse_db,
    }


def _build_log_writer(args):
    if not misc.is_main_process():
        return None
    log_dir = args.log_dir or args.output_dir
    _ensure_dir(log_dir)
    return SummaryWriter(log_dir=log_dir)


def _save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, filename=None):
    if args.output_dir:
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            filename=filename,
        )


def run_train(args):
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)

    train_loaders = _build_train_loaders(args)
    val_loaders = _build_eval_loaders(args, split="val") if misc.is_main_process() else []
    log_writer = _build_log_writer(args)

    model = _build_model(args, device)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    effective_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    optimizer, loss_scaler = _configure_optimizer(args, model_without_ddp, effective_batch_size)
    _maybe_resume_or_finetune(args, model_without_ddp, optimizer, loss_scaler)

    start_time = time.time()
    metrics_path = _metric_csv_path(args)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_stats = []
        for loader in train_loaders:
            if args.distributed and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            if args.mask_type == "all":
                stats = train_one_epoch_3mask(
                    model, loader, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
                )
            else:
                stats = train_one_epoch_csi(
                    model, loader, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
                )
            epoch_stats.append(stats)

        avg_loss = float(np.mean([stat["loss"] for stat in epoch_stats]))
        avg_lr = float(np.mean([stat["lr"] for stat in epoch_stats]))
        _append_metric_rows(
            metrics_path,
            [
                {
                    "epoch": epoch,
                    "split": "train",
                    "mask_type": "",
                    "train_loss": avg_loss,
                    "lr": avg_lr,
                    "avg_nmse_linear": "",
                    "avg_nmse_db": "",
                }
            ],
        )

        should_save = (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs
        if should_save:
            _save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        if should_save and misc.is_main_process() and val_loaders:
            val_metrics = evaluate_model(args, model_without_ddp, device, val_loaders, split="val")
            rows = []
            for mask_name, mask_metrics in val_metrics["mask_results"].items():
                rows.append(
                    {
                        "epoch": epoch,
                        "split": "val",
                        "mask_type": mask_name,
                        "train_loss": "",
                        "lr": "",
                        "avg_nmse_linear": mask_metrics["avg_nmse_linear"],
                        "avg_nmse_db": mask_metrics["avg_nmse_db"],
                    }
                )
            _append_metric_rows(metrics_path, rows)

    _save_checkpoint(
        args,
        args.epochs - 1,
        model,
        model_without_ddp,
        optimizer,
        loss_scaler,
        filename="checkpoint-final.pth",
    )

    if log_writer is not None:
        log_writer.close()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")


def run_finetune(args):
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)

    train_loader, num_train = _build_finetune_loader(args)
    log_writer = _build_log_writer(args)

    model = _build_model(args, device)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    effective_batch_size = min(args.batch_size, max(1, num_train)) * args.accum_iter * misc.get_world_size()
    optimizer, loss_scaler = _configure_optimizer(args, model_without_ddp, effective_batch_size)
    _maybe_resume_or_finetune(args, model_without_ddp, optimizer, loss_scaler)

    start_time = time.time()
    metrics_path = _metric_csv_path(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if args.mask_type == "all":
            stats = train_one_epoch_3mask(
                model, train_loader, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
            )
        else:
            stats = train_one_epoch_csi(
                model, train_loader, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
            )

        _append_metric_rows(
            metrics_path,
            [
                {
                    "epoch": epoch,
                    "split": "train",
                    "mask_type": "",
                    "train_loss": stats["loss"],
                    "lr": stats["lr"],
                    "avg_nmse_linear": "",
                    "avg_nmse_db": "",
                }
            ],
        )

        should_save = (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs
        if should_save:
            _save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

    _save_checkpoint(
        args,
        args.epochs - 1,
        model,
        model_without_ddp,
        optimizer,
        loss_scaler,
        filename="checkpoint-final.pth",
    )

    if log_writer is not None:
        log_writer.close()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Fine-tuning time {elapsed}")


def run_eval(args):
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)

    model = _build_model(args, device)
    model.to(device)
    _load_checkpoint(args.resume, model)

    if not misc.is_main_process():
        return

    eval_loaders = _build_eval_loaders(args, split="test")
    metrics = evaluate_model(args, model, device, eval_loaders, split="test")
    _write_eval_outputs(Path(args.output_dir), metrics)
