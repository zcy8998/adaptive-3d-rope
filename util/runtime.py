import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
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
from util.checkpoint import load_checkpoint as load_torch_checkpoint
from util.data import CSIDataset, data_load_main
from util.eval_reproducibility import stable_eval_mask_seed
from util.profiles import validate_checkpoint_profile
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

CONTROLLER_DIAGNOSTIC_FIELDS = (
    "dataset_name",
    "split",
    "mask_type",
    "sample_index",
    "axis",
    "scale_mean",
    "scale_std",
)


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def _ensure_dir(path):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _dataset_manifest(args):
    records = []
    for dataset_name in args.dataset.split(","):
        config_path = Path(args.data_dir) / dataset_name / "config.mat"
        record = {"dataset": dataset_name, "config_path": str(config_path)}
        if config_path.exists():
            record["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        records.append(record)
    return records


def _checkpoint_record(path_text):
    """Return a reproducible checkpoint identity without loading its tensors."""
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    record = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    record["sha256"] = digest.hexdigest()
    record["bytes"] = path.stat().st_size
    return record


def _write_run_manifest(args, device, phase):
    if not misc.is_main_process():
        return
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    checkpoints = {
        key: record
        for key in ("resume", "finetune", "rope_oracle_checkpoint")
        if (record := _checkpoint_record(getattr(args, key, ""))) is not None
    }
    primary_checkpoint = checkpoints.get("resume") or checkpoints.get("finetune")
    manifest = {
        "phase": phase,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": _git_commit(),
        "execution_host": platform.node(),
        "working_directory": os.getcwd(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
        "arguments": vars(args),
        "datasets": _dataset_manifest(args),
        "checkpoints": checkpoints,
        "checkpoint": primary_checkpoint["path"] if primary_checkpoint else None,
        "checkpoint_sha256": (
            primary_checkpoint.get("sha256") if primary_checkpoint else None
        ),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _frozen_probe_state_hash(model):
    """Hash persistent model state while excluding diagnostic oracle logits."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("rope_oracle_logits_"):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_frozen_probe_integrity(args, model, initial_hash=None):
    """Record the exact trainable set and verify frozen state did not move."""
    if not misc.is_main_process():
        return None
    if str(getattr(args, "rope_oracle_scale_axis", "none")) == "none":
        return None
    current_hash = _frozen_probe_state_hash(model)
    trainable = {
        name: int(parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    payload = {
        "excluded_state_prefix": "rope_oracle_logits_",
        "initial_frozen_state_sha256": initial_hash or current_hash,
        "final_frozen_state_sha256": current_hash if initial_hash else None,
        "frozen_state_unchanged": (
            bool(current_hash == initial_hash) if initial_hash is not None else None
        ),
        "trainable_parameters": trainable,
        "trainable_parameter_count": int(sum(trainable.values())),
        "deterministic_frozen_forward": bool(
            getattr(args, "frozen_probe_eval_mode", False)
        ),
    }
    destination = Path(args.output_dir) / "frozen_probe_integrity.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return current_hash


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
        deterministic_probe = bool(
            getattr(args, "frozen_probe_eval_mode", False)
        )
        cudnn.benchmark = not deterministic_probe
        cudnn.deterministic = deterministic_probe


def _reset_eval_mask_seed(base_seed, dataset_name, mask_name):
    seed = stable_eval_mask_seed(base_seed, dataset_name, mask_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


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


def _parse_positive_ints(value):
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected comma-separated positive integers, got {value}")
    return values


def _mask_plan(mask_type, mask_ratio):
    if mask_type == "all":
        return {"random": 0.85, "temporal": 0.5, "freq": 0.5}
    return {mask_type: mask_ratio}


def _metric_csv_path(args):
    return Path(args.output_dir) / "train_metrics.csv"


def _write_controller_snapshot(args, model, epoch, train_loaders, device):
    requested = {
        int(value.strip())
        for value in str(getattr(args, "controller_snapshot_epochs", "")).split(",")
        if value.strip()
    }
    if epoch not in requested:
        return
    was_training = model.training
    model.eval()
    collected = {
        (side, quantity): []
        for side in ("encoder", "decoder")
        for quantity in (
            "token_raw",
            "token_projected_input",
            "controller_logits",
            "scale",
        )
    }
    with torch.no_grad():
        for loader in train_loaders:
            batch = next(iter(loader))
            if len(batch) == 4:
                samples, token_length, input_size, phys_meta = batch
                phys_meta = phys_meta.to(device, non_blocking=True)
            else:
                samples, token_length, input_size = batch
                phys_meta = None
            samples = samples.to(device, non_blocking=True)
            token_length = token_length.to(device)
            for mask_name, mask_ratio in _mask_plan("all", args.mask_ratio).items():
                model(
                    imgs=samples,
                    token_length=token_length,
                    input_size=input_size,
                    mask_ratio=mask_ratio,
                    mask_strategy=mask_name,
                    phys_meta=phys_meta,
                )
                for side, name in (
                    ("encoder", "enc_rope_controller"),
                    ("decoder", "dec_rope_controller"),
                ):
                    controller = getattr(model, name, None)
                    if controller is None:
                        continue
                    tensors = {
                        "token_raw": getattr(controller, "last_token_raw", None),
                        "token_projected_input": getattr(
                            controller, "last_token_context", None
                        ),
                        "controller_logits": getattr(controller, "last_logits", None),
                        "scale": getattr(controller, "last_scale", None),
                    }
                    for quantity, values in tensors.items():
                        if values is not None and values.numel():
                            collected[(side, quantity)].append(
                                values.detach().float().flatten().cpu()
                            )
    if was_training:
        model.train()
    rows = []
    for (side, quantity), values in collected.items():
        if not values:
            continue
        flat = torch.cat(values)
        rows.append(
            {
                "epoch": epoch,
                "side": side,
                "quantity": quantity,
                "source": "balanced_D1_D16_first_batch_three_masks",
                "count": flat.numel(),
                "mean": float(flat.mean()),
                "std": float(flat.std(unbiased=False)),
                "min": float(flat.min()),
                "p01": float(torch.quantile(flat, 0.01)),
                "p50": float(torch.quantile(flat, 0.50)),
                "p99": float(torch.quantile(flat, 0.99)),
                "max": float(flat.max()),
            }
        )
    if not rows:
        return
    path = Path(args.output_dir) / "controller_training_snapshots.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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
    diagnostic_path = output_dir / "controller_diagnostics.csv"
    sample_path = output_dir / "sample_nmse.csv"

    controller_rows = metrics.pop("controller_diagnostics", [])
    sample_rows = metrics.pop("sample_nmse", [])

    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    if controller_rows:
        fieldnames = []
        for row in controller_rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with diagnostic_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(controller_rows)

    if sample_rows:
        with sample_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(sample_rows)

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
        rope_axes=args.rope_axes,
        attention_backbone=args.attention_backbone,
        rope_theta=_parse_rope_theta(args.rope_theta),
        rope_frequency_scale=_parse_rope_theta(args.rope_frequency_scale),
        use_ape=args.rope_use_ape,
        controller_mode=args.controller_mode,
        controller_ablation=args.controller_ablation,
        controller_max_scale=args.controller_max_scale,
        controller_feature_groups=args.controller_features,
        controller_lags=_parse_positive_ints(args.controller_lags),
        controller_validity_mode=args.controller_validity_mode,
        controller_token_groups=args.controller_token_groups,
        controller_scale_granularity=args.controller_scale_granularity,
        controller_token_normalization=args.controller_token_normalization,
        controller_descriptor_hidden_dim=args.controller_descriptor_hidden_dim,
        controller_token_hidden_dim=args.controller_token_hidden_dim,
        controller_raw_descriptor_groups=args.controller_raw_descriptor_groups,
        controller_context_mode=args.controller_context_mode,
        controller_token_pool=args.controller_token_pool,
        controller_decoder_token_scope=args.controller_decoder_token_scope,
        controller_architecture=args.controller_architecture,
        controller_fusion_hidden_dim=args.controller_fusion_hidden_dim,
        adaptive_scope=args.adaptive_scope,
        controller_boundary_regularization=args.controller_boundary_regularization,
        controller_neutralize_groups=args.controller_neutralize_groups,
        controller_shuffle_groups=args.controller_shuffle_groups,
        controller_shuffle_offset=args.controller_shuffle_offset,
        transfer_adapter_ratio=args.transfer_adapter_ratio,
        device=device,
    )


def _configure_optimizer(args, model_without_ddp, effective_batch_size):
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    controller_names = (
        "rope_controller",
        "base_freq",
        ".freqs",
    )
    probe_mode = str(getattr(args, "rope_probe_train_axis", "none")) != "none"
    oracle_mode = str(getattr(args, "rope_oracle_scale_axis", "none")) != "none"
    regularized, no_decay, controller = [], [], []
    for name, parameter in model_without_ddp.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in controller_names) or oracle_mode or (
            probe_mode and "freqs" in name
        ):
            controller.append(parameter)
        elif parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            regularized.append(parameter)
    param_groups = [
        {"params": regularized, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {
            "params": controller,
            "weight_decay": 0.0,
            "lr_scale": float(args.controller_lr_multiplier),
            "lr": args.lr * float(args.controller_lr_multiplier),
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer, NativeScaler()


def _configure_rope_probe(args, model):
    """Restrict a diagnostic run to one axis of both learnable RoPE banks."""
    axis_name = str(getattr(args, "rope_probe_train_axis", "none"))
    if axis_name == "none":
        return []
    if getattr(model, "rope_mode", None) != "learnable" or getattr(model, "rope_axes", None) != "3d":
        raise ValueError("--rope_probe_train_axis requires Learnable 3D-RoPE")
    axis_index = {"t": 0, "k": 1, "u": 2}[axis_name]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hooks = []
    for parameter_name in ("freqs", "dec_freqs"):
        parameter = getattr(model, parameter_name, None)
        if not isinstance(parameter, torch.nn.Parameter) or parameter.ndim < 1 or parameter.shape[0] != 3:
            raise RuntimeError(f"Missing three-axis learnable bank: {parameter_name}")
        parameter.requires_grad_(True)
        mask = torch.zeros_like(parameter)
        mask[axis_index] = 1
        hooks.append(parameter.register_hook(lambda gradient, m=mask: gradient * m))
    print(
        {
            "rope_probe_train_axis": axis_name,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "effective_trainable_axis_values": sum(
                getattr(model, name)[axis_index].numel() for name in ("freqs", "dec_freqs")
            ),
        },
        flush=True,
    )
    return hooks


def _install_rope_oracle_scale(args, model, freeze_model: bool):
    """Install the axis-head oracle scale diagnostic before checkpoint load."""
    axis_name = str(getattr(args, "rope_oracle_scale_axis", "none"))
    if axis_name == "none":
        return
    if str(getattr(args, "rope_probe_train_axis", "none")) != "none":
        raise ValueError("Full-bank and oracle-scale probes are mutually exclusive")
    if freeze_model:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.configure_rope_oracle_scale(axis_name)
    trainable = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("rope_oracle_logits_")
    ]
    if freeze_model:
        for parameter in trainable:
            parameter.requires_grad_(True)
    print(
        {
            "rope_oracle_scale_axis": axis_name,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ) if freeze_model else None,
            "oracle_scale_values": sum(parameter.numel() for parameter in trainable),
        },
        flush=True,
    )


def _load_checkpoint(path, model_without_ddp, optimizer=None, loss_scaler=None, load_state=False, strict=False):
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    profile = getattr(model_without_ddp, "_public_profile", None)
    if profile:
        validate_checkpoint_profile(checkpoint, profile)
    message = model_without_ddp.load_state_dict(checkpoint["model"], strict=strict)
    print(f"Loaded checkpoint from: {path}")
    if checkpoint.get("oracle_scale_only", False):
        print(
            {
                "oracle_scale_only": True,
                "loaded_keys": sorted(checkpoint["model"]),
            },
            flush=True,
        )
    else:
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
            strict=getattr(args, "strict_load", False),
        )
        if "epoch" in checkpoint:
            args.start_epoch = int(checkpoint["epoch"]) + 1
    elif getattr(args, "finetune", ""):
        _load_checkpoint(args.finetune, model_without_ddp, strict=getattr(args, "strict_load", False))


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
            SNR=args.snr_db,
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
        SNR=args.snr_db,
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
    # ``data_num`` is a training-data budget.  Validation and test splits must
    # remain complete so that model selection is comparable across fractions.
    # The legacy loader reads ``args.data_num`` directly, hence the temporary
    # override instead of passing the value as a separate argument.
    training_data_num = getattr(args, "data_num", 1.0)
    try:
        args.data_num = 1.0
        return data_load_main(args, dataset_type=split, test_type="normal")
    finally:
        args.data_num = training_data_num


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


def _summarize_nmse_energy(samples, pred, mask, token_length):
    """Return masked-patch error and target energies for each sample."""
    target = torch.cat([samples.real, samples.imag], dim=-1)
    batch_size, seq_len, _ = target.shape
    col_indices = torch.arange(seq_len, device=target.device).expand(batch_size, seq_len)
    valid = col_indices < token_length[:, None]
    selected = mask.bool() & valid
    rows = []
    for index in range(batch_size):
        current_mask = selected[index]
        if not torch.any(current_mask):
            rows.append({"error_energy": 0.0, "target_energy": 0.0, "nmse_linear": float("inf")})
            continue
        error = torch.sum((pred[index][current_mask] - target[index][current_mask]) ** 2).item()
        energy = torch.sum(target[index][current_mask] ** 2).item()
        rows.append({
            "error_energy": float(error),
            "target_energy": float(energy),
            "nmse_linear": float(error / max(energy, 1e-10)),
        })
    return rows


def _batch_sample_dims(input_size, index):
    dims = []
    for values in input_size:
        if torch.is_tensor(values):
            value = values[index] if values.ndim else values
            dims.append(int(value.item()))
        elif isinstance(values, (tuple, list)):
            dims.append(int(values[index]))
        else:
            dims.append(int(values))
    return tuple(dims)


def _unpatchify_complex_tokens(tokens, dims, patch_size=4):
    t, k, u = (int(value) for value in dims)
    if t % patch_size or k % patch_size or u % patch_size:
        raise ValueError(f"Dimensions {dims} are not divisible by patch_size={patch_size}")
    tb, kb, ub = t // patch_size, k // patch_size, u // patch_size
    count = tb * kb * ub
    values = tokens[:count].reshape(tb, kb, ub, patch_size, patch_size, patch_size)
    return values.permute(0, 3, 1, 4, 2, 5).reshape(t, k, u)


def _archive_reconstruction_sample(
    args,
    dataset_name,
    mask_name,
    sample_index,
    samples,
    pred,
    mask,
    token_length,
    input_size,
    batch_index,
    sample_nmse_linear,
):
    if not getattr(args, "reconstruction_output", ""):
        return
    if misc.get_world_size() != 1:
        raise RuntimeError("Reconstruction archival requires single-process evaluation")
    dims = _batch_sample_dims(input_size, batch_index)
    valid_length = int(token_length[batch_index].item())
    target_tokens = samples[batch_index, :valid_length]
    if not torch.is_complex(target_tokens):
        raise TypeError("Reconstruction archival expects complex CSI patch tokens")
    prediction_tokens = torch.complex(
        pred[batch_index, :valid_length, : target_tokens.shape[-1]],
        pred[batch_index, :valid_length, target_tokens.shape[-1] :],
    )
    target_grid = _unpatchify_complex_tokens(target_tokens, dims)
    prediction_grid = _unpatchify_complex_tokens(prediction_tokens, dims)
    patch_mask = mask[batch_index, :valid_length].bool()
    patch_size = 4
    t, k, u = dims
    tb, kb, ub = t // patch_size, k // patch_size, u // patch_size
    raw_mask = (
        patch_mask.reshape(tb, kb, ub)
        .repeat_interleave(patch_size, 0)
        .repeat_interleave(patch_size, 1)
        .repeat_interleave(patch_size, 2)
    )
    mask_bytes = np.packbits(
        raw_mask.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    ).tobytes()
    mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
    mask_seed = stable_eval_mask_seed(
        getattr(args, "eval_mask_seed", args.seed), dataset_name, mask_name
    )
    # A completed MAE reconstruction retains the observed CSI and uses the
    # decoder output only for masked patches.  Decoder values at visible
    # locations are not supervised by the masked reconstruction objective and
    # must not be treated as predictions in the Introduction comparison.
    prediction_grid = torch.where(raw_mask, prediction_grid, target_grid)
    output = Path(args.reconstruction_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset_name=np.asarray(dataset_name),
        mask_type=np.asarray(mask_name),
        seed=np.asarray(int(args.seed)),
        eval_mask_seed=np.asarray(int(getattr(args, "eval_mask_seed", args.seed))),
        mask_seed=np.asarray(mask_seed),
        mask_sha256=np.asarray(mask_sha256),
        sample_index=np.asarray(int(sample_index)),
        dims=np.asarray(dims, dtype=np.int64),
        target_real=target_grid.real.float().cpu().numpy(),
        target_imag=target_grid.imag.float().cpu().numpy(),
        prediction_real=prediction_grid.real.float().cpu().numpy(),
        prediction_imag=prediction_grid.imag.float().cpu().numpy(),
        masked_raw=raw_mask.cpu().numpy().astype(np.bool_),
        sample_nmse_linear=np.asarray(float(sample_nmse_linear)),
    )


def _batch_spatial_rms_maps(samples, pred, mask, token_length, input_size):
    rows = []
    for index in range(samples.shape[0]):
        dims = _batch_sample_dims(input_size, index)
        valid_length = int(token_length[index].item())
        target_tokens = samples[index, :valid_length]
        prediction_tokens = torch.complex(
            pred[index, :valid_length, : target_tokens.shape[-1]],
            pred[index, :valid_length, target_tokens.shape[-1] :],
        )
        target = _unpatchify_complex_tokens(target_tokens, dims)
        prediction = _unpatchify_complex_tokens(prediction_tokens, dims)
        patch_mask = mask[index, :valid_length].bool()
        t, k, u = dims
        raw_mask = (
            patch_mask.reshape(t // 4, k // 4, u // 4)
            .repeat_interleave(4, 0)
            .repeat_interleave(4, 1)
            .repeat_interleave(4, 2)
        )
        reconstruction = torch.where(raw_mask, prediction, target)
        target_rms = torch.sqrt(torch.mean(torch.abs(target) ** 2, dim=(0, 1)))
        reconstruction_rms = torch.sqrt(
            torch.mean(torch.abs(reconstruction) ** 2, dim=(0, 1))
        )
        mask_bytes = np.packbits(
            raw_mask.detach().cpu().numpy().astype(np.uint8).reshape(-1)
        ).tobytes()
        rows.append(
            {
                "dims": dims,
                "target_rms": target_rms.float().cpu().numpy(),
                "reconstruction_rms": reconstruction_rms.float().cpu().numpy(),
                "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            }
        )
    return rows


def _write_spatial_rms_archive(args, rows):
    output_text = getattr(args, "save_spatial_rms_output", "")
    if not output_text or not rows:
        return
    output = Path(output_text)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset_name=np.asarray(rows[0]["dataset_name"]),
        mask_type=np.asarray(rows[0]["mask_type"]),
        eval_mask_seed=np.asarray(int(getattr(args, "eval_mask_seed", args.seed))),
        mask_seed=np.asarray(int(rows[0]["mask_seed"])),
        sample_indices=np.asarray([row["sample_index"] for row in rows], dtype=np.int64),
        dims=np.asarray([row["dims"] for row in rows], dtype=np.int64),
        target_rms=np.stack([row["target_rms"] for row in rows]),
        reconstruction_rms=np.stack([row["reconstruction_rms"] for row in rows]),
        nmse_linear=np.asarray([row["nmse_linear"] for row in rows], dtype=np.float64),
        mask_sha256=np.asarray([row["mask_sha256"] for row in rows]),
    )


def _parameter_summary(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    controller = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "rope_controller" in name
    )
    return {"parameters": total, "controller_parameters": controller}


def _profile_batch_flops(model, batch_kwargs):
    try:
        with torch.profiler.profile(with_flops=True) as profile:
            model(**batch_kwargs)
        return int(sum(event.flops for event in profile.key_averages()))
    except (RuntimeError, AssertionError, NotImplementedError) as error:
        print(f"FLOP profiling unavailable: {error}")
        return None


def evaluate_model(args, model, device, eval_loaders, split):
    mask_results = {}
    flattened_dataset_results = []
    overall_linear = []
    controller_rows = []
    sample_rows = []
    spatial_rms_rows = []
    mask_plan = _mask_plan(args.mask_type, args.mask_ratio)

    model.eval()
    for mask_name, mask_ratio in mask_plan.items():
        dataset_results = []
        dataset_nmse_linear = []

        for dataset_loader in eval_loaders:
            dataset_name = dataset_loader.dataset.get_dataset_name()
            mask_seed = _reset_eval_mask_seed(
                getattr(args, "eval_mask_seed", args.seed), dataset_name, mask_name
            )
            per_sample_nmse = []
            dataset_sample_index = 0
            dataset_seen = 0
            losses = []
            total_inference_time = 0.0
            num_batches = 0
            profiled_flops = None

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
                    batch_kwargs = {
                        "imgs": samples,
                        "token_length": token_length,
                        "input_size": input_size,
                        "mask_ratio": mask_ratio,
                        "mask_strategy": mask_name,
                        "phys_meta": phys_meta,
                    }
                    if getattr(args, "profile_flops", False) and profiled_flops is None:
                        profiled_flops = _profile_batch_flops(model, batch_kwargs)
                    with _autocast_context(device):
                        loss, pred, mask = model(**batch_kwargs)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    total_inference_time += time.perf_counter() - start
                    num_batches += 1

                    losses.append(float(loss.item()))
                    batch_energy = _summarize_nmse_energy(samples, pred, mask, token_length)
                    batch_nmse = [row["nmse_linear"] for row in batch_energy]
                    per_sample_nmse.extend(batch_nmse)
                    if (
                        getattr(args, "save_spatial_rms_output", "")
                        and dataset_name == getattr(args, "spatial_rms_dataset_name", "")
                        and mask_name == getattr(args, "spatial_rms_mask_type", "random")
                    ):
                        spatial_batch = _batch_spatial_rms_maps(
                            samples, pred, mask, token_length, input_size
                        )
                        for local_index, row in enumerate(spatial_batch):
                            spatial_rms_rows.append(
                                {
                                    **row,
                                    "dataset_name": dataset_name,
                                    "mask_type": mask_name,
                                    "mask_seed": mask_seed,
                                    "sample_index": dataset_seen + local_index,
                                    "nmse_linear": batch_nmse[local_index],
                                }
                            )
                    requested_index = getattr(args, "reconstruction_sample_index", -1)
                    requested_dataset = getattr(args, "reconstruction_dataset_name", "")
                    requested_mask = getattr(args, "reconstruction_mask_type", "random")
                    if (
                        getattr(args, "reconstruction_output", "")
                        and dataset_name == requested_dataset
                        and mask_name == requested_mask
                        and dataset_seen <= requested_index < dataset_seen + samples.shape[0]
                    ):
                        local_index = requested_index - dataset_seen
                        _archive_reconstruction_sample(
                            args,
                            dataset_name,
                            mask_name,
                            requested_index,
                            samples,
                            pred,
                            mask,
                            token_length,
                            input_size,
                            local_index,
                            batch_nmse[local_index],
                        )
                    dataset_seen += samples.shape[0]
                    if getattr(args, "save_sample_nmse", False):
                        for value, energy in zip(batch_nmse, batch_energy):
                            sample_rows.append(
                                {
                                    "data_root": str(args.data_dir),
                                    "dataset_name": dataset_name,
                                    "split": split,
                                    "mask_type": mask_name,
                                    "mask_seed": mask_seed,
                                    "sample_index": dataset_sample_index,
                                    "nmse_linear": float(value),
                                    "error_energy": energy["error_energy"],
                                    "target_energy": energy["target_energy"],
                                }
                            )
                            dataset_sample_index += 1
                    if getattr(args, "save_controller_diagnostics", False):
                        diagnostics = model.get_last_controller_diagnostics()
                        if diagnostics is not None:
                            descriptor = diagnostics["descriptor"].cpu()
                            scales = diagnostics["scale"].cpu()
                            feature_names = diagnostics["feature_names"]
                            for sample_index in range(descriptor.shape[0]):
                                shared = {
                                    "dataset_name": dataset_name,
                                    "split": split,
                                    "mask_type": mask_name,
                                    "sample_index": (
                                        dataset_seen - samples.shape[0] + sample_index
                                    ),
                                }
                                descriptor_values = {
                                    name: float(descriptor[sample_index, idx])
                                    for idx, name in enumerate(feature_names)
                                }
                                for axis_index, axis in enumerate(("t", "k", "u")):
                                    axis_scale = scales[sample_index, axis_index]
                                    head_scales = axis_scale.reshape(
                                        axis_scale.shape[0], -1
                                    ).mean(dim=1)
                                    head_values = {
                                        f"scale_h{head_index}": float(value)
                                        for head_index, value in enumerate(head_scales)
                                    }
                                    lower_saturation = axis_scale <= (
                                        (1.0 / args.controller_max_scale) * 1.01
                                    )
                                    upper_saturation = axis_scale >= (
                                        args.controller_max_scale * 0.99
                                    )
                                    controller_rows.append(
                                        {
                                            **shared,
                                            "axis": axis,
                                            "scale_mean": float(axis_scale.mean()),
                                            "scale_std": float(axis_scale.std(unbiased=False)),
                                            "scale_min": float(axis_scale.min()),
                                            "scale_max": float(axis_scale.max()),
                                            "scale_saturation_fraction": float(
                                                (lower_saturation | upper_saturation)
                                                .float()
                                                .mean()
                                            ),
                                            "scale_lower_saturation_fraction": float(
                                                lower_saturation.float().mean()
                                            ),
                                            "scale_upper_saturation_fraction": float(
                                                upper_saturation.float().mean()
                                            ),
                                            **head_values,
                                            **descriptor_values,
                                        }
                                    )
                    max_batches = int(getattr(args, "max_eval_batches", 0))
                    if max_batches > 0 and num_batches >= max_batches:
                        break

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
                "mask_seed": mask_seed,
                "loss": avg_loss,
                "nmse_linear": avg_nmse_linear,
                "nmse_db": avg_nmse_db,
                "avg_inference_ms": avg_inference_ms,
                "peak_memory_mb": peak_memory_mb,
                "flops_per_batch": profiled_flops,
                **_parameter_summary(model),
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
    _write_spatial_rms_archive(args, spatial_rms_rows)
    return {
        "checkpoint": args.resume or getattr(args, "finetune", ""),
        "rope_mode": args.rope_mode,
        "mask_results": mask_results,
        "dataset_results": flattened_dataset_results,
        "overall_nmse_linear": overall_nmse_linear,
        "overall_nmse_db": overall_nmse_db,
        "controller_diagnostics": controller_rows,
        "sample_nmse": sample_rows,
    }


def _build_log_writer(args):
    if not misc.is_main_process():
        return None
    log_dir = args.log_dir or args.output_dir
    _ensure_dir(log_dir)
    return SummaryWriter(log_dir=log_dir)


def _save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler, filename=None):
    if args.output_dir:
        if str(getattr(args, "rope_oracle_scale_axis", "none")) != "none":
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / (
                filename if filename is not None else f"checkpoint-{epoch}.pth"
            )
            oracle_state = {
                name: value.detach().cpu()
                for name, value in model_without_ddp.state_dict().items()
                if name.startswith("rope_oracle_logits_")
            }
            payload = {
                "model": oracle_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict() if loss_scaler is not None else {},
                "args": args,
                "oracle_scale_only": True,
            }
            if misc.is_main_process():
                torch.save(payload, destination)
            return
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            filename=filename,
        )


@torch.no_grad()
def _calibrate_controller(args, model, model_without_ddp, device, train_loaders):
    standardizers = model_without_ddp.controller_standardizers()
    batches_per_domain = int(getattr(args, "controller_calibration_batches", 0))
    if not standardizers or batches_per_domain <= 0:
        return
    print(
        f"Calibrating controller descriptors with {batches_per_domain} "
        "batches per domain and mask strategy",
        flush=True,
    )
    model_without_ddp.begin_controller_calibration()
    model.train()
    mask_settings = (("random", 0.85), ("temporal", 0.5), ("freq", 0.5))
    for loader in train_loaders:
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches_per_domain:
                break
            if len(batch) == 4:
                samples, token_length, input_size, phys_meta = batch
                phys_meta = phys_meta.to(device, non_blocking=True)
            else:
                samples, token_length, input_size = batch
                phys_meta = None
            samples = samples.to(device, non_blocking=True)
            token_length = token_length.to(device, non_blocking=True)
            for strategy, ratio in mask_settings:
                model(
                    samples,
                    token_length,
                    input_size=input_size,
                    mask_ratio=ratio,
                    mask_strategy=strategy,
                    phys_meta=phys_meta,
                )
    if misc.is_dist_avail_and_initialized():
        for standardizer in standardizers:
            torch.distributed.all_reduce(standardizer.calibration_sum)
            torch.distributed.all_reduce(standardizer.calibration_sq_sum)
            torch.distributed.all_reduce(standardizer.calibration_count)
    model_without_ddp.finalize_controller_calibration()
    if misc.is_main_process():
        for index, standardizer in enumerate(standardizers):
            print(
                {
                    "controller_standardizer": index,
                    "samples": int(standardizer.calibration_count.item()),
                    "variance_min": float(standardizer.running_var.min().item()),
                    "variance_max": float(standardizer.running_var.max().item()),
                },
                flush=True,
            )


def run_train(args):
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)
    _write_run_manifest(args, device, "train")

    train_loaders = _build_train_loaders(args)
    val_loaders = _build_eval_loaders(args, split="val") if misc.is_main_process() else []
    log_writer = _build_log_writer(args)

    model = _build_model(args, device)
    model._public_profile = args.profile
    model.to(device)
    probe_hooks = _configure_rope_probe(args, model)
    _install_rope_oracle_scale(args, model, freeze_model=True)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module

    if (
        not getattr(args, "resume", "")
        and str(getattr(args, "rope_probe_train_axis", "none")) == "none"
        and str(getattr(args, "rope_oracle_scale_axis", "none")) == "none"
    ):
        _calibrate_controller(args, model, model_without_ddp, device, train_loaders)

    effective_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    optimizer, loss_scaler = _configure_optimizer(args, model_without_ddp, effective_batch_size)
    _maybe_resume_or_finetune(args, model_without_ddp, optimizer, loss_scaler)
    frozen_probe_initial_hash = _write_frozen_probe_integrity(
        args, model_without_ddp
    )

    start_time = time.time()
    metrics_path = _metric_csv_path(args)
    best_validation_nmse = float("inf")

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
        snapshot_requested = epoch in {
            int(value.strip())
            for value in str(args.controller_snapshot_epochs).split(",")
            if value.strip()
        }
        if snapshot_requested and args.distributed:
            torch.distributed.barrier()
        if snapshot_requested and misc.is_main_process():
            _write_controller_snapshot(
                args, model_without_ddp, epoch, train_loaders, device
            )
        if snapshot_requested and args.distributed:
            torch.distributed.barrier()

        should_save = (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs
        if should_save:
            _save_checkpoint(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        if should_save and misc.is_main_process() and val_loaders:
            val_metrics = evaluate_model(args, model_without_ddp, device, val_loaders, split="val")
            if val_metrics["overall_nmse_linear"] < best_validation_nmse:
                best_validation_nmse = val_metrics["overall_nmse_linear"]
                _save_checkpoint(
                    args, epoch, model, model_without_ddp, optimizer, loss_scaler,
                    filename="checkpoint-best.pth",
                )
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
    final_frozen_hash = _write_frozen_probe_integrity(
        args, model_without_ddp, initial_hash=frozen_probe_initial_hash
    )
    if (
        frozen_probe_initial_hash is not None
        and final_frozen_hash != frozen_probe_initial_hash
    ):
        raise RuntimeError("Frozen positional-probe model state changed during training")
    for hook in probe_hooks:
        hook.remove()

    if log_writer is not None:
        log_writer.close()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")


def run_finetune(args):
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)
    _write_run_manifest(args, device, "finetune")

    train_loader, num_train = _build_finetune_loader(args)
    log_writer = _build_log_writer(args)

    model = _build_model(args, device)
    model._public_profile = args.profile
    model.to(device)
    if getattr(args, "finetune", "") and not getattr(args, "resume", ""):
        _load_checkpoint(args.finetune, model, strict=getattr(args, "strict_load", False))
    if getattr(args, "finetune_mode", "full") == "adapter":
        if getattr(args, "transfer_adapter_ratio", 0.0) <= 0:
            raise ValueError("adapter fine-tuning requires --transfer_adapter_ratio > 0")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("transfer_adapter.")
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if trainable == 0:
            raise ValueError("adapter fine-tuning selected but no adapter parameters are trainable")
        print({"finetune_mode": "adapter", "trainable_parameters": trainable}, flush=True)
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
    if getattr(args, "resume", ""):
        _maybe_resume_or_finetune(args, model_without_ddp, optimizer, loss_scaler)

    start_time = time.time()
    metrics_path = _metric_csv_path(args)
    val_loaders = _build_eval_loaders(args, split="val") if misc.is_main_process() else []
    best_validation_nmse = float("inf")

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
        if misc.is_main_process() and val_loaders:
            val_metrics = evaluate_model(args, model_without_ddp, device, val_loaders, split="val")
            if val_metrics["overall_nmse_linear"] < best_validation_nmse:
                best_validation_nmse = val_metrics["overall_nmse_linear"]
                _save_checkpoint(
                    args,
                    epoch,
                    model,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    filename="checkpoint-best.pth",
                )
            for mask_name, mask_metrics in val_metrics["mask_results"].items():
                _append_metric_rows(
                    metrics_path,
                    [
                        {
                            "epoch": epoch,
                            "split": "val",
                            "mask_type": mask_name,
                            "train_loss": "",
                            "lr": "",
                            "avg_nmse_linear": mask_metrics["avg_nmse_linear"],
                            "avg_nmse_db": mask_metrics["avg_nmse_db"],
                        }
                    ],
                )

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
    _write_run_manifest(args, device, "eval")

    model = _build_model(args, device)
    model._public_profile = args.profile
    model.to(device)
    _install_rope_oracle_scale(args, model, freeze_model=False)
    _load_checkpoint(args.resume, model, strict=getattr(args, "strict_load", False))
    if getattr(args, "rope_oracle_checkpoint", ""):
        _load_checkpoint(args.rope_oracle_checkpoint, model)

    if getattr(args, "fast_inference", False) and hasattr(model, "set_fast_inference"):
        model.set_fast_inference(True)

    if not misc.is_main_process():
        return

    split = getattr(args, "eval_split", "test")
    eval_loaders = _build_eval_loaders(args, split=split)
    compile_capture_previous = None
    if getattr(args, "torch_compile", False):
        if device.type != "cuda":
            raise RuntimeError("--torch_compile requires CUDA")
        first_dims = next(iter(eval_loaders[0]))[2]
        if hasattr(model, "set_inference_grid_size"):
            patches = model.input_size[:3]
            model.set_inference_grid_size(
                tuple(
                    int(value.max().item()) // patch
                    for value, patch in zip(first_dims, patches)
                )
            )
        compile_capture_previous = torch._dynamo.config.capture_scalar_outputs
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(
            model, mode=args.compile_mode, fullgraph=False, dynamic=False
        )
    try:
        metrics = evaluate_model(args, model, device, eval_loaders, split=split)
    finally:
        if compile_capture_previous is not None:
            torch._dynamo.config.capture_scalar_outputs = compile_capture_previous
    _write_eval_outputs(Path(args.output_dir), metrics)


def run_calibrate_controller(args):
    """Clone a checkpoint after balanced input-only controller calibration."""
    misc.init_distributed_mode(args)
    device = _resolve_device(args)
    _ensure_dir(args.output_dir)
    _set_seed(args)
    _write_run_manifest(args, device, "calibrate_controller")

    model = _build_model(args, device)
    model._public_profile = args.profile
    model.to(device)
    _load_checkpoint(args.resume, model)
    preserved_descriptor_statistics = {}
    if getattr(args, "preserve_descriptor_statistics", False):
        for name in ("enc_rope_controller", "dec_rope_controller"):
            controller = getattr(model, name, None)
            standardizer = getattr(controller, "standardizer", None)
            if standardizer is not None:
                preserved_descriptor_statistics[name] = {
                    key: value.detach().clone()
                    for key, value in standardizer.state_dict().items()
                }
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module
    train_loaders = _build_train_loaders(args)
    _calibrate_controller(args, model, model_without_ddp, device, train_loaders)
    for name, state in preserved_descriptor_statistics.items():
        getattr(model_without_ddp, name).standardizer.load_state_dict(state, strict=True)

    if misc.is_main_process():
        checkpoint = load_torch_checkpoint(args.resume, map_location="cpu")
        checkpoint["model"] = model_without_ddp.state_dict()
        checkpoint["controller_recalibrated_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        checkpoint["descriptor_statistics_preserved"] = bool(
            preserved_descriptor_statistics
        )
        destination = Path(args.output_dir) / "checkpoint-calibrated.pth"
        torch.save(checkpoint, destination)
        print(f"Saved calibrated controller checkpoint to {destination}", flush=True)
