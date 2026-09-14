"""DeepMIMO downstream transfer and specialist-baseline runtime."""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.deepmimo.csi_grid import batch_target_grid, flatten_grid
from datasets.deepmimo.task_data import TaskCSIDataset
from datasets.deepmimo.task_metrics import binary_classification_metrics, nmse, one_db_margin_accuracy, rsrp_gap, topk_accuracy
from models.baselines.deepmimo_feedback import CSINetAdapter, CsiNetLSTMAdapter, TransNetAdapter
from models.downstream.task_heads import (
    BeamBaselineModel,
    BeamManagementAdapterModel,
    CSIFeedbackAdapterModel,
    MAEBackboneAdapter,
    LoSAdapterModel,
    LoSBaselineModel,
    configure_downstream_trainability,
)
from util.runtime import _build_model
from util.checkpoint import load_model_checkpoint_strict
from tools.final_method import FINAL_ADAPTIVE_CHECKPOINT_SHA256, FINAL_MODEL_COMMIT


def _device(args):
    return torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")


def _loader(args, split, shuffle=False):
    args.data_source = "deepmimo_v4"
    dataset = TaskCSIDataset(args, split=split, task=args.task)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=TaskCSIDataset.collate, drop_last=False, pin_memory=True,
    )


def _load_mae(args, device):
    mae = _build_model(args, device)
    if getattr(args, "profile", None):
        mae._public_profile = args.profile
    if args.pretrained:
        load_model_checkpoint_strict(
            mae,
            args.pretrained,
            require_calibration_frozen=(getattr(mae, "rope_mode", None) == "adaptive"),
        )
    return mae


def _build(args, dataset, device):
    if args.task == "los_classification" and args.baseline != "ours":
        if args.baseline not in {"mlp", "cnn", "lstm"}:
            raise ValueError(f"Invalid LoS baseline: {args.baseline}")
        return LoSBaselineModel(
            args.baseline, int(dataset.x.shape[-1]), args.baseline_hidden_dim
        ).to(device)
    if args.task == "beam_management" and args.baseline != "ours":
        return BeamBaselineModel(
            args.baseline, args.set_a_size, args.set_b_size, args.beam_rsrp_loss_weight
        ).to(device)
    if args.task == "csi_feedback" and args.baseline != "ours":
        classes = {"csinet": CSINetAdapter, "csinet_lstm": CsiNetLSTMAdapter, "transnet": TransNetAdapter}
        if args.baseline not in classes:
            raise ValueError(f"Invalid CSI feedback baseline: {args.baseline}")
        model = classes[args.baseline](compression_ratio=args.csi_compression_ratio).to(device)
        model.build_from_dataset(dataset, device)
        return model
    backbone = MAEBackboneAdapter(
        _load_mae(args, device), args.adapter_bottleneck_ratio, args.adapter_depth
    )
    if args.task == "beam_management":
        model = BeamManagementAdapterModel(
            backbone, args.set_a_size, args.set_b_size, True, args.beam_rsrp_loss_weight
        )
    elif args.task == "los_classification":
        model = LoSAdapterModel(backbone)
    else:
        model = CSIFeedbackAdapterModel(
            backbone,
            compression_ratio=args.csi_compression_ratio,
            seq_len=int(dataset.x.shape[1]),
            target_dim=int(dataset.x.shape[2] * 2),
            source_dim=int(np.prod(dataset.orig_dims[0]) * 2),
            codec_architecture=getattr(args, "csi_codec_architecture", "token_lowrank"),
            codec_hidden_dim=getattr(args, "csi_codec_hidden_dim", 64),
        )
    configure_downstream_trainability(model, args.adaptation_mode)
    return model.to(device)


def _move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, loader, args, device):
    model.eval()
    start = time.perf_counter()
    if args.task == "csi_feedback":
        predictions, targets, valids = [], [], []
        for batch in loader:
            batch = _move(batch, device)
            _, pred = model(batch)
            target_grid, valid_grid = batch_target_grid(batch)
            target, valid = flatten_grid(target_grid, valid_grid)
            predictions.append(pred.cpu()); targets.append(target.cpu()); valids.append(valid.cpu())
        linear, db = nmse(torch.cat(predictions), torch.cat(targets), torch.cat(valids))
        result = {"nmse_linear": linear, "nmse_db": db}
    elif args.task == "beam_management":
        logits_all, label_all, rsrp_all = [], [], []
        for batch in loader:
            batch = _move(batch, device)
            _, logits, _ = model(batch)
            logits_all.append(logits.cpu()); label_all.append(batch["label"].cpu()); rsrp_all.append(batch["full_rsrp"].cpu())
        logits, labels, rsrp = torch.cat(logits_all), torch.cat(label_all), torch.cat(rsrp_all)
        result = {
            **topk_accuracy(logits, labels, ks=(1, 5)),
            "one_db_margin_accuracy": one_db_margin_accuracy(logits, rsrp, labels),
            **rsrp_gap(logits, rsrp, labels),
        }
    else:
        logits_all, label_all = [], []
        for batch in loader:
            batch = _move(batch, device)
            _, logits = model(batch)
            logits_all.append(logits.cpu()); label_all.append(batch["label"].cpu())
        result = binary_classification_metrics(torch.cat(logits_all), torch.cat(label_all))
    result["inference_seconds"] = time.perf_counter() - start
    result["samples"] = len(loader.dataset)
    return result


def _manifest(args, model, phase):
    groups = {
        "pretrained_encoder": ("backbone.mae.",),
        "adapter": ("backbone.adapter.",),
    }
    breakdown = {}
    named_parameters = list(model.named_parameters())
    assigned = set()
    for group, prefixes in groups.items():
        members = [(name, parameter) for name, parameter in named_parameters if name.startswith(prefixes)]
        assigned.update(name for name, _ in members)
        breakdown[group] = {
            "parameters": sum(parameter.numel() for _, parameter in members),
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in members if parameter.requires_grad
            ),
        }
    task_head = [(name, parameter) for name, parameter in named_parameters if name not in assigned]
    breakdown["task_head"] = {
        "parameters": sum(parameter.numel() for _, parameter in task_head),
        "trainable_parameters": sum(
            parameter.numel() for _, parameter in task_head if parameter.requires_grad
        ),
    }
    checkpoint = Path(args.pretrained).expanduser().resolve() if args.pretrained else None
    checkpoint_record = {"path": str(checkpoint)} if checkpoint else {}
    if checkpoint and checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_record["sha256"] = digest.hexdigest()
        if checkpoint_record["sha256"] == FINAL_ADAPTIVE_CHECKPOINT_SHA256:
            checkpoint_record.update(source_commit=FINAL_MODEL_COMMIT, formal_adaptive=True)
    return {
        "phase": phase,
        "arguments": vars(args),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_breakdown": breakdown,
        "pretrained_checkpoint": args.pretrained,
        "pretrained_checkpoint_provenance": checkpoint_record,
    }


def run_downstream_train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = _device(args)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = _loader(args, "train", True), _loader(args, "val", False)
    model = _build(args, train_loader.dataset, device)
    (output / "run_manifest.json").write_text(json.dumps(_manifest(args, model, "downstream_train"), indent=2))
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.mae")]
    encoder = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.mae")]
    groups = [{"params": head, "lr": args.task_lr}]
    if encoder:
        groups.append({"params": encoder, "lr": args.encoder_lr})
    if not any(group["params"] for group in groups):
        metrics = _evaluate(model, val_loader, args, device)
        state = {
            "model": model.state_dict(),
            "epoch": 0,
            "args": vars(args),
            "validation": metrics,
        }
        torch.save(state, output / "checkpoint-best.pth")
        torch.save(state, output / "checkpoint-final.pth")
        (output / "history.json").write_text(
            json.dumps([{"epoch": 0, "train_loss": None, **metrics}], indent=2)
        )
        print(json.dumps({"epoch": 0, "train_loss": None, **metrics}), flush=True)
        return
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    accum_iter = max(1, int(getattr(args, "accum_iter", 1)))
    best = float("inf") if args.task == "csi_feedback" else -float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = _move(batch, device)
            output_values = model(batch); raw_loss = output_values[0]
            if not torch.isfinite(raw_loss):
                raise ValueError(f"Non-finite downstream loss at epoch {epoch}")
            loss = raw_loss / accum_iter
            loss.backward()
            should_update = (step + 1) % accum_iter == 0 or (step + 1) == len(train_loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach()))
        metrics = _evaluate(model, val_loader, args, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(row)
        value = metrics["nmse_linear"] if args.task == "csi_feedback" else (
            metrics["macro_f1"] if args.task == "los_classification" else metrics["top1_accuracy"]
        )
        improved = value < best if args.task == "csi_feedback" else value > best
        if improved:
            best = value
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "validation": metrics}, output / "checkpoint-best.pth")
        else:
            stale_epochs += 1
        print(json.dumps(row), flush=True)
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            break
    torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"], "args": vars(args)}, output / "checkpoint-final.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2))


def run_downstream_eval(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = _device(args)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    loader = _loader(args, args.eval_split, False)
    model = _build(args, loader.dataset, device)
    if not args.resume:
        raise ValueError("downstream_eval requires --resume")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = _evaluate(model, loader, args, device)
    payload = {**_manifest(args, model, "downstream_eval"), "metrics": metrics}
    (output / "eval_metrics.json").write_text(json.dumps(payload, indent=2))
