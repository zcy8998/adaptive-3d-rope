#!/usr/bin/env python3
"""Fine-tune LWM/Adaptive 2D-RoPE on K=64 beam prediction and test K=128 zero-shot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.lwm_beam import LwmBeamModel


class BeamTokens(Dataset):
    def __init__(self, root: Path, split: str, *, subset: np.ndarray | None = None):
        self.tokens = np.load(root / f"{split}_tokens.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r")
        self.full_rsrp = np.load(root / f"{split}_full_rsrp.npy", mmap_mode="r")
        self.source_row = np.load(root / f"{split}_source_row.npy", mmap_mode="r")
        if not (len(self.tokens) == len(self.labels) == len(self.full_rsrp) == len(self.source_row)):
            raise ValueError(f"Mismatched beam arrays in {root} split={split}")
        self.indices = np.arange(len(self.tokens), dtype=np.int64) if subset is None else np.asarray(subset, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        cls = np.full((1, self.tokens.shape[-1]), 0.2, dtype=np.float32)
        values = np.concatenate((cls, np.asarray(self.tokens[index], dtype=np.float32)), axis=0)
        return (
            torch.from_numpy(values),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            torch.from_numpy(np.asarray(self.full_rsrp[index], dtype=np.float32)),
            torch.tensor(int(self.source_row[index]), dtype=torch.long),
        )


def coordinates(antennas: int, subcarriers: int, device: torch.device) -> torch.Tensor:
    if antennas % 4 or subcarriers % 4:
        raise ValueError("K and U must be divisible by the 4x4 patch")
    array_patches, frequency_patches = antennas // 4, subcarriers // 4
    array = torch.arange(array_patches).repeat_interleave(frequency_patches)
    frequency = torch.arange(frequency_patches).repeat(array_patches)
    grid = torch.stack((array, frequency), dim=-1)
    return torch.cat((torch.zeros(1, 2, dtype=grid.dtype), grid), dim=0).to(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(prediction: np.ndarray, labels: np.ndarray, classes: int) -> float:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels.astype(np.int64), prediction.astype(np.int64)), 1)
    tp = np.diag(matrix).astype(np.float64)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = matrix.sum(axis=1) > 0
    return float(f1[present].mean()) if present.any() else 0.0


def evaluate(model, loader, coords, device, num_beams: int) -> tuple[dict, dict]:
    model.eval()
    prediction_rows = []
    label_rows = []
    rsrp_rows = []
    source_rows = []
    top5_rows = []
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for inputs, labels, full_rsrp, source_row in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            full_rsrp = full_rsrp.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, logits, _ = model(inputs, coords, labels)
            pred = logits.argmax(dim=-1)
            prediction_rows.append(pred.cpu().numpy())
            top5_rows.append(logits.topk(min(5, num_beams), dim=-1).indices.cpu().numpy())
            label_rows.append(labels.cpu().numpy())
            rsrp_rows.append(full_rsrp.cpu().numpy())
            source_rows.append(source_row.numpy())
            total_loss += float(loss.detach()) * len(labels)
            total_count += len(labels)
    prediction = np.concatenate(prediction_rows)
    labels = np.concatenate(label_rows)
    full_rsrp = np.concatenate(rsrp_rows)
    source_row = np.concatenate(source_rows)
    order = np.argsort(source_row)
    prediction, labels, full_rsrp, source_row = prediction[order], labels[order], full_rsrp[order], source_row[order]
    top1 = float(np.mean(prediction == labels))
    top5 = np.concatenate(top5_rows)[order]
    best = full_rsrp[np.arange(len(labels)), labels]
    chosen = full_rsrp[np.arange(len(labels)), prediction]
    gap = best - chosen
    metrics = {
        "loss": total_loss / max(1, total_count),
        "macro_f1": macro_f1(prediction, labels, num_beams),
        "top1_accuracy": top1,
        "top5_accuracy": float(np.mean((top5 == labels[:, None]).any(axis=1))),
        "one_db_margin_accuracy": float(np.mean(gap <= 1.0)),
        "avg_rsrp_gap_db": float(gap.mean()),
        "p50_rsrp_gap_db": float(np.quantile(gap, 0.50)),
        "p90_rsrp_gap_db": float(np.quantile(gap, 0.90)),
        "samples": int(len(labels)),
    }
    raw = {
        "source_row": source_row,
        "prediction": prediction,
        "top5": top5,
        "label": labels,
        "full_rsrp": full_rsrp,
    }
    return metrics, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("original", "proposed"), required=True)
    parser.add_argument("--display-method", choices=("LWM", "Adaptive 2D-RoPE"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--controller-statistics", choices=("mean_std", "std"), default="std")
    args = parser.parse_args()
    if args.output.joinpath("COMPLETE").exists():
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train_root = args.data_root / "K0064"
    k128_root = args.data_root / "K0128"
    train_set = BeamTokens(train_root, "train")
    val_set = BeamTokens(train_root, "val")
    seen_test = BeamTokens(train_root, "test")
    zero_test = BeamTokens(k128_root, "test")
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    seen_loader = DataLoader(seen_test, shuffle=False, **loader_kwargs)
    zero_loader = DataLoader(zero_test, shuffle=False, **loader_kwargs)
    coords64 = coordinates(16, 64, device)
    coords128 = coordinates(16, 128, device)
    model = LwmBeamModel(args.method, str(args.checkpoint), controller_statistics=args.controller_statistics).to(device)
    smoke_inputs, smoke_labels, _, _ = next(iter(train_loader))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        smoke_loss, smoke_logits, _ = model(smoke_inputs.to(device), coords64, smoke_labels.to(device))
    smoke_loss.backward()
    if not torch.isfinite(smoke_loss) or smoke_logits.shape[-1] != 256:
        raise RuntimeError(f"Invalid beam smoke: loss={smoke_loss} shape={smoke_logits.shape}")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    history_path = args.output / "history.csv"
    best_value = float("inf")
    best_epoch = -1
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_loss", "val_macro_f1", "lr_encoder", "lr_head"))
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            running = 0.0
            count = 0
            for inputs, labels, _, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, _, _ = model(inputs.to(device), coords64, labels.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += float(loss.detach()) * len(labels)
                count += len(labels)
            val_metrics, _ = evaluate(model, val_loader, coords64, device, 256)
            row = {
                "epoch": epoch,
                "train_loss": running / max(1, count),
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr_encoder": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[1]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "method": args.method,
                "display_method": args.display_method,
                "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "validation": val_metrics,
            }
            torch.save(state, args.output / "checkpoint-last.pth")
            if val_metrics["loss"] < best_value:
                best_value = val_metrics["loss"]
                best_epoch = epoch
                torch.save(state, args.output / "checkpoint-best.pth")
            print(json.dumps(row), flush=True)
    best = torch.load(args.output / "checkpoint-best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"], strict=True)
    seen_metrics, seen_raw = evaluate(model, seen_loader, coords64, device, 256)
    zero_metrics, zero_raw = evaluate(model, zero_loader, coords128, device, 256)
    np.savez_compressed(args.output / "seen_test_predictions.npz", **seen_raw)
    np.savez_compressed(args.output / "k128_zero_shot_predictions.npz", **zero_raw)
    summary = {
        "status": "complete",
        "method": args.method,
        "display_method": args.display_method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_sha256": sha256(args.output / "checkpoint-best.pth"),
        "pretraining_checkpoint": str(args.checkpoint),
        "pretraining_checkpoint_sha256": sha256(args.checkpoint),
        "seen_K0064": seen_metrics,
        "zero_shot_K0128": zero_metrics,
        "strict_load": True,
        "protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
