#!/usr/bin/env python3
"""Evaluate full-data LWM checkpoints on paired frequency-scale OOD data."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.lwm_adaptive_2d import LwmPortabilityModel


METHODS = ("original", "proposed")
SEEDS = (40, 41, 42)
CONFIG_PATTERN = re.compile(r"^U(?P<antennas>\d{3})_K(?P<subcarriers>\d{4})$")


def parse_config(name: str) -> tuple[int, int]:
    match = CONFIG_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid LWM configuration name: {name!r}")
    antennas = int(match.group("antennas"))
    subcarriers = int(match.group("subcarriers"))
    if antennas % 4 or subcarriers % 4:
        raise ValueError("LWM antenna and subcarrier counts must be divisible by four")
    return antennas, subcarriers


def make_coordinates(
    antennas: int, subcarriers: int, device: torch.device
) -> torch.Tensor:
    array_patches = antennas // 4
    frequency_patches = subcarriers // 4
    array = torch.arange(array_patches).repeat_interleave(frequency_patches)
    frequency = torch.arange(frequency_patches).repeat(array_patches)
    grid = torch.stack((array, frequency), dim=-1)
    cls = torch.zeros(1, 2, dtype=grid.dtype)
    return torch.cat((cls, grid), dim=0).to(device)


class MaskedOodTokens(Dataset):
    def __init__(
        self,
        tokens_path: Path,
        source_rows_path: Path,
        mask_seed: int,
        mask_fraction: float = 0.4,
    ) -> None:
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.source_rows = np.load(source_rows_path, mmap_mode="r")
        if len(self.tokens) != len(self.source_rows):
            raise ValueError("Token and source-row counts differ")
        self.mask_seed = int(mask_seed)
        self.mask_fraction = float(mask_fraction)
        self.mask_count = max(1, int(self.mask_fraction * self.tokens.shape[1]))

    def __len__(self) -> int:
        return len(self.tokens)

    def mask_positions(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(self.mask_seed + int(index) * 1000003)
        return np.sort(
            rng.choice(self.tokens.shape[1], self.mask_count, replace=False)
        ).astype(np.int64)

    def __getitem__(self, index: int):
        source_index = int(index)
        tokens = np.asarray(self.tokens[source_index]).copy()
        rng = np.random.default_rng(self.mask_seed + source_index * 1000003)
        masked_zero = np.sort(
            rng.choice(tokens.shape[0], self.mask_count, replace=False)
        )
        target = tokens[masked_zero].copy()
        for position in masked_zero:
            draw = rng.random()
            if draw < 0.8:
                tokens[position] = 0.1
            elif draw < 0.9:
                tokens[position] = rng.random(tokens.shape[-1], dtype=np.float32)
        cls = np.full((1, tokens.shape[-1]), 0.2, dtype=np.float32)
        return (
            torch.from_numpy(np.concatenate((cls, tokens), axis=0)),
            torch.from_numpy(target),
            torch.from_numpy(masked_zero.astype(np.int64) + 1),
            torch.tensor(int(self.source_rows[source_index]), dtype=torch.int64),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def acquire_output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another LWM OOD evaluator owns {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def sample_nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().flatten(1).sum(dim=1)
    power = target.float().square().flatten(1).sum(dim=1).clamp_min(1e-12)
    return error / power


def load_checkpoint(
    method: str, seed: int, checkpoint: Path, device: torch.device
) -> tuple[LwmPortabilityModel, dict, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("method") != method or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
    protocol = payload.get("protocol", {})
    if protocol.get("initialization") != "scratch" or not protocol.get("full_train"):
        raise ValueError(f"Checkpoint is not a full-data scratch run: {checkpoint}")
    if not protocol.get("dynamic_train_mask"):
        raise ValueError(f"Checkpoint did not use dynamic training masks: {checkpoint}")
    controller_statistics = str(protocol.get("controller_statistics", "mean_std"))
    model = LwmPortabilityModel(
        method, controller_statistics=controller_statistics
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload, sha256_file(checkpoint)


def evaluate_model(
    model: LwmPortabilityModel,
    loader: DataLoader,
    coordinates: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    nmse_values: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    prediction_square_sum = 0.0
    target_square_sum = 0.0
    element_count = 0
    scale_min = float("inf")
    scale_max = float("-inf")
    with torch.inference_mode():
        for inputs, targets, masked_pos, indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masked_pos = masked_pos.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction, diagnostics = model(inputs, masked_pos, coordinates)
            flat_prediction = prediction.float().flatten(1)
            flat_target = targets.float().flatten(1)
            nmse_values.append(sample_nmse(prediction, targets).cpu())
            cosine_values.append(
                F.cosine_similarity(flat_prediction, flat_target, dim=1).cpu()
            )
            sample_ids.append(indices)
            prediction_square_sum += float(flat_prediction.square().sum())
            target_square_sum += float(flat_target.square().sum())
            element_count += flat_target.numel()
            scales = diagnostics.get("scales")
            if scales is not None:
                scale_min = min(scale_min, float(scales.min()))
                scale_max = max(scale_max, float(scales.max()))
    linear_nmse = torch.cat(nmse_values).numpy()
    cosine = torch.cat(cosine_values).numpy()
    ids = torch.cat(sample_ids).numpy()
    prediction_rms = math.sqrt(prediction_square_sum / max(1, element_count))
    target_rms = math.sqrt(target_square_sum / max(1, element_count))
    return {
        "sample_id": ids,
        "linear_nmse": linear_nmse,
        "cosine_similarity": cosine,
        "prediction_rms": prediction_rms,
        "target_rms": target_rms,
        "rms_ratio": prediction_rms / max(target_rms, 1e-12),
        "scale_min": None if math.isinf(scale_min) else scale_min,
        "scale_max": None if math.isinf(scale_max) else scale_max,
    }


def comparison_summary(
    values: dict[tuple[str, int], np.ndarray],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    original = np.stack([values[("original", seed)] for seed in SEEDS])
    proposed = np.stack([values[("proposed", seed)] for seed in SEEDS])
    if original.shape != proposed.shape:
        raise ValueError("Original and Proposed sample arrays are not paired")
    per_seed = []
    for index, seed in enumerate(SEEDS):
        original_mean = float(original[index].mean())
        proposed_mean = float(proposed[index].mean())
        per_seed.append(
            {
                "seed": seed,
                "original_linear_nmse": original_mean,
                "proposed_linear_nmse": proposed_mean,
                "original_nmse_db": 10.0 * math.log10(original_mean),
                "proposed_nmse_db": 10.0 * math.log10(proposed_mean),
                "gain_db": 10.0 * math.log10(original_mean / proposed_mean),
            }
        )
    original_mean = float(original.mean(axis=1).mean())
    proposed_mean = float(proposed.mean(axis=1).mean())
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    sample_count = original.shape[1]
    chunk_size = 100
    for start in range(0, bootstrap_replicates, chunk_size):
        stop = min(start + chunk_size, bootstrap_replicates)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        original_draw = original[:, indices].mean(axis=(0, 2))
        proposed_draw = proposed[:, indices].mean(axis=(0, 2))
        bootstrap[start:stop] = 10.0 * np.log10(original_draw / proposed_draw)
    lower, upper = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "aggregation": "equal seed weight after sample aggregation in the linear domain",
        "bootstrap": {
            "type": "paired sample bootstrap conditional on the three trained seeds",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
        },
        "sample_count_per_seed": sample_count,
        "per_seed": per_seed,
        "overall": {
            "original_linear_nmse": original_mean,
            "proposed_linear_nmse": proposed_mean,
            "original_nmse_db": 10.0 * math.log10(original_mean),
            "proposed_nmse_db": 10.0 * math.log10(proposed_mean),
            "gain_db": 10.0 * math.log10(original_mean / proposed_mean),
            "gain_db_ci95": [float(lower), float(upper)],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", default="U016_K0128")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--mask-seed", default=20000042, type=int)
    parser.add_argument("--expected-samples", default=3969, type=int)
    parser.add_argument("--bootstrap-replicates", default=10000, type=int)
    parser.add_argument("--bootstrap-seed", default=20260727, type=int)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_lock = acquire_output_lock(args.output_root / ".output.lock")
    _held_lock = output_lock
    complete = args.output_root / "COMPLETE"
    if complete.is_file():
        print((args.output_root / "comparison_summary.json").read_text(), end="")
        return 0
    antennas, subcarriers = parse_config(args.config)
    config_root = args.data_root / args.config
    tokens_path = config_root / "test_tokens.npy"
    source_rows_path = config_root / "test_source_row.npy"
    if not (config_root / "READY").is_file():
        raise FileNotFoundError(f"OOD data is not ready: {config_root}")
    dataset = MaskedOodTokens(tokens_path, source_rows_path, args.mask_seed)
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, got {len(dataset)}")
    expected_tokens = (antennas // 4) * (subcarriers // 4)
    if dataset.tokens.shape[1:] != (expected_tokens, 32):
        raise ValueError(f"Unexpected token shape: {dataset.tokens.shape}")
    if dataset.mask_count != 51 or args.config != "U016_K0128":
        raise ValueError("Formal K128 evaluation requires 128 tokens and 51 masks")

    mask_positions = np.stack(
        [dataset.mask_positions(index) for index in range(len(dataset))]
    )
    source_rows = np.asarray(dataset.source_rows).copy()
    atomic_npz(
        args.output_root / "evaluation_mask.npz",
        sample_id=source_rows,
        masked_position_zero_based=mask_positions,
    )
    mask_manifest = {
        "config": args.config,
        "mask_seed": args.mask_seed,
        "mask_fraction": dataset.mask_fraction,
        "mask_count": dataset.mask_count,
        "samples": len(dataset),
        "tokens_per_sample": expected_tokens,
        "sequence_length_with_cls": expected_tokens + 1,
        "sample_ids_sha256": sha256_array(source_rows),
        "mask_positions_sha256": sha256_array(mask_positions),
        "test_tokens_sha256": sha256_file(tokens_path),
        "data_access": [str(tokens_path), str(source_rows_path)],
        "train_or_validation_accessed": False,
    }
    atomic_json(args.output_root / "mask_manifest.json", mask_manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal LWM OOD evaluation")
    device = torch.device("cuda:0")
    coordinates = make_coordinates(antennas, subcarriers, device)
    if coordinates.shape != (129, 2):
        raise ValueError(f"Unexpected coordinate shape: {tuple(coordinates.shape)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    values: dict[tuple[str, int], np.ndarray] = {}
    reference_ids: np.ndarray | None = None
    run_rows = []
    for method in METHODS:
        for seed in SEEDS:
            checkpoint = args.runs_root / method / f"seed_{seed}" / "checkpoint-best.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            model, payload, checkpoint_hash = load_checkpoint(
                method, seed, checkpoint, device
            )
            result = evaluate_model(model, loader, coordinates, device)
            ids = np.asarray(result["sample_id"])
            linear_nmse = np.asarray(result["linear_nmse"])
            cosine = np.asarray(result["cosine_similarity"])
            if reference_ids is None:
                reference_ids = ids.copy()
            elif not np.array_equal(reference_ids, ids):
                raise ValueError("Sample IDs differ between paired runs")
            if not np.array_equal(ids, source_rows):
                raise ValueError("Evaluation order differs from the registered test split")
            if not np.isfinite(linear_nmse).all() or not np.isfinite(cosine).all():
                raise ValueError(f"Non-finite evaluation output for {method}, seed {seed}")
            if float(result["rms_ratio"]) < 0.05:
                raise ValueError(f"Zero-prediction collapse for {method}, seed {seed}")
            values[(method, seed)] = linear_nmse
            run_root = args.output_root / "runs" / method / f"seed_{seed}"
            run_root.mkdir(parents=True, exist_ok=True)
            atomic_npz(
                run_root / "test_sample_nmse.npz",
                sample_id=ids,
                linear_nmse=linear_nmse,
                cosine_similarity=cosine,
            )
            run_summary = {
                "status": "complete",
                "evidence_status": "preliminary_50_epoch",
                "method": method,
                "seed": seed,
                "training_config": "U016_K0064",
                "evaluation_config": args.config,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": int(payload["epoch"]),
                "samples": len(linear_nmse),
                "linear_nmse": float(linear_nmse.mean()),
                "nmse_db": float(10.0 * np.log10(linear_nmse.mean())),
                "prediction_rms": result["prediction_rms"],
                "target_rms": result["target_rms"],
                "rms_ratio": result["rms_ratio"],
                "cosine_similarity": float(cosine.mean()),
                "scale_min": result["scale_min"],
                "scale_max": result["scale_max"],
                "mask_positions_sha256": mask_manifest["mask_positions_sha256"],
            }
            atomic_json(run_root / "summary.json", run_summary)
            run_rows.append(run_summary)
            del model
            torch.cuda.empty_cache()

    comparison = comparison_summary(
        values, args.bootstrap_replicates, args.bootstrap_seed
    )
    final_summary = {
        "schema_version": 1,
        "evidence_status": "preliminary_50_epoch",
        "training_config": "U016_K0064",
        "evaluation_config": args.config,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "batch_size": args.batch_size,
        "bf16": True,
        "inference_mode": True,
        "mask_manifest": mask_manifest,
        "comparison": comparison,
        "runs": run_rows,
    }
    atomic_json(args.output_root / "comparison_summary.json", final_summary)
    csv_path = args.output_root / "comparison_runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "original_linear_nmse",
                "proposed_linear_nmse",
                "original_nmse_db",
                "proposed_nmse_db",
                "gain_db",
            ),
        )
        writer.writeheader()
        writer.writerows(comparison["per_seed"])
    complete.touch()
    print(json.dumps(final_summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
