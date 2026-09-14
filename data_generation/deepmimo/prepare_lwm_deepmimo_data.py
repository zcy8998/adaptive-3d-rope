#!/usr/bin/env python3
"""Materialize paired LWM 1.1 patch tokens from the existing DeepMIMO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CONFIGS = (
    (8, 64), (16, 64), (32, 64), (64, 64), (128, 64),
    (16, 32), (16, 128), (16, 256), (16, 512),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_tokens(
    csi: np.ndarray,
    antennas: int,
    subcarriers: int,
    channel_scale: float = 1e6,
) -> np.ndarray:
    if antennas % 4 or subcarriers % 4:
        raise ValueError("LWM portability configurations must be divisible by the 4x4 patch")
    channel = csi[:, 0, :subcarriers, :antennas].transpose(0, 2, 1) * channel_scale
    batch = channel.shape[0]
    patches = channel.reshape(batch, antennas // 4, 4, subcarriers // 4, 4)
    patches = patches.transpose(0, 1, 3, 2, 4).reshape(batch, -1, 16)
    output = np.empty((*patches.shape[:2], 32), dtype=np.float32)
    output[..., 0::2] = patches.real
    output[..., 1::2] = patches.imag
    return output


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def materialize(source: Path, destination: Path, antennas: int, subcarriers: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for split in ("train", "val", "test"):
        output = destination / f"{split}_tokens.npy"
        if not output.exists():
            with np.load(source / f"{split}.npz", allow_pickle=True) as payload:
                csi = payload["csi"]
                if csi.shape[2] < subcarriers or csi.shape[3] < antennas:
                    return {
                        "status": "requires_raw_generation",
                        "available_shape": list(csi.shape),
                    }
                channel_power = np.mean(np.abs(csi[:, 0]) ** 2, axis=(1, 2))
                source_rows = np.flatnonzero(channel_power > 0).astype(np.int64)
                tokens = patch_tokens(csi[source_rows], antennas, subcarriers)
            temporary = output.with_suffix(f".npy.{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, tokens, allow_pickle=False)
            os.replace(temporary, output)
            row_path = destination / f"{split}_source_row.npy"
            with row_path.open("wb") as handle:
                np.save(handle, source_rows, allow_pickle=False)
        array = np.load(output, mmap_mode="r")
        row_path = destination / f"{split}_source_row.npy"
        source_rows = np.load(row_path, mmap_mode="r")
        records[split] = {
            "path": str(output.resolve()),
            "shape": list(array.shape),
            "sha256": sha256(output),
            "source_rows": str(row_path.resolve()),
            "valid_samples": int(len(source_rows)),
            "excluded_no_path": int({"train": 16000, "val": 4000, "test": 4000}[split] - len(source_rows)),
        }
    ready = destination / "READY"
    ready.touch()
    return {"status": "ready", "splits": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset-seed", default=20260724, type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "source_split_sizes": {"train": 16000, "val": 4000, "test": 4000},
        "subset_seed": args.subset_seed,
        "patch": [4, 4],
        "channel_scale": 1e6,
        "no_path_policy": "exclude samples with exactly zero channel power, matching LWM preprocessing",
        "configs": {},
    }
    # Base first so GPU training can start while the remaining views are materialized.
    ordered = ((16, 64),) + tuple(config for config in CONFIGS if config != (16, 64))
    for antennas, subcarriers in ordered:
        name = f"U{antennas:03d}_K{subcarriers:04d}"
        result = materialize(args.source, args.output_root / name, antennas, subcarriers)
        manifest["configs"][name] = result
        if (antennas, subcarriers) == (16, 64) and result["status"] == "ready":
            valid_train = result["splits"]["train"]["valid_samples"]
            training_users = int(0.1 * valid_train)
            subset_path = args.output_root / "train_subset_10pct.npy"
            if not subset_path.exists():
                indices = np.random.default_rng(args.subset_seed).permutation(valid_train)[:training_users]
                with subset_path.open("wb") as handle:
                    np.save(handle, np.sort(indices).astype(np.int64), allow_pickle=False)
            manifest["training_users"] = training_users
            manifest["subset_path"] = str(subset_path.resolve())
        atomic_json(args.output_root / "manifest.json", manifest)
        print(json.dumps({"config": name, **result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
