#!/usr/bin/env python3
"""Prepare CSI-only K=64 beam-transfer data and K=128 zero-shot test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_complex_csi(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if np.iscomplexobj(value):
        out = value
    elif value.ndim and value.shape[-1] == 2:
        out = value[..., 0] + 1j * value[..., 1]
    else:
        raise ValueError(f"CSI must be complex or have a final real/imag axis, got {value.shape}")
    if out.ndim == 3:
        out = out[:, None]
    if out.ndim != 4:
        raise ValueError(f"CSI must resolve to [N,T,K,U], got {out.shape}")
    return np.asarray(out, dtype=np.complex64)


def patch_tokens(csi: np.ndarray, k: int, u: int, channel_scale: float) -> np.ndarray:
    if csi.shape[2] < k or csi.shape[3] < u or k % 4 or u % 4:
        raise ValueError(f"Cannot make 4x4 patches from CSI shape {csi.shape} and K/U={k}/{u}")
    # The public LWM checkpoint and its reconstruction pretraining interface
    # operate on DeepMIMO CSI scaled by 1e6. Preserve that interface exactly
    # for the downstream beam task rather than feeding near-zero raw values.
    channel = csi[:, 0, :k, :u].transpose(0, 2, 1) * channel_scale
    patches = channel.reshape(channel.shape[0], u // 4, 4, k // 4, 4)
    patches = patches.transpose(0, 1, 3, 2, 4).reshape(channel.shape[0], -1, 16)
    tokens = np.empty((*patches.shape[:2], 32), dtype=np.float32)
    tokens[..., 0::2] = patches.real
    tokens[..., 1::2] = patches.imag
    return tokens


def write_array(path: Path, value: np.ndarray) -> dict:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)
    return {"path": str(path), "shape": list(value.shape), "sha256": sha256(path)}


def prepare_split(
    source: Path, destination: Path, k: int, u: int, split: str, channel_scale: float
) -> dict:
    with np.load(source, allow_pickle=True) as payload:
        if "csi" not in payload or "full_rsrp" not in payload:
            raise RuntimeError(
                f"{source} must contain real csi and full_rsrp; synthetic beam fallback is forbidden"
            )
        csi = as_complex_csi(payload["csi"])
        full_rsrp = np.asarray(payload["full_rsrp"], dtype=np.float32)
    if full_rsrp.ndim != 2 or full_rsrp.shape[1] < 256:
        raise ValueError(f"Expected at least 256 real beams in {source}, got {full_rsrp.shape}")
    n = min(csi.shape[0], full_rsrp.shape[0])
    csi = csi[:n]
    full_rsrp = full_rsrp[:n, :256]
    power = np.mean(np.abs(csi[:, 0, :k, :u]) ** 2, axis=(1, 2))
    keep = np.flatnonzero(np.isfinite(power) & (power > 0)).astype(np.int64)
    tokens = patch_tokens(csi[keep], k, u, channel_scale)
    labels = np.argmax(full_rsrp[keep], axis=1).astype(np.int64)
    if not np.isfinite(tokens).all() or not np.isfinite(full_rsrp[keep]).all():
        raise ValueError(f"Non-finite beam data in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "tokens": write_array(destination / f"{split}_tokens.npy", tokens),
        "labels": write_array(destination / f"{split}_labels.npy", labels),
        "full_rsrp": write_array(destination / f"{split}_full_rsrp.npy", full_rsrp[keep]),
        "source_row": write_array(destination / f"{split}_source_row.npy", keep),
        "input_shape": [int(k), int(u)],
        "channel_scale": float(channel_scale),
        "source_samples": int(n),
        "valid_samples": int(len(keep)),
        "excluded_zero_power": int(n - len(keep)),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--antennas", default=16, type=int)
    parser.add_argument("--train-subcarriers", default=64, type=int)
    parser.add_argument("--zero-shot-subcarriers", default=128, type=int)
    parser.add_argument("--channel-scale", default=1e6, type=float)
    args = parser.parse_args()
    scenario_root = args.source_root / args.scenario
    if not scenario_root.exists():
        raise FileNotFoundError(scenario_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "scenario": args.scenario,
        "target_beams": 256,
        "observed_set_b": "not used; CSI-only transfer",
        "train_protocol": "all K=64 train samples; validation selects checkpoint",
        "zero_shot_protocol": "K=128 test only; no target train/validation access",
        "channel_scale": float(args.channel_scale),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        source = scenario_root / f"{split}.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        manifest["splits"][f"K0064_{split}"] = prepare_split(
            source, args.output_root / "K0064", args.train_subcarriers, args.antennas, split,
            args.channel_scale,
        )
    manifest["splits"]["K0128_test"] = prepare_split(
        scenario_root / "test.npz", args.output_root / "K0128",
        args.zero_shot_subcarriers, args.antennas, "test", args.channel_scale,
    )
    manifest_path = args.output_root / "manifest.json"
    temporary = manifest_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    (args.output_root / "READY").touch()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
