#!/usr/bin/env python3
"""Create the one-time audited MaMIMO-UAV parent-window manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from datasets.mamimo_uav import OFFICIAL_MEASUREMENTS, MaMIMOUAVReconstructionDataset
from datasets.mamimo_uav_data_efficiency import SPLIT_NAMES, measurement_prefix


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("protocol") == "MaMIMO-UAV nonoverlapping T=16 valid-parent manifest":
            return 0

    files = []
    for spec in OFFICIAL_MEASUREMENTS:
        path = args.data_root / spec.name
        if not path.is_file() or path.stat().st_size != spec.bytes:
            raise ValueError(f"Missing or incomplete MaMIMO file: {path}")
        observed_md5 = md5_file(path)
        if observed_md5 != spec.md5:
            raise ValueError(f"MD5 mismatch for {spec.name}: {observed_md5} != {spec.md5}")
        files.append(
            {"name": spec.name, "bytes": spec.bytes, "expected_md5": spec.md5, "observed_md5": observed_md5}
        )

    dataset = MaMIMOUAVReconstructionDataset(
        args.data_root, measurements=OFFICIAL_MEASUREMENTS, stride=16
    )
    starts = {spec.name: [] for spec in OFFICIAL_MEASUREMENTS}
    for reference in dataset.references:
        starts[OFFICIAL_MEASUREMENTS[reference.measurement_index].name].append(reference.start)
    counts = {}
    for split, prefixes in SPLIT_NAMES.items():
        counts[split] = sum(
            len(starts[spec.name])
            for spec in OFFICIAL_MEASUREMENTS
            if measurement_prefix(spec) in prefixes
        )
    expected = {"train": 12183, "validation": 3125, "test": 3125}
    if counts != expected:
        raise ValueError(f"Unexpected MaMIMO split counts: {counts} != {expected}")
    payload = {
        "protocol": "MaMIMO-UAV nonoverlapping T=16 valid-parent manifest",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split_type": "measurement-recording-disjoint 4/1/1",
        "split_recordings": SPLIT_NAMES,
        "split_parent_counts": counts,
        "files": files,
        "quality_audit": dataset.quality_audit,
        "valid_starts_by_measurement": starts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
