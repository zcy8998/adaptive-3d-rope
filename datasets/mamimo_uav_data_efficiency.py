"""Flight-session-disjoint MaMIMO-UAV data-efficiency dataset."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.mamimo_uav import (
    ARRAY_ROWS,
    INPUT_SHAPE,
    MASK_COUNT,
    OFFICIAL_MEASUREMENTS,
    PATCH_COUNT,
    MaMIMOUAVReconstructionDataset,
    MeasurementSpec,
    decode_binary_window,
    patch_coordinates,
    patchify_complex,
    stable_uint64,
)


SPLIT_NAMES = {
    "train": (
        "20230316_154153",
        "20230316_154830",
        "20230316_160527",
        "20230316_160955",
    ),
    "validation": ("20230316_161206",),
    "test": ("20230316_161434",),
}


def measurement_prefix(spec: MeasurementSpec) -> str:
    return "_".join(spec.name.split("_")[:2])


def split_measurements(
    split: str,
    measurements: tuple[MeasurementSpec, ...] = OFFICIAL_MEASUREMENTS,
) -> tuple[MeasurementSpec, ...]:
    split = "validation" if split in {"val", "validation"} else split
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown MaMIMO split: {split}")
    prefixes = SPLIT_NAMES[split]
    selected = tuple(
        spec for spec in measurements if any(spec.name.startswith(prefix) for prefix in prefixes)
    )
    if len(selected) != len(prefixes):
        raise ValueError(
            f"Expected {len(prefixes)} measurements for {split}, found {len(selected)}"
        )
    return selected


def epoch_mask(
    seed: int,
    epoch: int,
    parent_id: str,
    row_id: int,
    *,
    training: bool,
) -> np.ndarray:
    mask_epoch = int(epoch) if training else 0
    rng = np.random.default_rng(
        stable_uint64("MaMIMO-UAV-data-efficiency-mask", seed, mask_epoch, parent_id, row_id)
    )
    selected = rng.choice(PATCH_COUNT, MASK_COUNT, replace=False)
    mask = np.zeros(PATCH_COUNT, dtype=np.bool_)
    mask[selected] = True
    return mask


class MaMIMOUAVDataEfficiencyDataset(Dataset):
    """One parent per item with eight rows and nested train fractions."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        *,
        fraction: float = 1.0,
        seed: int = 42,
        measurements: tuple[MeasurementSpec, ...] = OFFICIAL_MEASUREMENTS,
        max_parents: int | None = None,
        reference_manifest: Path | None = None,
    ):
        self.split = "validation" if split in {"val", "validation"} else split
        self.training = self.split == "train"
        if not 0 < fraction <= 1:
            raise ValueError("fraction must lie in (0, 1]")
        selected_measurements = split_measurements(self.split, measurements)
        valid_starts = None
        if reference_manifest is not None:
            payload = json.loads(Path(reference_manifest).read_text())
            if payload.get("protocol") != "MaMIMO-UAV nonoverlapping T=16 valid-parent manifest":
                raise ValueError(f"Unexpected MaMIMO reference manifest: {reference_manifest}")
            valid_starts = payload.get("valid_starts_by_measurement")
            if not isinstance(valid_starts, dict):
                raise ValueError("MaMIMO reference manifest lacks valid starts")
        self.base = MaMIMOUAVReconstructionDataset(
            data_root,
            seed=seed,
            stride=INPUT_SHAPE[0],
            measurements=selected_measurements,
            valid_starts_by_measurement=valid_starts,
        )
        references = list(self.base.references)
        if self.training and fraction < 1:
            nested = []
            for measurement_index in range(len(selected_measurements)):
                candidates = [
                    reference
                    for reference in references
                    if reference.measurement_index == measurement_index
                ]
                candidates.sort(
                    key=lambda reference: stable_uint64(
                        "MaMIMO-UAV-fraction",
                        seed,
                        selected_measurements[measurement_index].name,
                        reference.start,
                    )
                )
                count = max(1, math.floor(len(candidates) * fraction))
                nested.extend(candidates[:count])
            references = sorted(
                nested, key=lambda reference: (reference.measurement_index, reference.start)
            )
        if max_parents is not None:
            references = references[: int(max_parents)]
        self.references = references
        self.measurements = selected_measurements
        self.fraction = float(fraction)
        self.seed = int(seed)
        self.epoch = 0

    @property
    def parent_ids(self) -> list[str]:
        return [
            f"{self.measurements[reference.measurement_index].name}:{reference.start}"
            for reference in self.references
        ]

    @property
    def quality_audit(self) -> list[dict]:
        return self.base.quality_audit

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        reference = self.references[index]
        spec = self.measurements[reference.measurement_index]
        stop = reference.start + INPUT_SHAPE[0]
        channel = decode_binary_window(
            self.base._map(reference.measurement_index)[reference.start : stop]
        )
        array = channel.reshape(INPUT_SHAPE[0], 100, 8, 8)
        parent_id = f"{spec.name}:{reference.start}"
        tokens, masks, normalization = [], [], []
        for row_id in range(ARRAY_ROWS):
            raw_tokens = patchify_complex(array[:, :, row_id, :])
            mask = epoch_mask(
                self.seed,
                self.epoch,
                parent_id,
                row_id,
                training=self.training,
            )
            rms = float(np.sqrt(np.mean(np.abs(raw_tokens[~mask]) ** 2)))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError(f"Invalid visible RMS for {parent_id}/row-{row_id}")
            tokens.append(raw_tokens / rms)
            masks.append(mask)
            normalization.append(rms)
        return {
            "tokens": torch.from_numpy(np.stack(tokens)),
            "mask": torch.from_numpy(np.stack(masks)),
            "coords": torch.from_numpy(patch_coordinates()).long(),
            "parent_id": parent_id,
            "measurement_id": spec.name,
            "recording_id": measurement_prefix(spec),
            "start": reference.start,
            "row_id": torch.arange(ARRAY_ROWS),
            "normalization_rms": torch.tensor(normalization, dtype=torch.float32),
        }
