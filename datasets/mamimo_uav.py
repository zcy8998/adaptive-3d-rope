"""MaMIMO-UAV adapter for trajectory-preserving masked reconstruction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_SHAPE = (16, 100, 8)
PATCH_SIZE = 4
TOKEN_GRID = tuple(value // PATCH_SIZE for value in INPUT_SHAPE)
PATCH_COUNT = math.prod(TOKEN_GRID)
MASK_RATIO = 0.85
MASK_COUNT = round(PATCH_COUNT * MASK_RATIO)
ARRAY_ROWS = 8
SAMPLE_INTERVAL_MS = 1.0
STORAGE_SUBCARRIER_ORDER = np.concatenate(
    (np.arange(0, 100, 2), np.arange(1, 100, 2))
)
NATURAL_SUBCARRIER_INDEX = np.argsort(STORAGE_SUBCARRIER_ORDER)


@dataclass(frozen=True)
class MeasurementSpec:
    name: str
    datafile_id: int
    frames: int
    bytes: int
    md5: str


OFFICIAL_MEASUREMENTS = (
    MeasurementSpec(
        "20230316_154153_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119641,
        50000,
        1280000000,
        "af4560594152e307daa80bbea2cb4f93",
    ),
    MeasurementSpec(
        "20230316_154830_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119649,
        50000,
        1280000000,
        "dd7702f7535aaccaccc495f7f0531c87",
    ),
    MeasurementSpec(
        "20230316_160527_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119638,
        50000,
        1280000000,
        "ce02f19d218b5797297273dde49d0c53",
    ),
    MeasurementSpec(
        "20230316_160955_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119644,
        50000,
        1280000000,
        "c5178986bf9295dec518fc81b38098ed",
    ),
    MeasurementSpec(
        "20230316_161206_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119651,
        50000,
        1280000000,
        "cf39877499b4e75b0b912f7b53f9e456",
    ),
    MeasurementSpec(
        "20230316_161434_n64_l1_s50000_sr2_f2.610G_g30.0.bin",
        119637,
        50000,
        1280000000,
        "1eac768d53a5d848274ffa05806d8269",
    ),
)


@dataclass(frozen=True)
class WindowReference:
    measurement_index: int
    start: int


def stable_uint64(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def decode_binary_window(raw: np.ndarray) -> np.ndarray:
    """Decode official big-endian [imag, real] samples into [T, K, U]."""

    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[1:] != (100, 64, 2):
        raise ValueError(f"Expected raw MaMIMO window [T,100,64,2], got {raw.shape}")
    storage_order = (
        raw[..., 1].astype(np.float32) + 1j * raw[..., 0].astype(np.float32)
    ) * (2.0**-8)
    return np.asarray(
        storage_order[:, NATURAL_SUBCARRIER_INDEX, :], dtype=np.complex64
    )


def patch_coordinates() -> np.ndarray:
    return np.stack(
        np.meshgrid(*(np.arange(value) for value in TOKEN_GRID), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)


def patchify_complex(crop: np.ndarray) -> np.ndarray:
    crop = np.asarray(crop)
    if crop.shape != INPUT_SHAPE:
        raise ValueError(f"Expected MaMIMO crop {INPUT_SHAPE}, got {crop.shape}")
    t, k, u = crop.shape
    patches = crop.reshape(
        t // PATCH_SIZE,
        PATCH_SIZE,
        k // PATCH_SIZE,
        PATCH_SIZE,
        u // PATCH_SIZE,
        PATCH_SIZE,
    ).transpose(0, 2, 4, 1, 3, 5)
    return patches.reshape(PATCH_COUNT, PATCH_SIZE**3).astype(
        np.complex64, copy=False
    )


def deterministic_mask(seed: int, parent_id: str, row_id: int) -> np.ndarray:
    rng = np.random.default_rng(
        stable_uint64("MaMIMO-UAV-random-mask", seed, parent_id, row_id)
    )
    selected = rng.choice(PATCH_COUNT, MASK_COUNT, replace=False)
    mask = np.zeros(PATCH_COUNT, dtype=np.bool_)
    mask[selected] = True
    return mask


class MaMIMOUAVReconstructionDataset(Dataset):
    """Non-overlapping continuous windows with each physical URA row as a crop."""

    def __init__(
        self,
        data_root: Path,
        *,
        seed: int = 42,
        stride: int = 16,
        max_windows: int | None = None,
        measurements: tuple[MeasurementSpec, ...] = OFFICIAL_MEASUREMENTS,
        valid_starts_by_measurement: dict[str, list[int]] | None = None,
    ):
        self.data_root = Path(data_root)
        self.seed = int(seed)
        self.stride = int(stride)
        self.measurements = tuple(measurements)
        self.quality_audit = []
        if self.stride < INPUT_SHAPE[0]:
            raise ValueError("MaMIMO evaluation windows must not overlap")
        references = []
        for measurement_index, spec in enumerate(self.measurements):
            path = self.data_root / spec.name
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != spec.bytes:
                raise ValueError(
                    f"MaMIMO file size mismatch for {spec.name}: "
                    f"{path.stat().st_size} != {spec.bytes}"
                )
            candidate_starts = np.arange(
                0, spec.frames - INPUT_SHAPE[0] + 1, self.stride, dtype=np.int64
            )
            if valid_starts_by_measurement is not None:
                if spec.name not in valid_starts_by_measurement:
                    raise ValueError(f"Missing audited starts for {spec.name}")
                starts = np.asarray(valid_starts_by_measurement[spec.name], dtype=np.int64)
                if (
                    len(np.unique(starts)) != len(starts)
                    or np.any(starts < 0)
                    or np.any(starts + INPUT_SHAPE[0] > spec.frames)
                    or np.any(starts % self.stride != 0)
                ):
                    raise ValueError(f"Invalid audited starts for {spec.name}")
                starts.sort()
                excluded = np.setdiff1d(candidate_starts, starts, assume_unique=True)
            else:
                mapped = np.memmap(
                    path,
                    dtype=">i2",
                    mode="r",
                    shape=(spec.frames, 100, 64, 2),
                )
                valid_parts = []
                for offset in range(0, len(candidate_starts), 128):
                    selected = candidate_starts[offset : offset + 128]
                    block = np.asarray(
                        mapped[selected[0] : selected[-1] + INPUT_SHAPE[0]]
                    ).reshape(len(selected), INPUT_SHAPE[0], 100, 8, 8, 2)
                    row_valid = np.any(block != 0, axis=(1, 2, 4, 5))
                    valid_parts.append(np.all(row_valid, axis=1))
                valid = np.concatenate(valid_parts)
                starts = candidate_starts[valid]
                excluded = candidate_starts[~valid]
                del mapped
            self.quality_audit.append(
                {
                    "measurement": spec.name,
                    "candidate_parent_windows": int(len(candidate_starts)),
                    "included_parent_windows": int(len(starts)),
                    "excluded_all_zero_parent_windows": int(len(excluded)),
                    "excluded_starts": excluded.tolist(),
                    "starts_source": (
                        "audited_manifest"
                        if valid_starts_by_measurement is not None
                        else "binary_scan"
                    ),
                }
            )
            references.extend(
                WindowReference(measurement_index, int(start))
                for start in starts
            )
        if max_windows is not None and len(references) > int(max_windows):
            references = sorted(
                references,
                key=lambda item: stable_uint64(
                    "MaMIMO-UAV-cap",
                    self.seed,
                    self.measurements[item.measurement_index].name,
                    item.start,
                ),
            )[: int(max_windows)]
            references.sort(key=lambda item: (item.measurement_index, item.start))
        self.references = references
        self._maps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.references)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def _map(self, measurement_index: int) -> np.memmap:
        mapped = self._maps.get(measurement_index)
        if mapped is None:
            spec = self.measurements[measurement_index]
            mapped = np.memmap(
                self.data_root / spec.name,
                dtype=">i2",
                mode="r",
                shape=(spec.frames, 100, 64, 2),
            )
            self._maps[measurement_index] = mapped
        return mapped

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        reference = self.references[index]
        spec = self.measurements[reference.measurement_index]
        stop = reference.start + INPUT_SHAPE[0]
        channel = decode_binary_window(
            self._map(reference.measurement_index)[reference.start : stop]
        )
        # The official parser restores logical antenna indices 0..63. The
        # published 8x8 URA is represented in row-major logical order here.
        ura = channel.reshape(INPUT_SHAPE[0], 100, 8, 8)
        parent_id = f"{spec.name}:{reference.start}"
        tokens, masks, normalization = [], [], []
        for row_id in range(ARRAY_ROWS):
            raw_tokens = patchify_complex(ura[:, :, row_id, :])
            mask = deterministic_mask(self.seed, parent_id, row_id)
            rms = float(np.sqrt(np.mean(np.abs(raw_tokens[~mask]) ** 2)))
            if not np.isfinite(rms) or rms <= 0:
                raise ValueError(f"Invalid visible-only RMS for {parent_id}/row-{row_id}")
            tokens.append(raw_tokens / rms)
            masks.append(mask)
            normalization.append(rms)
        return {
            "tokens": torch.from_numpy(np.stack(tokens)),
            "mask": torch.from_numpy(np.stack(masks)),
            "coords": torch.from_numpy(patch_coordinates()).long(),
            "parent_id": parent_id,
            "measurement_id": spec.name,
            "start": torch.tensor(reference.start, dtype=torch.long),
            "row_id": torch.arange(ARRAY_ROWS, dtype=torch.long),
            "normalization_rms": torch.tensor(normalization, dtype=torch.float32),
        }
