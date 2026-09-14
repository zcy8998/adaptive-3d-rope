import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import hdf5storage
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data

import util.misc as misc


SPEED_OF_LIGHT = 299792458.0


def _to_float_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    values = np.asarray(value)
    return float(values.reshape(-1)[0]) if values.size else float(default)


def load_dataset_phys_meta(config_path: str) -> np.ndarray:
    config = hdf5storage.loadmat(config_path)
    carrier = _to_float_scalar(config.get("fc"), default=1.0)
    spacing = 0.5 * SPEED_OF_LIGHT / max(carrier, 1.0)
    return np.asarray(
        [
            carrier,
            _to_float_scalar(config.get("delta_f"), default=1.0),
            _to_float_scalar(config.get("delta_t"), default=1.0),
            spacing,
        ],
        dtype=np.float32,
    )


def generate_gaussian_noise(values: np.ndarray, snr_db: float) -> np.ndarray:
    axes = tuple(range(1, values.ndim))
    signal_power = np.mean(np.abs(values) ** 2, axis=axes, keepdims=True)
    snr_linear = 10 ** (float(snr_db) / 10)
    noise_power = signal_power / snr_linear
    return (
        np.random.standard_normal(values.shape) * np.sqrt(noise_power / 2)
        + 1j * np.random.standard_normal(values.shape) * np.sqrt(noise_power / 2)
    )


def patch_maker(values: np.ndarray, patch_size: int = 4) -> np.ndarray:
    batch, temporal, frequency, antenna = values.shape
    if any(length % patch_size for length in (temporal, frequency, antenna)):
        raise ValueError("CSI dimensions must be divisible by the patch size")
    reshaped = values.reshape(
        batch,
        temporal // patch_size,
        patch_size,
        frequency // patch_size,
        patch_size,
        antenna // patch_size,
        patch_size,
    )
    return reshaped.transpose(0, 1, 3, 5, 2, 4, 6).reshape(
        batch, -1, patch_size**3
    )


class CSIDataset(data.Dataset):
    """In-memory CSI patches for the core train and fine-tune paths."""

    def __init__(
        self,
        dataset,
        world_size: int = 1,
        rank: int = 0,
        dataset_type: str = "train",
        SNR: float | None = 20,
        patch_size: int = 4,
        data_num: float | int | None = None,
        max_workers: int = 4,
        data_dir: str | None = None,
        return_phys_meta: bool = False,
    ):
        super().__init__()
        if not data_dir:
            raise ValueError("data_dir is required")
        self.rank = rank
        self.dataset_type = dataset_type
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.snr = SNR
        self.return_phys_meta = return_phys_meta
        self.max_workers = max_workers
        self.datasets_list = dataset.split(",") if isinstance(dataset, str) else list(dataset)
        self.dataset_bounds = []
        self.dataset_arrays = {}
        self.dataset_phys_meta = {}
        self._read_metadata(data_num)
        self._load_all()
        if dist.is_available() and dist.is_initialized():
            misc.synchronize()

    @staticmethod
    def _sample_count(total: int, data_num: float | int | None) -> int:
        if data_num is None:
            return total
        if isinstance(data_num, float) and 0 < data_num <= 1:
            return max(1, int(round(total * data_num)))
        return max(1, min(total, int(data_num)))

    def _read_metadata(self, data_num) -> None:
        start = 0
        for name in self.datasets_list:
            path = os.path.join(self.data_dir, name, f"{self.dataset_type}_data.mat")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Data file not found: {path}")
            with h5py.File(path, "r") as handle:
                antenna, frequency, temporal, total = handle[f"H_{self.dataset_type}"].shape
            samples = self._sample_count(total, data_num)
            dims = (temporal, frequency, antenna)
            if any(length % self.patch_size for length in dims):
                raise ValueError(f"{name} dimensions {dims} are not divisible by {self.patch_size}")
            config_path = os.path.join(self.data_dir, name, "config.mat")
            self.dataset_bounds.append(
                {
                    "name": name,
                    "path": path,
                    "start": start,
                    "end": start + samples,
                    "samples": samples,
                    "dims": dims,
                    "token_length": int(np.prod(dims) // self.patch_size**3),
                    "phys_meta": (
                        load_dataset_phys_meta(config_path)
                        if self.return_phys_meta and os.path.isfile(config_path)
                        else None
                    ),
                }
            )
            start += samples
        self.total_samples = start

    def _load_all(self) -> None:
        with ThreadPoolExecutor(max_workers=min(len(self.dataset_bounds), self.max_workers)) as executor:
            futures = [executor.submit(self._load_one, record) for record in self.dataset_bounds]
            for future in as_completed(futures):
                future.result()

    def _load_one(self, record: dict) -> None:
        values = hdf5storage.loadmat(record["path"])[f"H_{self.dataset_type}"]
        values = values[: record["samples"]]
        power = np.maximum(np.mean(np.abs(values) ** 2, axis=(1, 2, 3), keepdims=True), 1e-12)
        values = values / np.sqrt(power)
        if self.snr is not None:
            values = values + generate_gaussian_noise(values, self.snr)
        patches = patch_maker(values, self.patch_size)
        expected = (record["samples"], record["token_length"], self.patch_size**3)
        if patches.shape != expected:
            raise ValueError(f"{record['name']} patch shape {patches.shape} does not match {expected}")
        self.dataset_arrays[record["name"]] = np.asarray(
            patches, dtype=np.complex64 if np.iscomplexobj(patches) else np.float32
        )
        if record["phys_meta"] is not None:
            self.dataset_phys_meta[record["name"]] = record["phys_meta"]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        for record in self.dataset_bounds:
            if record["start"] <= index < record["end"]:
                local_index = index - record["start"]
                sample = torch.from_numpy(self.dataset_arrays[record["name"]][local_index].copy())
                if self.return_phys_meta:
                    metadata = self.dataset_phys_meta.get(record["name"], np.zeros(4, dtype=np.float32))
                    return sample, record["token_length"], record["dims"], torch.from_numpy(metadata.copy())
                return sample, record["token_length"], record["dims"]
        raise IndexError(f"Sample index {index} is out of range")


class _SingleCSIDataset(data.Dataset):
    def __init__(self, values, token_length: int, dims, dataset_name: str, phys_meta=None):
        self.values = values
        self.token_length = token_length
        self.dims = dims
        self.dataset_name = dataset_name
        self.phys_meta = None if phys_meta is None else np.asarray(phys_meta, dtype=np.float32)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, index: int):
        if self.phys_meta is None:
            return self.values[index], self.token_length, self.dims
        return self.values[index], self.token_length, self.dims, torch.from_numpy(self.phys_meta.copy())

    def get_dataset_name(self) -> str:
        return self.dataset_name


def _evaluation_loader(args, dataset_name: str, dataset_type: str):
    path = os.path.join(args.data_dir, dataset_name, f"{dataset_type}_data.mat")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    values = hdf5storage.loadmat(path)[f"H_{dataset_type}"]
    power = np.maximum(np.mean(np.abs(values) ** 2, axis=(1, 2, 3), keepdims=True), 1e-12)
    values = values / np.sqrt(power)
    if args.snr_db is not None:
        values = values + generate_gaussian_noise(values, args.snr_db)
    batch, temporal, frequency, antenna = values.shape
    patches = patch_maker(values, 4)
    config_path = os.path.join(args.data_dir, dataset_name, "config.mat")
    phys_meta = load_dataset_phys_meta(config_path) if args.use_phys_coord and os.path.isfile(config_path) else None
    dataset = _SingleCSIDataset(
        patches,
        temporal * frequency * antenna // 4**3,
        (temporal, frequency, antenna),
        dataset_name,
        phys_meta,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )


def data_load_main(args, dataset_type: str = "val", test_type: str = "normal"):
    if test_type != "normal":
        raise ValueError("The core release supports only the standard CSI data layout")
    return [_evaluation_loader(args, name, dataset_type) for name in args.dataset.split(",")]
