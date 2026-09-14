import json

import hdf5storage
import numpy as np
import torch
from torch.utils.data import Dataset

from util.data import generate_gaussian_noise, load_dataset_phys_meta, patch_maker
from datasets.deepmimo.adapter import load_or_build_deepmimo_cache, validate_beam_observations


class TaskCSIDataset(Dataset):
    def __init__(self, args, split="train", task="csi_feedback"):
        self.args = args
        self.split = split
        self.task = task
        if getattr(args, "data_source", "mat") == "deepmimo_v4":
            self._load_deepmimo()
        else:
            self._load_mat()

    def _load_deepmimo(self):
        path = load_or_build_deepmimo_cache(self.args, self.task, self.split)
        data = np.load(path, allow_pickle=True)
        limit = None
        if self.split == "train" and 0 < float(getattr(self.args, "data_num", 1.0)) < 1.0:
            limit = max(1, int(data["x"].shape[0] * float(self.args.data_num)))
        sl = slice(None, limit)
        self.x = np.asarray(data["x"][sl], dtype=np.complex64)
        self.lengths = np.asarray(data["lengths"][sl], dtype=np.int64)
        self.dims = np.asarray(data["dims"][sl], dtype=np.int64)
        self.orig_dims = np.asarray(data["orig_dims"][sl], dtype=np.int64) if "orig_dims" in data else self.dims.copy()
        self.target_mask = (
            np.asarray(data["target_mask"][sl], dtype=np.float32)
            if "target_mask" in data
            else np.ones(self.x.shape, dtype=np.float32)
        )
        self.token_mask = (
            np.asarray(data["token_mask"][sl], dtype=np.float32)
            if "token_mask" in data
            else np.ones(self.x.shape[:2], dtype=np.float32)
        )
        self.meta = np.asarray(data["meta"][sl], dtype=np.float32)
        self.meta_json = str(data["meta_json"].item()) if "meta_json" in data else "{}"
        if self.task == "beam_management":
            self.rsrp_set_b = np.asarray(data["rsrp_set_b"][sl], dtype=np.float32)
            self.set_b = np.asarray(data["set_b"][sl], dtype=np.int64)
            self.label = np.asarray(data["label"][sl], dtype=np.int64)
            self.full_rsrp = np.asarray(data["full_rsrp"][sl], dtype=np.float32)
            self._validate_beam_arrays()

    def _load_mat(self):
        dataset_name = str(self.args.dataset).split(",")[0]
        path = f"{self.args.data_dir}/{dataset_name}/{self.split}_data.mat"
        loaded = hdf5storage.loadmat(path)
        key = f"H_{self.split}"
        if key not in loaded:
            raise KeyError(f"{path} does not contain {key}")
        csi = loaded[key]
        if 0 < float(self.args.data_num) < 1.0:
            keep = max(1, int(csi.shape[0] * float(self.args.data_num)))
            csi = csi[:keep]
        power = np.mean(np.abs(csi) ** 2, axis=(1, 2, 3), keepdims=True).clip(1e-12)
        csi = csi / np.sqrt(power)
        csi = csi + generate_gaussian_noise(csi, 20)
        self.x = patch_maker(csi, patch_size=4).astype(np.complex64)
        n = self.x.shape[0]
        self.lengths = np.full((n,), self.x.shape[1], dtype=np.int64)
        self.dims = np.repeat(np.asarray(csi.shape[1:], dtype=np.int64)[None, :], n, axis=0)
        self.orig_dims = self.dims.copy()
        self.target_mask = np.ones(self.x.shape, dtype=np.float32)
        self.token_mask = np.ones(self.x.shape[:2], dtype=np.float32)
        config_path = f"{self.args.data_dir}/{dataset_name}/config.mat"
        phys = load_dataset_phys_meta(config_path)
        self.meta = np.repeat(phys[None, :], n, axis=0).astype(np.float32)
        self.meta_json = json.dumps({"source": "mat", "scenario": dataset_name, "split": self.split})
        if self.task == "beam_management":
            self._build_synthetic_beam(csi)

    def _build_synthetic_beam(self, csi):
        n = csi.shape[0]
        set_a = int(getattr(self.args, "set_a_size", 64))
        set_b = int(getattr(self.args, "set_b_size", 16))
        rng = np.random.default_rng(int(getattr(self.args, "seed", 0)) + hash(self.split) % 1000)
        full = rng.normal(-85.0, 6.0, size=(n, set_a)).astype(np.float32)
        energy = np.mean(np.abs(csi) ** 2, axis=(1, 2, 3)).astype(np.float32)
        full[:, : min(set_a, 8)] += energy[:, None] * 3.0
        self.label = np.argmax(full, axis=1).astype(np.int64)
        self.set_b = np.zeros((n, set_b), dtype=np.int64)
        self.rsrp_set_b = np.zeros((n, set_b), dtype=np.float32)
        fixed_candidates = np.linspace(0, set_a - 1, set_b).round().astype(np.int64)
        for i in range(n):
            self.set_b[i] = fixed_candidates
            self.rsrp_set_b[i] = full[i, fixed_candidates]
        self.full_rsrp = full
        self._validate_beam_arrays()

    def _validate_beam_arrays(self):
        set_a = int(getattr(self.args, "set_a_size", self.full_rsrp.shape[1]))
        set_b = int(getattr(self.args, "set_b_size", self.set_b.shape[1]))
        self.set_b = validate_beam_observations(
            self.set_b,
            set_a,
            set_b,
            f"{self.task}:{self.split}",
        )
        if self.full_rsrp.shape[1] < set_a:
            raise ValueError(
                f"{self.task}:{self.split} full_rsrp has {self.full_rsrp.shape[1]} beams, "
                f"expected at least {set_a}."
            )
        self.full_rsrp = self.full_rsrp[:, :set_a]
        self.label = np.argmax(self.full_rsrp, axis=1).astype(np.int64)
        self.rsrp_set_b = np.take_along_axis(self.full_rsrp, self.set_b, axis=1).astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {
            "x": torch.from_numpy(self.x[idx].copy()),
            "lengths": torch.tensor(int(self.lengths[idx]), dtype=torch.long),
            "dims": torch.from_numpy(self.dims[idx].copy()).long(),
            "orig_dims": torch.from_numpy(self.orig_dims[idx].copy()).long(),
            "target_mask": torch.from_numpy(self.target_mask[idx].copy()).float(),
            "token_mask": torch.from_numpy(self.token_mask[idx].copy()).float(),
            "meta": torch.from_numpy(self.meta[idx].copy()).float(),
        }
        if self.task == "beam_management":
            item.update(
                {
                    "rsrp_set_b": torch.from_numpy(self.rsrp_set_b[idx].copy()).float(),
                    "set_b": torch.from_numpy(self.set_b[idx].copy()).long(),
                    "label": torch.tensor(int(self.label[idx]), dtype=torch.long),
                    "full_rsrp": torch.from_numpy(self.full_rsrp[idx].copy()).float(),
                }
            )
        return item

    @staticmethod
    def collate(batch):
        max_len = max(int(item["lengths"]) for item in batch)
        feat_dim = batch[0]["x"].shape[-1]
        out = {}
        x = torch.zeros(len(batch), max_len, feat_dim, dtype=batch[0]["x"].dtype)
        for i, item in enumerate(batch):
            length = int(item["lengths"])
            x[i, :length] = item["x"][:length]
        out["x"] = x
        for key in ("lengths", "dims", "orig_dims", "meta", "target_mask", "token_mask"):
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        for key in ("rsrp_set_b", "set_b", "label", "full_rsrp"):
            if key in batch[0]:
                out[key] = torch.stack([item[key] for item in batch], dim=0)
        return out
