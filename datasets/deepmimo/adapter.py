import json
import os
import time
from pathlib import Path

import h5py
import hdf5storage
import numpy as np

from util.data import patch_maker


CSI_KEYS = ("csi", "H", "channel", "channels", "H_train", "H_val", "H_test")
RSRP_KEYS = ("rsrp_set_b", "set_b_rsrp", "rsrp", "RSRP")
SET_B_KEYS = ("set_b", "beam_set_b", "candidate_beams")
LABEL_KEYS = ("label", "beam_label", "best_beam", "beam_index")
FULL_RSRP_KEYS = ("full_rsrp", "rsrp_full", "all_rsrp")
CACHE_VERSION = "v4"


def pad_csi_to_patch(csi: np.ndarray, patch_size: int = 4) -> np.ndarray:
    b, t, k, u = csi.shape
    pad_t = int(np.ceil(t / patch_size) * patch_size)
    pad_k = int(np.ceil(k / patch_size) * patch_size)
    pad_u = int(np.ceil(u / patch_size) * patch_size)
    if (pad_t, pad_k, pad_u) == (t, k, u):
        return csi
    padded = np.zeros((b, pad_t, pad_k, pad_u), dtype=csi.dtype)
    padded[:, :t, :k, :u] = csi
    return padded


def validate_beam_observations(set_b, set_a_size, set_b_size, source="beam data"):
    set_b = np.asarray(set_b, dtype=np.int64)
    if set_b.ndim != 2:
        raise ValueError(f"{source} set_b must be [N,set_b_size], got {set_b.shape}.")
    if set_b.shape[1] != int(set_b_size):
        raise ValueError(
            f"{source} set_b has width {set_b.shape[1]}, expected {int(set_b_size)}."
        )
    if set_b.size and (set_b.min() < 0 or set_b.max() >= int(set_a_size)):
        raise ValueError(f"{source} set_b contains beam indices outside [0,{int(set_a_size)}).")
    sorted_set_b = np.sort(set_b, axis=1)
    if sorted_set_b.shape[1] > 1 and np.any(np.diff(sorted_set_b, axis=1) == 0):
        raise ValueError(f"{source} set_b contains duplicate observed beams.")
    return set_b


def _patch_target_mask(original_shape, padded_shape, patch_size: int = 4) -> np.ndarray:
    _, t, k, u = original_shape
    _, pad_t, pad_k, pad_u = padded_shape
    mask = np.zeros((1, pad_t, pad_k, pad_u), dtype=np.float32)
    mask[:, :t, :k, :u] = 1.0
    return patch_maker(mask, patch_size=patch_size).reshape(-1, patch_size ** 3)


def _find_file(root: Path, scenario: str, split: str) -> Path:
    base = root / scenario if scenario else root
    if not base.exists():
        raise FileNotFoundError(f"DeepMIMO v4 path does not exist: {base}")
    candidates = []
    for suffix in (".npz", ".npy", ".mat", ".h5", ".hdf5"):
        candidates.extend(base.glob(f"*{split}*{suffix}"))
        candidates.extend(base.glob(f"{split}{suffix}"))
    if not candidates:
        for suffix in (".npz", ".npy", ".mat", ".h5", ".hdf5"):
            candidates.extend(base.rglob(f"*{split}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No DeepMIMO v4 split file found under {base} for split={split}. "
            "Expected an existing .npz/.npy/.mat/.h5 file containing CSI and optional beam fields."
        )
    return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]


def _first(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    lower = {str(k).lower(): k for k in mapping.keys()}
    for key in keys:
        lk = key.lower()
        if lk in lower:
            return mapping[lower[lk]]
    return None


def _load_mapping(path: Path) -> dict:
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr.item() if arr.shape == () else None, dict):
            return arr.item()
        return {"csi": arr}
    if path.suffix == ".mat":
        return hdf5storage.loadmat(str(path))
    if path.suffix in {".h5", ".hdf5"}:
        out = {}
        with h5py.File(path, "r") as handle:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    out[name.split("/")[-1]] = obj[()]
            handle.visititems(visit)
        return out
    raise ValueError(f"Unsupported DeepMIMO file type: {path}")


def _as_complex_csi(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        csi = arr
    elif arr.ndim >= 1 and arr.shape[-1] == 2:
        csi = arr[..., 0] + 1j * arr[..., 1]
    elif arr.ndim >= 1 and arr.shape[0] == 2:
        csi = arr[0] + 1j * arr[1]
    else:
        csi = arr.astype(np.float32) + 0j

    if csi.ndim == 3:
        csi = csi[:, None, :, :]
    if csi.ndim != 4:
        raise ValueError(
            f"DeepMIMO CSI must resolve to (B,T,K,U), got shape {csi.shape}."
        )
    return np.asarray(csi, dtype=np.complex64)


def _sample(data, n):
    if data is None:
        return None
    data = np.asarray(data)
    return data[:n]


def _metadata(args, scenario: str, split: str) -> dict:
    return {
        "source": "deepmimo_v4",
        "scenario": scenario,
        "split": split,
        "carrier_frequency_hz": float(getattr(args, "carrier_frequency_hz", 3.5e9)),
        "bandwidth_mhz": float(getattr(args, "bandwidth_mhz", 100.0)),
        "subcarrier_spacing_khz": float(getattr(args, "subcarrier_spacing_khz", 30.0)),
        "antenna_config": str(getattr(args, "antenna_config", "")),
        "rank": int(getattr(args, "rank", 1)),
        "csi_payload_bits": int(getattr(args, "csi_payload_bits", 256)),
    }


def deepmimo_cache_path(args, task: str, split: str) -> Path:
    cache_dir = getattr(args, "deepmimo_cache_dir", "") or os.path.join(
        getattr(args, "deepmimo_root", ""), ".adaptive_3d_rope_cache"
    )
    scenario = getattr(args, "deepmimo_scenario", "") or "scenario"
    suffix = ""
    if task == "beam_management":
        suffix = f"_a{int(getattr(args, 'set_a_size', 64))}_b{int(getattr(args, 'set_b_size', 16))}"
    return Path(cache_dir) / f"{scenario}_{task}_{split}{suffix}_{CACHE_VERSION}.npz"


def load_or_build_deepmimo_cache(args, task: str, split: str) -> Path:
    cache_path = deepmimo_cache_path(args, task, split)
    if cache_path.exists():
        return cache_path
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_fd = None
    while lock_fd is None:
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            if cache_path.exists():
                return cache_path
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > 6 * 3600:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            time.sleep(1.0)

    try:
        if cache_path.exists():
            return cache_path

        root = Path(getattr(args, "deepmimo_root", ""))
        scenario = getattr(args, "deepmimo_scenario", "")
        source_file = _find_file(root, scenario, split)
        mapping = _load_mapping(source_file)

        csi = _first(mapping, CSI_KEYS)
        if csi is None:
            raise KeyError(
                f"{source_file} does not contain CSI. Tried keys: {', '.join(CSI_KEYS)}"
            )
        csi = _as_complex_csi(csi)
        n = csi.shape[0]

        original_dims = np.asarray(csi.shape[1:], dtype=np.int64)
        power = np.mean(np.abs(csi) ** 2, axis=(1, 2, 3), keepdims=True).clip(1e-12)
        csi = csi / np.sqrt(power)
        original_shape = csi.shape
        csi = pad_csi_to_patch(csi, patch_size=4)
        tokens = patch_maker(csi, patch_size=4).astype(np.complex64)
        padded_dims = np.asarray(csi.shape[1:], dtype=np.int64)
        target_mask = _patch_target_mask(original_shape, csi.shape, patch_size=4)
        target_mask = np.repeat(target_mask[None, :, :], n, axis=0)
        valid_tokens = target_mask.any(axis=-1)
        lengths = np.full((n,), tokens.shape[1], dtype=np.int64)

        meta = _metadata(args, scenario or source_file.parent.name, split)
        phys_meta = np.asarray(
            [
                meta["carrier_frequency_hz"],
                meta["subcarrier_spacing_khz"] * 1e3,
                1e-3,
                0.5,
            ],
            dtype=np.float32,
        )
        phys_meta = np.repeat(phys_meta[None, :], n, axis=0)

        arrays = {
            "x": tokens,
            "lengths": lengths,
            "dims": np.repeat(padded_dims[None, :], n, axis=0),
            "orig_dims": np.repeat(original_dims[None, :], n, axis=0),
            "target_mask": target_mask.astype(np.float32),
            "token_mask": valid_tokens.astype(np.float32),
            "meta": phys_meta,
            "meta_json": np.asarray(json.dumps(meta)),
        }

        if task == "beam_management":
            rsrp = _sample(_first(mapping, RSRP_KEYS), n)
            set_b = _sample(_first(mapping, SET_B_KEYS), n)
            label = _sample(_first(mapping, LABEL_KEYS), n)
            full_rsrp = _sample(_first(mapping, FULL_RSRP_KEYS), n)
            if rsrp is None or set_b is None or label is None:
                raise KeyError(
                    f"{source_file} is missing beam fields. Required: rsrp_set_b/set_b/label; "
                    "full_rsrp is optional but recommended for 38.843 RSRP metrics."
                )
            rsrp = np.asarray(rsrp, dtype=np.float32)
            set_b = np.asarray(set_b, dtype=np.int64)
            label = np.asarray(label, dtype=np.int64).reshape(-1)[:n]
            if full_rsrp is None:
                set_a = int(getattr(args, "set_a_size", max(int(set_b.max()) + 1, 64)))
                full_rsrp = np.full((n, set_a), float(np.min(rsrp)) - 5.0, dtype=np.float32)
                for i in range(n):
                    full_rsrp[i, set_b[i]] = rsrp[i]
            else:
                full_rsrp = np.asarray(full_rsrp, dtype=np.float32)[:n]
            set_a = int(getattr(args, "set_a_size", full_rsrp.shape[1]))
            set_b_size = int(getattr(args, "set_b_size", set_b.shape[1]))
            if full_rsrp.shape[1] < set_a:
                raise ValueError(
                    f"Requested set_a_size={set_a}, but {source_file} only contains "
                    f"{full_rsrp.shape[1]} real beams in full_rsrp. Regenerate the "
                    "DeepMIMO export with a larger --set_a_size instead of padding "
                    "dummy beams."
                )
            full_rsrp = full_rsrp[:, :set_a]
            label = np.argmax(full_rsrp, axis=1).astype(np.int64)
            if set_b_size > set_a:
                raise ValueError(f"set_b_size={set_b_size} must be <= set_a_size={set_a}.")
            if set_b.shape[1] != set_b_size:
                if set_b.shape[1] < set_b_size:
                    raise ValueError(
                        f"Requested set_b_size={set_b_size}, but {source_file} only has "
                        f"{set_b.shape[1]} observed beams."
                    )
                set_b = set_b[:, :set_b_size]
            set_b = validate_beam_observations(set_b, set_a, set_b_size, str(source_file))
            rsrp = np.take_along_axis(full_rsrp, set_b, axis=1).astype(np.float32)
            arrays.update(
                {
                    "rsrp_set_b": rsrp,
                    "set_b": set_b,
                    "label": label,
                    "full_rsrp": np.asarray(full_rsrp, dtype=np.float32),
                }
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        np.savez_compressed(tmp_path, **arrays)
        tmp_npz = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(tmp_path.suffix + ".npz")
        os.replace(tmp_npz, cache_path)
        return cache_path
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
