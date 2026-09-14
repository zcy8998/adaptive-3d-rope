"""Lightweight deterministic identifiers for model-independent evaluation masks."""

import hashlib


def stable_eval_mask_seed(base_seed, dataset_name, mask_name):
    payload = f"{int(base_seed)}:{dataset_name}:{mask_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**31 - 1)
