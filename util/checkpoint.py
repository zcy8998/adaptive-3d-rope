"""Checkpoint loading compatible with both legacy and current PyTorch."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch


def load_checkpoint(path, map_location="cpu"):
    serialization = torch.serialization
    safe_context = getattr(serialization, "safe_globals", None)
    if safe_context is not None:
        context = safe_context([argparse.Namespace])
    else:
        add_safe = getattr(serialization, "add_safe_globals", None)
        if add_safe is not None:
            add_safe([argparse.Namespace])
        context = nullcontext()
    with context:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def assert_controller_calibration_frozen(model) -> None:
    """Reject Adaptive checkpoints whose calibration buffers are not frozen."""
    checked = 0
    for name, module in model.named_modules():
        frozen = getattr(module, "frozen", None)
        if not torch.is_tensor(frozen) or not name.endswith("standardizer"):
            continue
        checked += 1
        if frozen.numel() != 1 or not bool(frozen.item()):
            raise RuntimeError(f"Controller calibration is not frozen: {name}")
        if bool(getattr(module, "_calibrating", False)):
            raise RuntimeError(f"Controller calibration is still active: {name}")
    if getattr(model, "rope_mode", None) == "adaptive" and checked == 0:
        raise RuntimeError("Adaptive model has no checkpointed calibration standardizer")


def load_model_checkpoint_strict(
    model,
    path,
    *,
    map_location="cpu",
    require_calibration_frozen=False,
):
    """Load a complete model checkpoint with zero missing or unexpected keys."""
    checkpoint = load_checkpoint(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping: {path}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint has no model state mapping: {path}")
    model.load_state_dict(state, strict=True)
    if require_calibration_frozen:
        assert_controller_calibration_frozen(model)
    return checkpoint
