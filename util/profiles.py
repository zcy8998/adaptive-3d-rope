"""Published model profiles and checkpoint compatibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "position_encoding_profiles.yaml"
_COMPATIBILITY_KEYS = (
    "model",
    "rope_mode",
    "rope_axes",
    "rope_theta",
    "encoder_pe",
    "decoder_pe",
    "controller_mode",
    "controller_features",
    "controller_token_groups",
    "controller_architecture",
    "controller_decoder_token_scope",
    "controller_token_normalization",
    "controller_scale_granularity",
    "controller_token_hidden_dim",
    "adaptive_scope",
)


def _load_config() -> dict:
    payload = yaml.safe_load(_CONFIG_PATH.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), dict):
        raise ValueError(f"Invalid profile configuration: {_CONFIG_PATH}")
    return payload


def profile_names() -> tuple[str, ...]:
    return tuple(_load_config()["profiles"])


def resolve_profile(name: str) -> dict:
    config = _load_config()
    try:
        profile = config["profiles"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown published profile: {name}") from exc
    values = dict(config.get("shared", {}))
    values.update(profile.get("model", {}))
    values["checkpoint"] = dict(profile["checkpoint"])
    return values


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    values = resolve_profile(args.profile)
    for key, value in values.items():
        if key != "checkpoint":
            setattr(args, key, value)
    return args


def validate_checkpoint_profile(checkpoint: dict, profile_name: str) -> None:
    """Reject a checkpoint whose saved model options disagree with its profile."""
    saved = checkpoint.get("args")
    if isinstance(saved, argparse.Namespace):
        saved = vars(saved)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint has no saved argument mapping for profile validation")
    expected = resolve_profile(profile_name)
    mismatches = []
    for key in _COMPATIBILITY_KEYS:
        if key not in saved:
            continue
        if str(saved[key]) != str(expected[key]):
            mismatches.append(f"{key}: checkpoint={saved[key]!r}, profile={expected[key]!r}")
    if mismatches:
        raise ValueError(
            f"Checkpoint does not match --profile {profile_name}: " + "; ".join(mismatches)
        )
