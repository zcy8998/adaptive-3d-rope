import argparse

import pytest

from util.profiles import apply_profile, profile_names, resolve_profile, validate_checkpoint_profile


def test_six_public_profiles_are_complete():
    assert profile_names() == ("ape_1d", "ape_3d", "fixed_1d", "fixed_3d", "learnable_3d", "proposed_std_only")
    for name in profile_names():
        profile = resolve_profile(name)
        assert profile["checkpoint"]["sha256"] and len(profile["checkpoint"]["sha256"]) == 64


def test_profile_applies_model_arguments():
    args = argparse.Namespace(profile="proposed_std_only")
    apply_profile(args)
    assert args.rope_mode == "adaptive"
    assert args.encoder_pe == args.decoder_pe == "none"
    assert args.controller_decoder_token_scope == "visible_only"


def test_checkpoint_profile_rejects_mismatch():
    checkpoint = {"args": {"rope_mode": "fixed", "rope_axes": "3d"}}
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_profile(checkpoint, "proposed_std_only")
