import torch

from models.csi_mae import csi_mae_tiny


def test_axis_frequency_scale_changes_only_selected_learnable_axis():
    model = csi_mae_tiny(
        rope_mode="learnable",
        rope_axes="3d",
        pos_emb_type="None",
        decoder_pos_emb_type="None",
        rope_frequency_scale=(2.0, 1.0, 1.0),
    )
    frequency = model.freqs.detach()
    scale = frequency.new_tensor(model.rope_frequency_scale).view(3, 1, 1)
    scaled = frequency * scale
    torch.testing.assert_close(scaled[0], 2.0 * frequency[0])
    torch.testing.assert_close(scaled[1], frequency[1])
    torch.testing.assert_close(scaled[2], frequency[2])


def test_axis_frequency_scale_must_be_positive_triplet():
    try:
        csi_mae_tiny(
            rope_mode="learnable",
            pos_emb_type="None",
            decoder_pos_emb_type="None",
            rope_frequency_scale=(1.0, 0.0, 1.0),
        )
    except ValueError as error:
        assert "three positive values" in str(error)
    else:
        raise AssertionError("Expected invalid rotary scale to fail")
