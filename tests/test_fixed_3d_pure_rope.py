import torch

from models.csi_mae import csi_mae_tiny
from util.checkpoint import load_model_checkpoint_strict


MODEL_OPTIONS = {
    "rope_axes": "3d",
    "rope_theta": 10,
    "pos_emb_type": "None",
    "decoder_pos_emb_type": "None",
}


def test_fixed_bank_matches_learnable_initialization_without_axis_collisions():
    torch.manual_seed(42)
    fixed = csi_mae_tiny(rope_mode="fixed", **MODEL_OPTIONS)
    torch.manual_seed(42)
    learnable = csi_mae_tiny(rope_mode="learnable", **MODEL_OPTIONS)
    assert fixed.freqs.shape == learnable.freqs.shape
    assert fixed.dec_freqs.shape == learnable.dec_freqs.shape
    assert "freqs" not in dict(fixed.named_parameters())
    assert isinstance(learnable.freqs, torch.nn.Parameter)


def test_fixed_model_forward_and_strict_checkpoint_round_trip(tmp_path):
    model = csi_mae_tiny(rope_mode="fixed", **MODEL_OPTIONS)
    csi = torch.randn(2, 8, 8, 8) + 1j * torch.randn(2, 8, 8, 8)
    patches = (
        csi.to(torch.complex64)
        .reshape(2, 2, 4, 2, 4, 2, 4)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .reshape(2, 8, 64)
    )
    loss, _, _ = model(
        patches,
        torch.tensor([8, 8]),
        input_size=torch.tensor([[8, 8], [8, 8], [8, 8]]),
        mask_ratio=0.5,
        mask_strategy="random",
    )
    assert torch.isfinite(loss)
    loss.backward()

    checkpoint = tmp_path / "fixed.pth"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = csi_mae_tiny(rope_mode="fixed", **MODEL_OPTIONS)
    load_model_checkpoint_strict(restored, checkpoint)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
