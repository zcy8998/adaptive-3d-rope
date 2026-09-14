"""CSI-only beam prediction on the LWM downstream interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lwm_adaptive_2d import LwmPortabilityModel, load_lwm11_checkpoint


class Residual1DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResidualBeamHead(nn.Module):
    """A fixed downstream head shared by LWM and Adaptive 2D-RoPE."""

    def __init__(self, embed_dim: int = 128, num_beams: int = 256) -> None:
        super().__init__()
        hidden = 128
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            Residual1DBlock(hidden),
            Residual1DBlock(hidden),
            Residual1DBlock(hidden),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_beams),
        )

    def forward(self, channel_embeddings: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, D, L]. Global pooling makes K=64 and K=128 compatible.
        x = channel_embeddings.transpose(1, 2).contiguous()
        return self.classifier(self.blocks(self.stem(x)))


class LwmBeamModel(nn.Module):
    """LWM encoder plus a CSI-only beam classifier.

    The observed Set-B beam measurements are deliberately not consumed here;
    this is the LWM-style CSI-to-beam transfer task, rather than the existing
    side-information-assisted beam-management adapter.
    """

    def __init__(
        self,
        method: str,
        checkpoint: str,
        *,
        controller_statistics: str = "mean_std",
        num_beams: int = 256,
        max_len: int = 513,
    ) -> None:
        super().__init__()
        self.backbone = LwmPortabilityModel(
            method,
            max_len=max_len,
            controller_statistics=controller_statistics,
        )
        self.load_audit = load_lwm11_checkpoint(self.backbone, checkpoint)
        self.head = ResidualBeamHead(embed_dim=128, num_beams=num_beams)
        self.num_beams = int(num_beams)

    def forward(
        self,
        input_ids: torch.Tensor,
        coordinates: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor]]:
        encoded, diagnostics = self.backbone.encode(input_ids, coordinates)
        channel_embeddings = encoded[:, 1:, :]
        logits = self.head(channel_embeddings)
        loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
        return loss, logits, diagnostics

