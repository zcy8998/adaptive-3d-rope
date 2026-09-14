"""LWM 1.1 with original APE, learnable 2D RoPE, or token-only Adaptive 2D-RoPE."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LwmLayerNorm(nn.Module):
    """The normalization used by the public LWM 1.1 checkpoint."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class LwmEmbedding(nn.Module):
    def __init__(self, element_length: int, dim: int, max_len: int, use_ape: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(element_length, dim)
        self.pos_embed = nn.Embedding(max_len, dim) if use_ape else None
        self.norm = LwmLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_embedding = self.proj(x.float())
        if self.pos_embed is not None:
            positions = torch.arange(x.shape[1], device=x.device)
            output = token_embedding + self.pos_embed(positions)
        else:
            output = token_embedding
        return self.norm(output), token_embedding


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    paired = x.float().reshape(*shape[:-1], shape[-1] // 2, 2)
    even, odd = paired.unbind(dim=-1)
    output = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return output.flatten(-2).to(dtype=x.dtype)


class Adaptive2DController(nn.Module):
    """Map selected visible-token statistics to bounded axis-head scales."""

    def __init__(
        self,
        dim: int,
        heads: int,
        max_scale: float = 4.0,
        statistics: str = "mean_std",
    ) -> None:
        super().__init__()
        if statistics not in {"mean_std", "std"}:
            raise ValueError(f"Unsupported Controller statistics: {statistics}")
        self.heads = heads
        self.max_scale = float(max_scale)
        self.statistics = statistics
        context_dim = 2 * dim if statistics == "mean_std" else dim
        self.frozen_normalization = nn.LayerNorm(
            context_dim, elementwise_affine=False
        )
        self.projection = nn.Sequential(
            nn.Linear(context_dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2 * heads),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, tokens: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        weights = visible.to(tokens.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (tokens * weights).sum(dim=1) / count
        variance = ((tokens - mean[:, None]) ** 2 * weights).sum(dim=1) / count
        std = torch.sqrt(variance + 1e-6)
        context = torch.cat((mean, std), dim=-1) if self.statistics == "mean_std" else std
        logits = self.projection(self.frozen_normalization(context))
        logits = logits.view(tokens.shape[0], 2, self.heads)
        return torch.exp(math.log(self.max_scale) * torch.tanh(logits))


class RotaryAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        learnable: bool,
        adaptive: bool,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % heads or (dim // heads) % 4:
            raise ValueError("Adaptive 2D-RoPE requires a head dimension divisible by four")
        self.heads = heads
        self.head_dim = dim // heads
        self.axis_dim = self.head_dim // 2
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        pairs = self.axis_dim // 2
        base = 1.0 / (theta ** (torch.arange(pairs, dtype=torch.float32) / pairs))
        base = base[None, None].repeat(2, heads, 1)
        if learnable:
            self.log_inv_freq = nn.Parameter(base.log())
        else:
            self.register_buffer("log_inv_freq", base.log(), persistent=True)
        self.adaptive = adaptive

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coordinates: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs_q, outputs_k = [], []
        inv_freq = self.log_inv_freq.exp()
        for axis in range(2):
            start = axis * self.axis_dim
            stop = start + self.axis_dim
            position = coordinates[:, axis].float()
            angle = position[None, None, :, None] * inv_freq[axis][None, :, None, :]
            if scales is not None:
                angle = angle * scales[:, axis, :, None, None]
            cos, sin = angle.cos(), angle.sin()
            outputs_q.append(_rotate_pairs(q[..., start:stop], cos, sin))
            outputs_k.append(_rotate_pairs(k[..., start:stop], cos, sin))
        return torch.cat(outputs_q, dim=-1), torch.cat(outputs_k, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual, batch, length = x, x.shape[0], x.shape[1]
        q = self.W_Q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, coordinates, scales)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return residual + self.dropout(self.linear(context)), attention


class OriginalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_k = dim // heads
        self.d_v = dim // heads
        self.n_heads = heads
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *_args) -> tuple[torch.Tensor, torch.Tensor]:
        residual, batch = x, x.shape[0]
        q = self.W_Q(x).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, -1, self.n_heads * self.d_v)
        return residual + self.dropout(self.linear(context)), attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class LwmEncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, dim: int, dropout: float) -> None:
        super().__init__()
        self.enc_self_attn = attention
        self.pos_ffn = PositionwiseFeedForward(dim, dropout)
        self.norm1 = LwmLayerNorm(dim)
        self.norm2 = LwmLayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention = self.enc_self_attn(x, coordinates, scales)
        attention_output = self.norm1(x + attention_output)
        feedforward = self.pos_ffn(attention_output)
        return self.norm2(attention_output + feedforward), attention


class LwmPortabilityModel(nn.Module):
    METHODS = {"original", "learnable_2d", "proposed"}

    def __init__(
        self,
        method: str,
        element_length: int = 32,
        dim: int = 128,
        layers: int = 12,
        max_len: int = 513,
        heads: int = 8,
        dropout: float = 0.1,
        controller_max_scale: float = 4.0,
        controller_statistics: str = "mean_std",
    ) -> None:
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(self.METHODS)}")
        self.method = method
        self.embedding = LwmEmbedding(element_length, dim, max_len, use_ape=method == "original")
        self.controller = (
            Adaptive2DController(
                dim, heads, controller_max_scale, statistics=controller_statistics
            )
            if method == "proposed"
            else None
        )
        blocks = []
        for _ in range(layers):
            attention = (
                OriginalAttention(dim, heads, dropout)
                if method == "original"
                else RotaryAttention2D(dim, heads, dropout, learnable=True, adaptive=method == "proposed")
            )
            blocks.append(LwmEncoderLayer(attention, dim, dropout))
        self.layers = nn.ModuleList(blocks)
        self.linear = nn.Linear(dim, dim)
        self.norm = LwmLayerNorm(dim)
        self.decoder = nn.Linear(dim, element_length, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(element_length))

    def encode(
        self,
        input_ids: torch.Tensor,
        coordinates: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode a complete token sequence for downstream tasks.

        ``visible`` is optional because downstream beam prediction uses complete
        CSI. When omitted, every non-CLS token is visible. The same method is
        also used by masked reconstruction so the positional interface cannot
        diverge between pretraining and transfer.
        """
        output, raw_tokens = self.embedding(input_ids)
        scales = None
        if self.controller is not None:
            if visible is None:
                visible = torch.ones(
                    input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
                )
                visible[:, 0] = False
            scales = self.controller(raw_tokens, visible)
        for layer in self.layers:
            output, _ = layer(output, coordinates, scales)
        diagnostics = {"scales": scales} if scales is not None else {}
        return output, diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        masked_pos: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visible = torch.ones(
            input_ids.shape[:2], dtype=torch.bool, device=input_ids.device
        )
        visible[:, 0] = False
        if masked_pos.numel():
            visible.scatter_(1, masked_pos.long(), False)
        output, diagnostics = self.encode(input_ids, coordinates, visible)
        gather = masked_pos.long()[:, :, None].expand(-1, -1, output.shape[-1])
        hidden = torch.gather(output, 1, gather)
        prediction = self.decoder(self.norm(F.relu(self.linear(hidden)))) + self.decoder_bias
        return prediction, diagnostics


def load_lwm11_checkpoint(model: LwmPortabilityModel, checkpoint: str) -> dict[str, list[str]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    clean = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    incompatible = model.load_state_dict(clean, strict=False)
    allowed_unexpected = {"embedding.pos_embed.weight"} if model.method != "original" else set()
    unexpected = set(incompatible.unexpected_keys)
    # Some archived Adaptive 2D-RoPE checkpoints contain the legacy absolute
    # embedding table, while newer pure-relative checkpoints do not.  Accept
    # that optional legacy key, but keep the rest of the load strict.
    disallowed_unexpected = unexpected - allowed_unexpected
    if disallowed_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(disallowed_unexpected)}")
    allowed_missing_prefixes = ("controller.",)
    missing = set(incompatible.missing_keys)
    bad_missing = {
        key for key in missing
        if not key.endswith("log_inv_freq") and not key.startswith(allowed_missing_prefixes)
    }
    if bad_missing:
        raise RuntimeError(f"Missing checkpoint keys: {sorted(bad_missing)}")
    if model.method == "original" and (missing or unexpected):
        raise RuntimeError(f"Original LWM checkpoint must load strictly: {missing=}, {unexpected=}")
    return {"missing": sorted(missing), "unexpected": sorted(unexpected)}
