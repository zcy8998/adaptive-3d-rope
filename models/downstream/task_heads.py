import math

import torch
from torch import nn
from torch.nn import functional as F

from datasets.deepmimo.csi_grid import (
    batch_target_grid,
    flatten_grid,
    tokens_to_complex_grid,
)


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_ratio=0.125):
        super().__init__()
        hidden = max(4, int(dim * float(bottleneck_ratio)))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def _valid_mask(lengths, seq_len):
    return torch.arange(seq_len, device=lengths.device)[None, :] < lengths[:, None]


def _complex_features(x):
    return torch.cat([x.real, x.imag], dim=-1)


def _token_mask(batch, valid):
    if "token_mask" not in batch:
        return valid
    mask = batch["token_mask"].to(device=valid.device, dtype=torch.bool)
    return valid & mask


class MAEBackboneAdapter(nn.Module):
    def __init__(
        self,
        mae,
        bottleneck_ratio=0.125,
        adapter_depth=1,
        strip_pretraining_decoder=True,
    ):
        super().__init__()
        self.mae = mae
        if strip_pretraining_decoder:
            self._strip_pretraining_decoder()
        self.adapter = nn.Sequential(
            *[Adapter(mae.embed_dim, bottleneck_ratio) for _ in range(max(1, int(adapter_depth)))]
        )
        self.embed_dim = mae.embed_dim
        self.target_dim = 128

    def _strip_pretraining_decoder(self):
        """Remove modules that are not executed by downstream encoder transfer."""
        for attribute in (
            "mask_token",
            "decoder_embed",
            "decoder_pos_embed",
            "decoder_blocks",
            "decoder_norm",
            "decoder_pred",
            "dec_base_freqs",
            "dec_rope_controller",
        ):
            if hasattr(self.mae, attribute):
                setattr(self.mae, attribute, None)

    def encode(self, x, lengths, dims, meta=None, token_mask=None):
        raw_patches = x
        z = self.mae._patch_embed_inputs(x)
        batch_size, seq_len, _ = z.shape
        input_size = dims.T if dims.dim() == 2 and dims.shape[1] == 3 else dims
        grid_size = self.mae._compute_grid_size(input_size)
        ids = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(batch_size, seq_len)
        valid = ids < lengths[:, None]
        if token_mask is not None:
            valid = valid & token_mask.to(device=z.device, dtype=torch.bool)
        if self.mae.pos_embed is not None:
            z = self.mae.pos_embed(z, grid_size, ids_keep=None)
        descriptor = None
        if self.mae.rope_mode == "adaptive" and self.mae.controller_mode != "mean_std":
            descriptor = self.mae.coherence_descriptor(
                raw_patches, input_size, valid, phys_meta=meta
            )
        freqs_cis = self.mae._get_full_rope_frequencies(
            grid_size,
            z,
            mode="encoder",
            descriptor=descriptor,
            token_mask=valid,
        )
        if freqs_cis is not None:
            freqs_cis = self.mae._expand_decoder_rope(freqs_cis, batch_size, seq_len)
        axial_coords = None
        if self.mae.attention_backbone == "axial":
            axial_coords = self.mae._full_grid_coordinates(grid_size, z.device)
            axial_coords = axial_coords.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.mae.blocks:
            if self.mae.rope_enabled:
                z = block(
                    z,
                    freqs_cis=freqs_cis,
                    attn_mask=valid,
                    axial_coords=axial_coords,
                )
            else:
                z = block(z, attn_mask=valid)
        z = self.adapter(self.mae.norm(z))
        return z, valid


class CSIFeedbackAdapterModel(nn.Module):
    def __init__(
        self,
        backbone,
        compression_ratio=16,
        latent_dim=None,
        seq_len=None,
        target_dim=None,
        source_dim=None,
        codec_hidden_mult=1.0,
        codec_architecture="token_lowrank",
        codec_hidden_dim=64,
    ):
        super().__init__()
        self.backbone = backbone
        dim = backbone.embed_dim
        self.compression_ratio = int(compression_ratio)
        self.latent_dim = int(latent_dim) if latent_dim is not None else None
        self.ue_norm = nn.LayerNorm(dim)
        self.feedback_projector = None
        self.ue_encoder = None
        self.bs_decoder = None
        self.seq_len = None
        self.target_dim = None
        self.source_dim = int(source_dim) if source_dim is not None else None
        self.grid_shape = None
        self.codec_hidden_mult = float(codec_hidden_mult)
        self.codec_architecture = str(codec_architecture)
        self.codec_hidden_dim = int(codec_hidden_dim)
        if self.codec_architecture not in {"token_lowrank", "dense_legacy"}:
            raise ValueError(
                f"Unsupported CSI feedback codec architecture: {self.codec_architecture}"
            )
        if seq_len is not None and target_dim is not None:
            self._build_codec(int(seq_len), int(target_dim), self.source_dim, device=None)

    def _build_codec(self, seq_len, target_dim, source_dim, device):
        if (
            self.ue_encoder is not None
            and self.seq_len == seq_len
            and self.target_dim == target_dim
            and self.source_dim == source_dim
        ):
            return
        total_target = int(seq_len) * int(target_dim)
        source_dim = int(source_dim) if source_dim is not None else total_target
        code_dim = self.latent_dim or max(1, source_dim // max(1, self.compression_ratio))
        dim = self.backbone.embed_dim
        if self.codec_architecture == "dense_legacy":
            hidden_dim = max(dim, int(round(dim * self.codec_hidden_mult)))
            self.feedback_projector = nn.Linear(dim, target_dim)
            self.ue_encoder = nn.Sequential(
                nn.Flatten(1), nn.Linear(total_target, code_dim), nn.Tanh()
            )
            self.bs_decoder = nn.Sequential(
                nn.Linear(code_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, total_target),
            )
            self.code_channels = None
        else:
            # The feedback budget is defined relative to the *valid* real-valued
            # CSI grid.  Instead of flattening all padded patch features into a
            # 131072 x 2048 dense matrix, allocate a small number of real code
            # values to every encoder token.  For the 1 x 128 x 128 DeepMIMO
            # setting at compression ratio 16 this is exactly two values/token.
            self.code_channels = max(1, int(math.ceil(code_dim / int(seq_len))))
            hidden_dim = max(4, self.codec_hidden_dim)
            self.feedback_projector = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
            )
            self.ue_encoder = nn.Sequential(
                nn.Linear(hidden_dim, self.code_channels),
                nn.Tanh(),
            )
            self.bs_decoder = nn.Sequential(
                nn.Linear(self.code_channels, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, target_dim),
            )
        if device is not None:
            self.feedback_projector = self.feedback_projector.to(device)
            self.ue_encoder = self.ue_encoder.to(device)
            self.bs_decoder = self.bs_decoder.to(device)
        self.seq_len = int(seq_len)
        self.target_dim = int(target_dim)
        self.source_dim = int(source_dim)
        self.latent_dim = int(code_dim)

    def forward(self, batch):
        z, valid = self.backbone.encode(
            batch["x"],
            batch["lengths"],
            batch["dims"],
            batch.get("meta"),
            batch.get("token_mask"),
        )
        target_grid, grid_mask = batch_target_grid(batch)
        target_tokens = _complex_features(batch["x"])
        source_dim = int(grid_mask[0].sum().item())
        self._build_codec(z.shape[1], target_tokens.shape[-1], source_dim, z.device)
        z = self.feedback_projector(self.ue_norm(z))
        if self.codec_architecture == "dense_legacy":
            codeword = self.ue_encoder(z)
            pred_tokens = self.bs_decoder(codeword).view_as(target_tokens)
        else:
            code_tokens = self.ue_encoder(z)
            code_tokens = code_tokens * valid[:, :, None].to(code_tokens.dtype)
            dense_code = code_tokens.flatten(1)
            codeword = dense_code[:, : self.latent_dim]
            padded_length = z.shape[1] * self.code_channels
            if codeword.shape[1] < padded_length:
                codeword_for_decode = F.pad(
                    codeword, (0, padded_length - codeword.shape[1])
                )
            else:
                codeword_for_decode = codeword
            pred_tokens = self.bs_decoder(
                codeword_for_decode.view(z.shape[0], z.shape[1], self.code_channels)
            )
        patch_dim = batch["x"].shape[-1]
        pred_complex_tokens = torch.complex(
            pred_tokens[..., :patch_dim], pred_tokens[..., patch_dim:]
        )
        pred_grid_complex = tokens_to_complex_grid(pred_complex_tokens, batch["dims"])
        pred_grid = torch.stack([pred_grid_complex.real, pred_grid_complex.imag], dim=1)
        loss = ((pred_grid - target_grid) ** 2) * grid_mask.to(pred_grid.dtype)
        loss = loss.sum() / grid_mask.sum().clamp_min(1).to(pred_grid.dtype)
        pred_flat, _ = flatten_grid(pred_grid, grid_mask)
        return loss, pred_flat


class BeamManagementAdapterModel(nn.Module):
    def __init__(self, backbone, set_a_size=64, set_b_size=16, use_backbone=True, rsrp_loss_weight=0.05):
        super().__init__()
        self.backbone = backbone if use_backbone else None
        self.num_beams = int(set_a_size)
        hidden = max(128, backbone.embed_dim // 2 if backbone is not None else 128)
        self.set_b_size = int(set_b_size)
        self.rsrp_loss_weight = float(rsrp_loss_weight)
        self.rsrp_encoder = nn.Sequential(
            nn.Linear(self.set_b_size * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        input_dim = hidden + (backbone.embed_dim if self.backbone is not None else 0)
        self.classifier = nn.Linear(input_dim, self.num_beams)
        self.rsrp_head = nn.Linear(input_dim, self.num_beams)

    def forward(self, batch):
        rsrp = batch["rsrp_set_b"].float()
        set_b = batch["set_b"].float() / max(1.0, float(self.num_beams - 1))
        feat = torch.cat([rsrp, set_b], dim=-1)
        h = self.rsrp_encoder(feat)
        if self.backbone is not None and "x" in batch:
            z, valid = self.backbone.encode(
                batch["x"],
                batch["lengths"],
                batch["dims"],
                batch.get("meta"),
                batch.get("token_mask"),
            )
            valid = _token_mask(batch, valid)
            pooled = (z * valid[:, :, None].to(z.dtype)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(z.dtype)
            h = torch.cat([h, pooled], dim=-1)
        logits = self.classifier(h)
        pred_rsrp = self.rsrp_head(h)
        loss = torch.nn.functional.cross_entropy(logits, batch["label"].long())
        if "full_rsrp" in batch and self.rsrp_loss_weight > 0:
            target = batch["full_rsrp"].float()
            target = (target - target.mean(dim=1, keepdim=True)) / target.std(dim=1, keepdim=True).clamp_min(1.0)
            pred = (pred_rsrp - pred_rsrp.mean(dim=1, keepdim=True)) / pred_rsrp.std(dim=1, keepdim=True).clamp_min(1.0)
            loss = loss + self.rsrp_loss_weight * torch.nn.functional.smooth_l1_loss(pred, target)
        return loss, logits, pred_rsrp


class LoSAdapterModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone.embed_dim),
            nn.Linear(backbone.embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        z, valid = self.backbone.encode(
            batch["x"], batch["lengths"], batch["dims"],
            batch.get("meta"), batch.get("token_mask"),
        )
        weight = valid[:, :, None].to(z.dtype)
        pooled = (z * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, batch["label"].long())
        return loss, logits


class LoSBaselineModel(nn.Module):
    """Task-specific CSI classifiers; sequence order is token order, not time."""
    def __init__(self, kind, patch_dim=64, hidden=128):
        super().__init__()
        self.kind = kind
        self.input = nn.Linear(2 * patch_dim, hidden)
        if kind == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(2 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(0.1)
            )
            out_dim = 2 * hidden
        elif kind == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            )
            out_dim = hidden
        elif kind == "lstm":
            self.encoder = nn.LSTM(
                hidden, hidden // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=0.1,
            )
            out_dim = hidden
        else:
            raise ValueError(f"Unsupported LoS baseline: {kind}")
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, batch):
        x = self.input(_complex_features(batch["x"]))
        valid = _token_mask(batch, _valid_mask(batch["lengths"], x.shape[1]))
        weight = valid[:, :, None].to(x.dtype)
        if self.kind == "mlp":
            mean = (x * weight).sum(1) / weight.sum(1).clamp_min(1)
            var = (((x - mean[:, None]) * weight) ** 2).sum(1) / weight.sum(1).clamp_min(1)
            h = self.encoder(torch.cat([mean, var.sqrt()], dim=-1))
        elif self.kind == "cnn":
            h = self.encoder((x * weight).transpose(1, 2)).transpose(1, 2)
            h = (h * weight).sum(1) / weight.sum(1).clamp_min(1)
        else:
            h, _ = self.encoder(x * weight)
            h = (h * weight).sum(1) / weight.sum(1).clamp_min(1)
        logits = self.classifier(h)
        return F.cross_entropy(logits, batch["label"].long()), logits


class BeamBaselineModel(nn.Module):
    def __init__(self, kind, set_a_size=64, set_b_size=16, rsrp_loss_weight=0.05):
        super().__init__()
        self.kind = kind
        self.num_beams = int(set_a_size)
        self.set_b_size = int(set_b_size)
        self.rsrp_loss_weight = float(rsrp_loss_weight)
        hidden = 192
        feature_dim = 7
        if kind == "beam_codebook":
            return
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        if kind == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden, hidden, 7, padding=3, padding_mode="circular"),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
                nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular"),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
                nn.Conv1d(hidden, hidden, 3, padding=1, padding_mode="circular"),
                nn.GELU(),
            )
            self.classifier = nn.Linear(hidden, 1)
            self.rsrp_head = nn.Linear(hidden, 1)
        elif kind == "lstm":
            self.encoder = nn.LSTM(
                hidden,
                hidden // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.1,
            )
            self.classifier = nn.Linear(hidden, 1)
            self.rsrp_head = nn.Linear(hidden, 1)
        elif kind == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(self.num_beams * feature_dim, hidden * 4),
                nn.GELU(),
                nn.LayerNorm(hidden * 4),
                nn.Dropout(0.1),
                nn.Linear(hidden * 4, hidden * 2),
                nn.GELU(),
                nn.LayerNorm(hidden * 2),
            )
            self.classifier = nn.Linear(hidden * 2, self.num_beams)
            self.rsrp_head = nn.Linear(hidden * 2, self.num_beams)
        else:
            raise ValueError(f"Unsupported beam baseline: {kind}")

    def _beam_map_features(self, batch):
        rsrp = batch["rsrp_set_b"].float()
        set_b = batch["set_b"].long()
        bsz = rsrp.shape[0]
        device = rsrp.device
        dtype = rsrp.dtype
        observed = torch.zeros(bsz, self.num_beams, device=device, dtype=dtype)
        values = torch.zeros_like(observed)
        observed.scatter_(1, set_b, 1.0)
        values.scatter_(1, set_b, rsrp)

        obs_count = observed.sum(dim=1, keepdim=True).clamp_min(1.0)
        obs_mean = (values * observed).sum(dim=1, keepdim=True) / obs_count
        obs_var = (((values - obs_mean) * observed) ** 2).sum(dim=1, keepdim=True) / obs_count
        # RSRP is measured in dB.  Nearly equal Set-B observations otherwise
        # create an arbitrarily large auxiliary-regression target and caused
        # deterministic NaNs in the low-data CNN/LSTM runs.
        obs_std = obs_var.sqrt().clamp_min(1.0)
        values = (values - obs_mean) / obs_std
        values = values * observed
        interp = values.clone()
        if self.num_beams > 1:
            for idx in range(1, self.num_beams):
                missing = observed[:, idx] < 0.5
                interp[missing, idx] = interp[missing, idx - 1]
            for idx in range(self.num_beams - 2, -1, -1):
                missing = (observed[:, idx] < 0.5) & (interp[:, idx].abs() < 1e-8)
                interp[missing, idx] = interp[missing, idx + 1]

        pos = torch.arange(self.num_beams, device=device, dtype=dtype) / max(1, self.num_beams)
        angle = 2.0 * torch.pi * pos
        pos = pos.expand(bsz, -1)
        sin_pos = torch.sin(angle).expand(bsz, -1)
        cos_pos = torch.cos(angle).expand(bsz, -1)
        density = obs_count.expand(-1, self.num_beams) / max(1.0, float(self.num_beams))
        return torch.stack([values, interp, observed, pos, sin_pos, cos_pos, density], dim=-1), obs_mean, obs_std

    @staticmethod
    def _normalized_rsrp_target(full_rsrp, mean, std):
        return (full_rsrp.float() - mean) / std.clamp_min(1.0)

    def forward(self, batch):
        if self.kind == "beam_codebook":
            rsrp = batch["rsrp_set_b"].float()
            set_b = batch["set_b"].long()
            fill = float(rsrp.min().item()) - 5.0
            logits = torch.full((rsrp.shape[0], self.num_beams), fill, device=rsrp.device, dtype=rsrp.dtype)
            logits.scatter_(1, set_b, rsrp)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"].long())
            return loss, logits, None
        feat, obs_mean, obs_std = self._beam_map_features(batch)
        if self.kind == "mlp":
            h = self.encoder(feat)
            logits = self.classifier(h)
            pred_rsrp = self.rsrp_head(h)
        else:
            h = self.input(feat)
            if self.kind == "cnn":
                h = self.encoder(h.transpose(1, 2)).transpose(1, 2)
            elif self.kind == "lstm":
                h, _ = self.encoder(h)
            logits = self.classifier(h).squeeze(-1)
            pred_rsrp = self.rsrp_head(h).squeeze(-1)
        loss = torch.nn.functional.cross_entropy(logits, batch["label"].long())
        if "full_rsrp" in batch and self.rsrp_loss_weight > 0:
            target_rsrp = self._normalized_rsrp_target(batch["full_rsrp"], obs_mean, obs_std)
            loss = loss + self.rsrp_loss_weight * torch.nn.functional.smooth_l1_loss(pred_rsrp, target_rsrp)
        return loss, logits, pred_rsrp


_UNUSED_PRETRAINING_PREFIXES = (
    "backbone.mae.mask_token",
    "backbone.mae.decoder_",
    "backbone.mae.decoder_blocks.",
    "backbone.mae.dec_base_freqs",
    "backbone.mae.dec_rope_controller.",
)


def _is_unused_pretraining_parameter(name):
    return any(name.startswith(prefix) for prefix in _UNUSED_PRETRAINING_PREFIXES)


def configure_downstream_trainability(
    model,
    mode,
    unfreeze_last_blocks=0,
    train_backbone_norm=False,
):
    """Configure an auditable frozen-head, adapter, or full-transfer protocol.

    The task head is every parameter outside ``backbone``.  Adapter transfer
    additionally trains only ``backbone.adapter``.  Full transfer trains the
    downstream encoder path but keeps the unused MAE reconstruction decoder
    frozen so parameter counts and optimizer state describe the executed graph.
    """
    if mode not in {"head", "adapter", "full"}:
        raise ValueError(f"Unsupported downstream adaptation mode: {mode}")

    for name, param in model.named_parameters():
        is_task_head = not name.startswith("backbone.")
        is_adapter = name.startswith("backbone.adapter.")
        if mode == "head":
            trainable = is_task_head
        elif mode == "adapter":
            trainable = is_task_head or is_adapter
        else:
            trainable = not _is_unused_pretraining_parameter(name)
        if train_backbone_norm and ".mae.norm" in name and mode != "head":
            trainable = True
        param.requires_grad = trainable

    n_last = int(unfreeze_last_blocks)
    if n_last <= 0 or mode == "head":
        return
    mae = None
    if hasattr(model, "backbone") and model.backbone is not None and hasattr(model.backbone, "mae"):
        mae = model.backbone.mae
    if mae is None or not hasattr(mae, "blocks"):
        return
    for block in list(mae.blocks)[-n_last:]:
        for param in block.parameters():
            param.requires_grad = True


def freeze_backbone_for_adapters(model, unfreeze_last_blocks=0, train_backbone_norm=False):
    """Backward-compatible wrapper for the strict adapter protocol."""
    configure_downstream_trainability(
        model,
        "adapter",
        unfreeze_last_blocks=unfreeze_last_blocks,
        train_backbone_norm=train_backbone_norm,
    )
