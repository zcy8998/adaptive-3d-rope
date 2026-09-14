import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


SPEED_OF_LIGHT = 299792458.0
DEFAULT_LAGS = (1, 2, 4)
AXIS_NAMES = ("t", "k", "u")
DESCRIPTOR_GROUPS = (
    "rho",
    "validity",
    "power",
    "raw_mean",
    "raw_std",
    "physics_meta",
    "size_meta",
)
TOKEN_GROUPS = ("token_mean", "token_std")
ALL_CONTROLLER_GROUPS = DESCRIPTOR_GROUPS + TOKEN_GROUPS


def parse_csv(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


def validate_groups(groups, allowed, label):
    groups = parse_csv(groups)
    unknown = sorted(set(groups) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {label}: {unknown}; allowed={allowed}")
    return groups


def _feature_names(include_metadata: bool, lags: Sequence[int] = DEFAULT_LAGS):
    names = []
    for axis in AXIS_NAMES:
        names.extend(f"rho_{axis}_{lag}" for lag in lags)
    for axis in AXIS_NAMES:
        names.extend(f"pair_ratio_{axis}_{lag}" for lag in lags)
    names.extend(("log_power", "mean_real", "mean_imag", "std_real", "std_imag"))
    if include_metadata:
        names.extend(
            (
                "log10_fc_ghz",
                "log10_delta_f_15khz",
                "log10_delta_t_1ms",
                "spacing_wavelength",
                "log2_t_16",
                "log2_k_64",
                "log2_u_16",
            )
        )
    return tuple(names)


def _legacy_feature_groups(include_metadata: bool, lags: Sequence[int]):
    groups = ["rho"] * (3 * len(lags))
    groups.extend(["validity"] * (3 * len(lags)))
    groups.extend(("power", "raw_mean", "raw_mean", "raw_std", "raw_std"))
    if include_metadata:
        groups.extend(
            (
                "physics_meta",
                "physics_meta",
                "physics_meta",
                "legacy_spacing",
                "size_meta",
                "size_meta",
                "size_meta",
            )
        )
    return tuple(groups)


class RunningFeatureStandardizer(nn.Module):
    """Checkpointed, explicitly calibrated normalization for controller inputs.

    Formal runs calibrate this module on a balanced pass over the pretraining
    domains and then freeze it.  This avoids the domain-order drift caused by
    updating an exponential moving average every time a different mask or CSI
    configuration is visited.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.01,
        eps: float = 1e-5,
        clip_value: float = 5.0,
    ):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.clip_value = float(clip_value)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_sum", torch.zeros(num_features))
        self.register_buffer("calibration_sq_sum", torch.zeros(num_features))
        self.register_buffer("calibration_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self._calibrating = False

    def begin_calibration(self):
        self.calibration_sum.zero_()
        self.calibration_sq_sum.zero_()
        self.calibration_count.zero_()
        self.frozen.zero_()
        self._calibrating = True

    @torch.no_grad()
    def accumulate(self, features: torch.Tensor):
        values = features.detach().float()
        self.calibration_sum.add_(values.sum(dim=0))
        self.calibration_sq_sum.add_(values.square().sum(dim=0))
        self.calibration_count.add_(values.shape[0])

    @torch.no_grad()
    def finalize_calibration(self):
        if self.calibration_count.item() <= 0:
            raise RuntimeError("Controller standardizer received no calibration samples")
        count = self.calibration_count.to(self.calibration_sum.dtype)
        mean = self.calibration_sum / count
        var = self.calibration_sq_sum / count - mean.square()
        self.running_mean.copy_(mean)
        self.running_var.copy_(var.clamp_min(self.eps))
        self.num_updates.copy_(self.calibration_count)
        self.frozen.fill_(True)
        self._calibrating = False

    @torch.no_grad()
    def _update(self, features: torch.Tensor):
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0, unbiased=False)
        if self.num_updates.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp_min(self.eps))
        else:
            delta = batch_mean - self.running_mean
            self.running_var.copy_(
                (1.0 - self.momentum)
                * (self.running_var + self.momentum * delta.square())
                + self.momentum * batch_var
            )
            self.running_mean.add_(delta, alpha=self.momentum)
        self.num_updates.add_(1)

    def forward(self, features: torch.Tensor):
        original_dtype = features.dtype
        features = features.float()
        if self._calibrating:
            self.accumulate(features)
        elif self.training and not bool(self.frozen.item()):
            self._update(features.detach())
        normalized = (features - self.running_mean) * torch.rsqrt(
            self.running_var.clamp_min(self.eps)
        )
        normalized = normalized.clamp(-self.clip_value, self.clip_value)
        return normalized.to(original_dtype)


def _group_mask(groups_by_index, selected, reference):
    selected = set(parse_csv(selected))
    return reference.new_tensor(
        [group in selected for group in groups_by_index], dtype=torch.bool
    )


def _apply_descriptor_interventions(
    descriptor,
    standardizer,
    groups_by_index,
    neutralize_groups=(),
    shuffle_groups=(),
    shuffle_offset=1,
):
    values = descriptor
    if neutralize_groups:
        mask = _group_mask(groups_by_index, neutralize_groups, descriptor)
        mean = standardizer.running_mean.unsqueeze(0).expand_as(descriptor)
        values = torch.where(mask.unsqueeze(0), mean, values)
    if shuffle_groups:
        mask = _group_mask(groups_by_index, shuffle_groups, descriptor)
        shuffled = values.roll(shifts=int(shuffle_offset), dims=0)
        values = torch.where(mask.unsqueeze(0), shuffled, values)
    return values


class CoherenceRoPEController(nn.Module):
    """Maps observable CSI descriptors to bounded axis/head RoPE scales."""

    def __init__(
        self,
        descriptor_dim: int,
        num_heads: int,
        hidden_dim: int = 128,
        max_scale: float = 4.0,
        feature_groups=None,
        neutralize_groups=(),
        shuffle_groups=(),
        shuffle_offset: int = 1,
    ):
        super().__init__()
        if max_scale <= 1.0:
            raise ValueError("max_scale must be greater than one")
        self.num_heads = int(num_heads)
        self.max_log_scale = math.log(float(max_scale))
        self.feature_groups = tuple(feature_groups or ("descriptor",) * descriptor_dim)
        self.neutralize_groups = parse_csv(neutralize_groups)
        self.shuffle_groups = parse_csv(shuffle_groups)
        self.shuffle_offset = int(shuffle_offset)
        self.standardizer = RunningFeatureStandardizer(descriptor_dim)
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.num_heads),
        )
        self.last_descriptor = None
        self.last_scale = None
        self.last_scale_live = None
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, descriptor, base_freq_param, ablation="none"):
        if ablation not in {"none", "off", "shuffle", "mean"}:
            raise ValueError(f"Unsupported controller ablation: {ablation}")

        if ablation == "off":
            scale = descriptor.new_ones(descriptor.shape[0], 3, self.num_heads)
        else:
            if ablation == "shuffle":
                descriptor = descriptor.roll(shifts=self.shuffle_offset, dims=0)
                tokens = tokens.roll(shifts=self.shuffle_offset, dims=0)
            elif ablation == "mean":
                descriptor = self.standardizer.running_mean.unsqueeze(0).expand_as(
                    descriptor
                )
            descriptor = _apply_descriptor_interventions(
                descriptor,
                self.standardizer,
                self.feature_groups,
                self.neutralize_groups,
                self.shuffle_groups,
                self.shuffle_offset,
            )
            with torch.autocast(device_type=descriptor.device.type, enabled=False):
                normalized = self.standardizer(descriptor.float())
                logits = self.mlp(normalized).view(-1, 3, self.num_heads)
                scale = torch.exp(self.max_log_scale * torch.tanh(logits))

        self.last_descriptor = descriptor.detach()
        self.last_scale = scale.detach()
        return base_freq_param.float().unsqueeze(0) * scale.float().unsqueeze(-1)


class HybridCoherenceRoPEController(nn.Module):
    """Combines explicit coherence descriptors with compressed token statistics."""

    def __init__(
        self,
        descriptor_dim: int,
        token_dim: int,
        num_heads: int,
        head_dim: int,
        descriptor_hidden_dim: int = 128,
        max_scale: float = 2.0,
        feature_groups=None,
        neutralize_groups=(),
        shuffle_groups=(),
        shuffle_offset: int = 1,
    ):
        super().__init__()
        if max_scale <= 1.0:
            raise ValueError("max_scale must be greater than one")
        self.num_heads = int(num_heads)
        self.half_head_dim = int(head_dim) // 2
        self.max_log_scale = math.log(float(max_scale))
        self.feature_groups = tuple(feature_groups or ("descriptor",) * descriptor_dim)
        self.neutralize_groups = parse_csv(neutralize_groups)
        self.shuffle_groups = parse_csv(shuffle_groups)
        self.shuffle_offset = int(shuffle_offset)
        token_hidden_dim = max(64, min(256, token_dim // 4))
        self.standardizer = RunningFeatureStandardizer(descriptor_dim)
        self.token_standardizer = RunningFeatureStandardizer(2 * token_dim)
        self.descriptor_projection = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_hidden_dim), nn.GELU()
        )
        self.token_projection = nn.Sequential(
            nn.Linear(2 * token_dim, token_hidden_dim), nn.GELU()
        )
        self.output = nn.Linear(
            descriptor_hidden_dim + token_hidden_dim,
            3 * self.num_heads * self.half_head_dim,
        )
        self.last_descriptor = None
        self.last_scale = None
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, descriptor, tokens, base_freq_param, ablation="none"):
        if ablation not in {"none", "off", "shuffle", "mean"}:
            raise ValueError(f"Unsupported controller ablation: {ablation}")
        if ablation == "off":
            scale = descriptor.new_ones(
                descriptor.shape[0], 3, self.num_heads, self.half_head_dim
            )
        else:
            if ablation == "shuffle":
                descriptor = descriptor.roll(shifts=self.shuffle_offset, dims=0)
            elif ablation == "mean":
                descriptor = self.standardizer.running_mean.unsqueeze(0).expand_as(
                    descriptor
                )
            descriptor = _apply_descriptor_interventions(
                descriptor,
                self.standardizer,
                self.feature_groups,
                self.neutralize_groups,
                self.shuffle_groups,
                self.shuffle_offset,
            )
            with torch.autocast(device_type=descriptor.device.type, enabled=False):
                normalized_descriptor = self.standardizer(descriptor.float())
                token_values = tokens.float()
                token_mean = token_values.mean(dim=1)
                token_std = torch.sqrt(
                    token_values.var(dim=1, unbiased=False) + 1e-6
                )
                token_context = torch.cat((token_mean, token_std), dim=-1)
                self.token_standardizer(token_context)
                if ablation == "mean":
                    token_context = self.token_standardizer.running_mean.unsqueeze(0).expand_as(
                        token_context
                    )
                token_group_layout = ("token_mean",) * token_mean.shape[-1] + (
                    "token_std",
                ) * token_std.shape[-1]
                token_context = _apply_descriptor_interventions(
                    token_context,
                    self.token_standardizer,
                    token_group_layout,
                    self.neutralize_groups,
                    self.shuffle_groups,
                    self.shuffle_offset,
                )
                context = torch.cat(
                    (
                        self.descriptor_projection(normalized_descriptor),
                        self.token_projection(token_context),
                    ),
                    dim=-1,
                )
                logits = self.output(context).view(
                    -1, 3, self.num_heads, self.half_head_dim
                )
                scale = torch.exp(self.max_log_scale * torch.tanh(logits))
        self.last_descriptor = descriptor.detach()
        self.last_scale = scale.detach()
        return base_freq_param.float().unsqueeze(0) * scale.float()


class CompactCoherenceRoPEController(nn.Module):
    """Configurable compact controller used only by newly trained ablations."""

    def __init__(
        self,
        descriptor_dim,
        token_dim,
        num_heads,
        head_dim,
        feature_groups,
        token_groups=(),
        scale_granularity="axis_head",
        hidden_dim=128,
        token_hidden_dim=0,
        token_normalization="raw",
        token_pool="channel",
        architecture="dual_branch",
        fusion_hidden_dim=64,
        max_scale=4.0,
        raw_descriptor_groups=(),
        context_mode="sample",
        neutralize_groups=(),
        shuffle_groups=(),
        shuffle_offset=1,
    ):
        super().__init__()
        if scale_granularity not in {"axis_head", "axis_head_frequency"}:
            raise ValueError(f"Unsupported scale granularity: {scale_granularity}")
        if token_normalization not in {"raw", "frozen", "sample"}:
            raise ValueError(
                f"Unsupported token normalization: {token_normalization}"
            )
        if token_pool not in {"channel", "head"}:
            raise ValueError(f"Unsupported token pooling: {token_pool}")
        if architecture not in {
            "dual_branch",
            "fused_mlp",
            "direct_linear",
            "token_only",
        }:
            raise ValueError(f"Unsupported controller architecture: {architecture}")
        if context_mode not in {"sample", "constant"}:
            raise ValueError(f"Unsupported controller context mode: {context_mode}")
        self.num_heads = int(num_heads)
        self.half_head_dim = int(head_dim) // 2
        self.token_dim = int(token_dim)
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.scale_granularity = scale_granularity
        self.max_log_scale = math.log(float(max_scale))
        self.token_normalization = str(token_normalization)
        self.token_pool = str(token_pool)
        self.architecture = str(architecture)
        self.context_mode = str(context_mode)
        self.feature_groups = tuple(feature_groups)
        self.raw_descriptor_groups = validate_groups(
            raw_descriptor_groups, DESCRIPTOR_GROUPS, "raw descriptor groups"
        )
        self.token_groups = validate_groups(token_groups, TOKEN_GROUPS, "token groups")
        if self.architecture == "token_only":
            if int(descriptor_dim) != 0 or self.feature_groups:
                raise ValueError(
                    "token_only controller requires descriptor_dim=0 and no "
                    "descriptor feature groups"
                )
            if self.raw_descriptor_groups:
                raise ValueError(
                    "token_only controller cannot use raw descriptor groups"
                )
            if not self.token_groups:
                raise ValueError(
                    "token_only controller requires token_mean and/or token_std"
                )
        self.neutralize_groups = parse_csv(neutralize_groups)
        self.shuffle_groups = parse_csv(shuffle_groups)
        self.shuffle_offset = int(shuffle_offset)

        self.standardizer = (
            None
            if self.architecture == "token_only"
            else RunningFeatureStandardizer(descriptor_dim)
        )
        descriptor_hidden_dim = int(hidden_dim)
        token_input_dim = 0
        if self.token_groups:
            token_width = (
                self.token_dim if self.token_pool == "channel" else self.num_heads
            )
            self.token_stat_width = token_width
            self.token_standardizer = RunningFeatureStandardizer(2 * token_width)
            token_input_dim = token_width * len(self.token_groups)
        else:
            self.token_standardizer = None

        self.descriptor_projection = None
        self.token_projection = None
        self.fusion = None
        if self.architecture == "dual_branch":
            self.descriptor_projection = nn.Sequential(
                nn.Linear(descriptor_dim, descriptor_hidden_dim), nn.GELU()
            )
            context_dim = descriptor_hidden_dim
            if self.token_groups:
                if int(token_hidden_dim) <= 0:
                    token_hidden_dim = max(64, min(256, self.token_dim // 4))
                token_hidden_dim = int(token_hidden_dim)
                self.token_projection = nn.Sequential(
                    nn.Linear(token_input_dim, token_hidden_dim), nn.GELU()
                )
                context_dim += token_hidden_dim
        elif self.architecture == "token_only":
            if int(token_hidden_dim) <= 0:
                token_hidden_dim = max(64, min(256, self.token_dim // 4))
            token_hidden_dim = int(token_hidden_dim)
            self.token_projection = nn.Sequential(
                nn.Linear(token_input_dim, token_hidden_dim), nn.GELU()
            )
            context_dim = token_hidden_dim
        elif self.architecture == "fused_mlp":
            fusion_hidden_dim = int(fusion_hidden_dim)
            if fusion_hidden_dim <= 0:
                raise ValueError("fusion_hidden_dim must be positive for fused_mlp")
            self.fusion = nn.Sequential(
                nn.Linear(descriptor_dim + token_input_dim, fusion_hidden_dim),
                nn.GELU(),
            )
            context_dim = fusion_hidden_dim
        else:
            context_dim = descriptor_dim + token_input_dim
        per_head = 1 if scale_granularity == "axis_head" else self.half_head_dim
        self.output = nn.Linear(context_dim, 3 * self.num_heads * per_head)
        self.last_descriptor = None
        self.last_scale = None
        self.last_scale_live = None
        self.last_logits = None
        self.last_token_raw = None
        self.last_token_context = None
        # Diagnostics are useful for audits, but retaining per-forward tensors
        # is unnecessary on the latency-critical inference path.
        self.record_diagnostics = True
        self.fast_inference = False
        self._fast_std_mean = None
        self._fast_std_inv = None
        self.reset_output()

    def set_fast_inference(self, enabled=True):
        """Disable diagnostic tensor retention without changing controller math."""
        self.fast_inference = bool(enabled)
        self.record_diagnostics = not self.fast_inference
        if enabled:
            self.last_descriptor = None
            self.last_scale = None
            self.last_scale_live = None
            self.last_logits = None
            self.last_token_raw = None
            self.last_token_context = None
            if (
                self.token_groups == ("token_std",)
                and self.token_normalization == "frozen"
                and self.token_standardizer is not None
            ):
                offset = self.token_stat_width
                standardizer = self.token_standardizer
                self._fast_std_mean = standardizer.running_mean[offset:]
                self._fast_std_inv = torch.rsqrt(
                    standardizer.running_var[offset:].clamp_min(standardizer.eps)
                )
        else:
            self._fast_std_mean = None
            self._fast_std_inv = None
        return self

    def reset_output(self):
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _sample_standardize(values):
        mean = values.mean(dim=-1, keepdim=True)
        variance = values.var(dim=-1, unbiased=False, keepdim=True)
        return ((values - mean) * torch.rsqrt(variance + 1e-6)).clamp(-5.0, 5.0)

    def _pooled_token_statistics(self, tokens, token_mask=None):
        token_values = tokens.float()
        batch, length, channels = token_values.shape
        if token_mask is None:
            token_mask = torch.ones(
                batch, length, device=tokens.device, dtype=torch.bool
            )
        if token_mask.shape != (batch, length):
            raise ValueError(
                f"token_mask must have shape {(batch, length)}, got {tuple(token_mask.shape)}"
            )
        if self.token_pool == "channel":
            weights = token_mask.to(token_values.dtype).unsqueeze(-1)
            count = weights.sum(dim=1).clamp_min(1.0)
            token_mean = (token_values * weights).sum(dim=1) / count
            second = (token_values.square() * weights).sum(dim=1) / count
        else:
            head_dim = channels // self.num_heads
            values = token_values.view(batch, length, self.num_heads, head_dim)
            weights = token_mask.to(token_values.dtype).view(batch, length, 1, 1)
            count = (weights.sum(dim=1) * head_dim).clamp_min(1.0)
            token_mean = (values * weights).sum(dim=(1, 3)) / count.squeeze(-1)
            second = (values.square() * weights).sum(dim=(1, 3)) / count.squeeze(-1)
        token_std = torch.sqrt((second - token_mean.square()).clamp_min(0.0) + 1e-6)
        return token_mean, token_std

    def _token_context(self, tokens, token_mask=None, use_calibration_mean=False):
        token_mean, token_std = self._pooled_token_statistics(tokens, token_mask)
        # The submission model consumes only token_std.  During fixed eval
        # inference, avoid concatenating and normalizing the unused mean half.
        if (
            self.fast_inference
            and not self.training
            and self.token_groups == ("token_std",)
            and self.token_normalization == "frozen"
            and not use_calibration_mean
            and not self.neutralize_groups
            and not self.shuffle_groups
        ):
            standardizer = self.token_standardizer
            mean = self._fast_std_mean
            inv = self._fast_std_inv
            if mean is None or inv is None:
                offset = token_mean.shape[-1]
                mean = standardizer.running_mean[offset:]
                inv = torch.rsqrt(standardizer.running_var[offset:].clamp_min(standardizer.eps))
            result = (token_std - mean) * inv
            return result.clamp(-standardizer.clip_value, standardizer.clip_value)
        full_context = torch.cat((token_mean, token_std), dim=-1)
        if use_calibration_mean:
            full_context = self.token_standardizer.running_mean.unsqueeze(0).expand_as(
                full_context
            )
        layout = ("token_mean",) * token_mean.shape[-1] + (
            "token_std",
        ) * token_std.shape[-1]
        full_context = _apply_descriptor_interventions(
            full_context,
            self.token_standardizer,
            layout,
            self.neutralize_groups,
            self.shuffle_groups,
            self.shuffle_offset,
        )
        normalized_context = self.token_standardizer(full_context)
        if self.token_normalization == "raw":
            projected_context = full_context
        elif self.token_normalization == "frozen":
            projected_context = normalized_context
        else:
            projected_context = torch.cat(
                (
                    self._sample_standardize(full_context[:, : token_mean.shape[-1]]),
                    self._sample_standardize(full_context[:, token_mean.shape[-1] :]),
                ),
                dim=-1,
            )
        selected = []
        if "token_mean" in self.token_groups:
            selected.append(projected_context[:, : token_mean.shape[-1]])
        if "token_std" in self.token_groups:
            selected.append(projected_context[:, token_mean.shape[-1] :])
        result = torch.cat(selected, dim=-1)
        if self.record_diagnostics:
            self.last_token_raw = full_context.detach()
            self.last_token_context = result.detach()
        return result

    def forward(
        self, descriptor, tokens, base_freq_param, ablation="none", token_mask=None
    ):
        if ablation not in {"none", "off", "shuffle", "mean"}:
            raise ValueError(f"Unsupported controller ablation: {ablation}")
        batch = tokens.shape[0]
        per_head = 1 if self.scale_granularity == "axis_head" else self.half_head_dim
        if ablation == "off":
            scale = tokens.new_ones(batch, 3, self.num_heads, per_head)
        else:
            if ablation == "shuffle":
                if descriptor is not None:
                    descriptor = descriptor.roll(shifts=self.shuffle_offset, dims=0)
                tokens = tokens.roll(shifts=self.shuffle_offset, dims=0)
                if token_mask is not None:
                    token_mask = token_mask.roll(shifts=self.shuffle_offset, dims=0)
            elif ablation == "mean" and descriptor is not None:
                descriptor = self.standardizer.running_mean.unsqueeze(0).expand_as(
                    descriptor
                )
            normalized_descriptor = None
            if self.architecture != "token_only":
                if descriptor is None:
                    raise ValueError(
                        f"{self.architecture} controller requires a descriptor"
                    )
                descriptor = _apply_descriptor_interventions(
                    descriptor,
                    self.standardizer,
                    self.feature_groups,
                    self.neutralize_groups,
                    self.shuffle_groups,
                    self.shuffle_offset,
                )
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                if descriptor is not None:
                    normalized_descriptor = self.standardizer(descriptor.float())
                    if self.raw_descriptor_groups:
                        raw_mask = _group_mask(
                            self.feature_groups,
                            self.raw_descriptor_groups,
                            descriptor,
                        )
                        normalized_descriptor = torch.where(
                            raw_mask.unsqueeze(0),
                            descriptor.float(),
                            normalized_descriptor,
                        )
                    if self.context_mode == "constant":
                        normalized_descriptor = torch.ones_like(normalized_descriptor)
                token_context = None
                if self.token_groups:
                    token_context = self._token_context(
                        tokens,
                        token_mask=token_mask,
                        use_calibration_mean=ablation == "mean",
                    )
                    if self.context_mode == "constant":
                        token_context = torch.ones_like(token_context)
                if self.architecture == "dual_branch":
                    contexts = [self.descriptor_projection(normalized_descriptor)]
                    if token_context is not None:
                        contexts.append(self.token_projection(token_context))
                    context = torch.cat(contexts, dim=-1)
                elif self.architecture == "token_only":
                    context = self.token_projection(token_context)
                else:
                    contexts = [normalized_descriptor]
                    if token_context is not None:
                        contexts.append(token_context)
                    context = torch.cat(contexts, dim=-1)
                    if self.fusion is not None:
                        context = self.fusion(context)
                logits = self.output(context).view(
                    batch, 3, self.num_heads, per_head
                )
                scale = torch.exp(self.max_log_scale * torch.tanh(logits))
        if self.record_diagnostics:
            self.last_descriptor = None if descriptor is None else descriptor.detach()
            self.last_scale = scale.detach()
            self.last_scale_live = scale
            self.last_logits = None if ablation == "off" else logits.detach()
        return base_freq_param.float().unsqueeze(0) * scale.float()


class MaskAwareCoherenceDescriptor(nn.Module):
    """Computes raw-CSI axis correlations using only visible patch contents."""

    def __init__(
        self,
        patch_size: int = 4,
        lags: Iterable[int] = DEFAULT_LAGS,
        include_metadata: bool = True,
        feature_groups=None,
        validity_mode: str = "per_lag",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.lags = tuple(int(lag) for lag in lags)
        self.include_metadata = bool(include_metadata)
        self.validity_mode = str(validity_mode)
        if self.validity_mode not in {"per_lag", "axis_mean", "none"}:
            raise ValueError(f"Unsupported validity_mode: {self.validity_mode}")
        self.legacy_layout = feature_groups is None
        if self.legacy_layout:
            self.selected_groups = None
        else:
            self.selected_groups = validate_groups(
                feature_groups, DESCRIPTOR_GROUPS, "descriptor feature groups"
            )
            if not self.selected_groups:
                raise ValueError("Compact descriptor requires at least one feature group")
        self.eps = float(eps)
        self.feature_names, self.feature_groups = self._build_feature_layout()

    def _build_feature_layout(self):
        if self.legacy_layout:
            return (
                _feature_names(self.include_metadata, self.lags),
                _legacy_feature_groups(self.include_metadata, self.lags),
            )
        names = []
        groups = []
        if "rho" in self.selected_groups:
            names.extend(
                f"rho_{axis}_{lag}" for axis in AXIS_NAMES for lag in self.lags
            )
            groups.extend(["rho"] * (3 * len(self.lags)))
        if "validity" in self.selected_groups and self.validity_mode != "none":
            if self.validity_mode == "per_lag":
                names.extend(
                    f"pair_ratio_{axis}_{lag}"
                    for axis in AXIS_NAMES
                    for lag in self.lags
                )
                groups.extend(["validity"] * (3 * len(self.lags)))
            else:
                names.extend(f"validity_{axis}" for axis in AXIS_NAMES)
                groups.extend(["validity"] * 3)
        mapping = (
            ("power", ("log_power",)),
            ("raw_mean", ("mean_real", "mean_imag")),
            ("raw_std", ("std_real", "std_imag")),
            (
                "physics_meta",
                ("log10_fc_ghz", "log10_delta_f_15khz", "log10_delta_t_1ms"),
            ),
            ("size_meta", ("log2_t_16", "log2_k_64", "log2_u_16")),
        )
        for group, group_names in mapping:
            if group in self.selected_groups:
                names.extend(group_names)
                groups.extend([group] * len(group_names))
        return tuple(names), tuple(groups)

    @property
    def output_dim(self):
        return len(self.feature_names)

    def _assemble_features(self, correlations, pair_ratios, global_features, metadata):
        if self.legacy_layout:
            parts = [correlations, pair_ratios, global_features]
            if self.include_metadata:
                parts.append(metadata)
            return torch.cat(parts, dim=-1)

        parts = []
        if "rho" in self.selected_groups:
            parts.append(correlations)
        if "validity" in self.selected_groups and self.validity_mode != "none":
            if self.validity_mode == "per_lag":
                parts.append(pair_ratios)
            else:
                parts.append(
                    pair_ratios.reshape(*pair_ratios.shape[:-1], 3, len(self.lags)).mean(-1)
                )
        if "power" in self.selected_groups:
            parts.append(global_features[..., 0:1])
        if "raw_mean" in self.selected_groups:
            parts.append(global_features[..., 1:3])
        if "raw_std" in self.selected_groups:
            parts.append(global_features[..., 3:5])
        if "physics_meta" in self.selected_groups:
            parts.append(metadata[..., 0:3])
        if "size_meta" in self.selected_groups:
            parts.append(metadata[..., 4:7])
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _sample_dims(input_size, index):
        dims = []
        for values in input_size:
            if torch.is_tensor(values):
                value = values[index] if values.ndim else values
                dims.append(int(value.item()))
            else:
                dims.append(int(values[index] if isinstance(values, (list, tuple)) else values))
        return tuple(dims)

    def _unpatchify(self, patches, dims, visible_tokens):
        t, k, u = dims
        p = self.patch_size
        if t % p or k % p or u % p:
            raise ValueError(f"Input dimensions {dims} must be divisible by patch size {p}")
        tb, kb, ub = t // p, k // p, u // p
        count = tb * kb * ub
        grid = patches[:count].reshape(tb, kb, ub, p, p, p)
        csi = grid.permute(0, 3, 1, 4, 2, 5).reshape(t, k, u)
        patch_visible = visible_tokens[:count].reshape(tb, kb, ub)
        visible = patch_visible.repeat_interleave(p, 0).repeat_interleave(p, 1).repeat_interleave(p, 2)
        return csi, visible

    def _unpatchify_batch(self, patches, dims, visible_tokens):
        t, k, u = dims
        p = self.patch_size
        tb, kb, ub = t // p, k // p, u // p
        count = tb * kb * ub
        batch = patches.shape[0]
        grid = patches[:, :count].reshape(batch, tb, kb, ub, p, p, p)
        csi = grid.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, t, k, u)
        patch_visible = visible_tokens[:, :count].reshape(batch, tb, kb, ub)
        visible = patch_visible.repeat_interleave(p, 1).repeat_interleave(p, 2).repeat_interleave(p, 3)
        return csi, visible

    def _axis_features(self, csi, visible, axis, lag):
        if csi.shape[axis] <= lag:
            return csi.real.new_zeros(()), csi.real.new_zeros(())
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -lag)
        right_slice[axis] = slice(lag, None)
        left = csi[tuple(left_slice)]
        right = csi[tuple(right_slice)]
        pair_mask = visible[tuple(left_slice)] & visible[tuple(right_slice)]
        pair_ratio = pair_mask.float().mean()
        if not torch.any(pair_mask):
            return csi.real.new_zeros(()), pair_ratio
        left = left[pair_mask]
        right = right[pair_mask]
        numerator = torch.abs(torch.sum(left * torch.conj(right)))
        denominator = torch.sqrt(
            torch.sum(torch.abs(left) ** 2) * torch.sum(torch.abs(right) ** 2)
            + self.eps
        )
        return (numerator / denominator).clamp(0.0, 1.0), pair_ratio

    def _global_features(self, csi, visible):
        values = csi[visible]
        if values.numel() == 0:
            return csi.real.new_zeros(5)
        power = torch.mean(torch.abs(values) ** 2).clamp_min(self.eps)
        return torch.stack(
            (
                torch.log(power),
                values.real.mean(),
                values.imag.mean(),
                values.real.std(unbiased=False),
                values.imag.std(unbiased=False),
            )
        )

    def _batch_axis_features(self, csi, visible, axis, lag):
        batch_axis = axis + 1
        if csi.shape[batch_axis] <= lag:
            zeros = csi.real.new_zeros(csi.shape[0])
            return zeros, zeros
        left_slice = [slice(None)] * 4
        right_slice = [slice(None)] * 4
        left_slice[batch_axis] = slice(0, -lag)
        right_slice[batch_axis] = slice(lag, None)
        left = csi[tuple(left_slice)]
        right = csi[tuple(right_slice)]
        pair_mask = visible[tuple(left_slice)] & visible[tuple(right_slice)]
        reduce_dims = tuple(range(1, left.ndim))
        pair_float = pair_mask.to(left.real.dtype)
        pair_ratio = pair_float.mean(dim=reduce_dims)
        numerator = torch.abs(
            torch.sum(left * torch.conj(right) * pair_float, dim=reduce_dims)
        )
        left_power = torch.sum(torch.abs(left) ** 2 * pair_float, dim=reduce_dims)
        right_power = torch.sum(torch.abs(right) ** 2 * pair_float, dim=reduce_dims)
        denominator = torch.sqrt(left_power * right_power + self.eps)
        rho = torch.where(
            pair_mask.flatten(1).any(dim=1),
            (numerator / denominator).clamp(0.0, 1.0),
            torch.zeros_like(numerator),
        )
        return rho, pair_ratio

    def _batch_global_features(self, csi, visible):
        mask = visible.to(csi.real.dtype)
        reduce_dims = tuple(range(1, csi.ndim))
        count = mask.sum(dim=reduce_dims).clamp_min(1.0)
        mean_real = (csi.real * mask).sum(dim=reduce_dims) / count
        mean_imag = (csi.imag * mask).sum(dim=reduce_dims) / count
        second_real = (csi.real.square() * mask).sum(dim=reduce_dims) / count
        second_imag = (csi.imag.square() * mask).sum(dim=reduce_dims) / count
        std_real = (second_real - mean_real.square()).clamp_min(0.0).sqrt()
        std_imag = (second_imag - mean_imag.square()).clamp_min(0.0).sqrt()
        power = ((torch.abs(csi) ** 2) * mask).sum(dim=reduce_dims) / count
        return torch.stack(
            (torch.log(power.clamp_min(self.eps)), mean_real, mean_imag, std_real, std_imag),
            dim=-1,
        )

    def _batch_metadata_features(self, phys_meta, dims, reference):
        if phys_meta is None:
            raise ValueError("coherence_meta requires physical metadata")
        meta = phys_meta.to(reference)
        fc, delta_f, delta_t, ant_spacing = meta.unbind(dim=-1)
        t, k, u = dims
        constants = reference.new_tensor(
            [math.log2(t / 16.0), math.log2(k / 64.0), math.log2(u / 16.0)]
        ).unsqueeze(0).expand(meta.shape[0], -1)
        return torch.cat(
            (
                torch.log10((fc / 1e9).clamp_min(self.eps)).unsqueeze(-1),
                torch.log10((delta_f / 15e3).clamp_min(self.eps)).unsqueeze(-1),
                torch.log10((delta_t / 1e-3).clamp_min(self.eps)).unsqueeze(-1),
                (ant_spacing * fc / SPEED_OF_LIGHT).unsqueeze(-1),
                constants,
            ),
            dim=-1,
        )

    def _metadata_features(self, phys_meta, dims, reference):
        if phys_meta is None:
            raise ValueError("coherence_meta requires physical metadata")
        fc, delta_f, delta_t, ant_spacing = phys_meta.to(reference).unbind()
        t, k, u = dims
        return torch.stack(
            (
                torch.log10((fc / 1e9).clamp_min(self.eps)),
                torch.log10((delta_f / 15e3).clamp_min(self.eps)),
                torch.log10((delta_t / 1e-3).clamp_min(self.eps)),
                ant_spacing * fc / SPEED_OF_LIGHT,
                torch.log2(reference.new_tensor(float(t) / 16.0)),
                torch.log2(reference.new_tensor(float(k) / 64.0)),
                torch.log2(reference.new_tensor(float(u) / 16.0)),
            )
        )

    @torch.no_grad()
    def forward(self, patches, input_size, visible_tokens, phys_meta=None):
        sample_dims = [self._sample_dims(input_size, index) for index in range(patches.shape[0])]
        if sample_dims and all(dims == sample_dims[0] for dims in sample_dims):
            dims = sample_dims[0]
            csi, visible = self._unpatchify_batch(patches, dims, visible_tokens)
            correlations = []
            pair_ratios = []
            for axis in range(3):
                for lag in self.lags:
                    rho, pair_ratio = self._batch_axis_features(csi, visible, axis, lag)
                    correlations.append(rho)
                    pair_ratios.append(pair_ratio)
            needs_global = self.legacy_layout or bool(
                {"power", "raw_mean", "raw_std"}
                & set(self.selected_groups or ())
            )
            global_features = (
                self._batch_global_features(csi, visible)
                if needs_global
                else csi.real.new_zeros(csi.shape[0], 5)
            )
            parts = [
                torch.stack(correlations, dim=-1),
                torch.stack(pair_ratios, dim=-1),
                global_features,
            ]
            needs_metadata = self.include_metadata or (
                self.selected_groups is not None
                and bool({"physics_meta", "size_meta"} & set(self.selected_groups))
            )
            metadata = (
                self._batch_metadata_features(phys_meta, dims, csi.real)
                if needs_metadata
                else csi.real.new_zeros(csi.shape[0], 7)
            )
            return self._assemble_features(*parts, metadata).to(dtype=patches.real.dtype)

        descriptors = []
        for index, dims in enumerate(sample_dims):
            csi, visible = self._unpatchify(patches[index], dims, visible_tokens[index])
            correlations = []
            pair_ratios = []
            for axis in range(3):
                for lag in self.lags:
                    rho, pair_ratio = self._axis_features(csi, visible, axis, lag)
                    correlations.append(rho)
                    pair_ratios.append(pair_ratio)
            needs_global = self.legacy_layout or bool(
                {"power", "raw_mean", "raw_std"}
                & set(self.selected_groups or ())
            )
            global_features = (
                self._global_features(csi, visible)
                if needs_global
                else csi.real.new_zeros(5)
            )
            parts = [
                torch.stack(correlations),
                torch.stack(pair_ratios),
                global_features,
            ]
            needs_metadata = self.include_metadata or (
                self.selected_groups is not None
                and bool({"physics_meta", "size_meta"} & set(self.selected_groups))
            )
            if needs_metadata:
                sample_meta = None if phys_meta is None else phys_meta[index]
                metadata = self._metadata_features(sample_meta, dims, csi.real)
            else:
                metadata = csi.real.new_zeros(7)
            descriptors.append(self._assemble_features(*parts, metadata))
        return torch.stack(descriptors).to(dtype=patches.real.dtype)
