import math
from functools import partial

import torch
import torch.nn as nn

from util import video_vit
from util.logging import master_print as print
from util.pos_embed import UniversalPosEmbed
from util.rope import compute_mixed_cis_3d, init_t_3d
from models.coherence_controller import (
    CompactCoherenceRoPEController,
    CoherenceRoPEController,
    HybridCoherenceRoPEController,
    MaskAwareCoherenceDescriptor,
    parse_csv,
)


def _init_random_3d_freqs_with_thetas(dim: int, num_heads: int, thetas):
    if isinstance(thetas, (float, int)):
        thetas = (float(thetas), float(thetas), float(thetas))
    if len(thetas) != 3:
        raise ValueError(f"Expected three theta values, got {thetas}")

    freqs_list = []
    for theta in thetas:
        mag = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        per_head = []
        for _ in range(num_heads):
            angle = torch.rand(1) * 2 * torch.pi
            per_head.append(
                torch.cat(
                    [mag * torch.cos(angle), mag * torch.cos(torch.pi / 2 + angle)],
                    dim=-1,
                )
            )
        freqs_list.append(torch.stack(per_head, dim=0))
    return torch.stack(freqs_list, dim=0)


class DynamicRoPEController(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.half_head_dim = head_dim // 2
        self.out_dim = 3 * self.num_heads * self.half_head_dim

        self.mlp_s = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, self.out_dim),
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, self.out_dim),
        )

        nn.init.zeros_(self.mlp_s[-1].weight)
        nn.init.zeros_(self.mlp_s[-1].bias)
        nn.init.zeros_(self.mlp_b[-1].weight)
        nn.init.zeros_(self.mlp_b[-1].bias)

    def reset_output(self):
        nn.init.zeros_(self.mlp_s[-1].weight)
        nn.init.zeros_(self.mlp_s[-1].bias)
        nn.init.zeros_(self.mlp_b[-1].weight)
        nn.init.zeros_(self.mlp_b[-1].bias)

    def forward(self, x, base_freq_param):
        batch_size = x.shape[0]
        c_mean = x.mean(dim=1)
        c_std = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-6)
        context = torch.cat([c_mean, c_std], dim=-1)

        delta_s = self.mlp_s(context).view(
            batch_size, 3, self.num_heads, self.half_head_dim
        )
        delta_b = self.mlp_b(context).view(
            batch_size, 3, self.num_heads, self.half_head_dim
        )
        return base_freq_param.unsqueeze(0) * (1.0 + delta_s) + delta_b


class CSIModelMAE(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        depth=8,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        patch_embed=video_vit.CSIPatchEmbed_Complex,
        pos_emb_type="ComplexRotation",
        decoder_pos_emb_type="SinCos_3D",
        no_qkv_bias=False,
        trunc_init=False,
        cls_embed=False,
        rope_mode="adaptive",
        rope_axes="3d",
        attention_backbone="global",
        rope_theta=10.0,
        rope_frequency_scale=(1.0, 1.0, 1.0),
        use_ape=False,
        controller_mode="mean_std",
        controller_ablation="none",
        controller_max_scale=4.0,
        controller_feature_groups="rho,validity",
        controller_lags=(1, 2, 4),
        controller_validity_mode="axis_mean",
        controller_token_groups="",
        controller_scale_granularity="axis_head",
        controller_token_normalization="raw",
        controller_descriptor_hidden_dim=128,
        controller_token_hidden_dim=0,
        controller_raw_descriptor_groups=(),
        controller_context_mode="sample",
        controller_token_pool="channel",
        controller_decoder_token_scope="all_valid",
        controller_architecture="dual_branch",
        controller_fusion_hidden_dim=64,
        adaptive_scope="encoder_decoder",
        controller_boundary_regularization=0.0,
        controller_neutralize_groups="",
        controller_shuffle_groups="",
        controller_shuffle_offset=1,
        transfer_adapter_ratio=0.0,
        device=None,
        **kwargs,
    ):
        super().__init__()
        del device, kwargs
        self.trunc_init = trunc_init
        self.cls_embed = cls_embed
        self.pos_emb_type = pos_emb_type
        self.decoder_pos_emb_type = decoder_pos_emb_type
        self.rope_mode = rope_mode
        self.rope_axes = rope_axes
        if isinstance(rope_frequency_scale, (int, float)):
            rope_frequency_scale = (float(rope_frequency_scale),) * 3
        self.rope_frequency_scale = tuple(float(value) for value in rope_frequency_scale)
        if len(self.rope_frequency_scale) != 3 or any(
            value <= 0 for value in self.rope_frequency_scale
        ):
            raise ValueError("rope_frequency_scale must contain three positive values")
        self.attention_backbone = attention_backbone
        self.use_ape = use_ape
        self.controller_mode = controller_mode
        self.controller_ablation = controller_ablation
        self.controller_max_scale = float(controller_max_scale)
        self.controller_feature_groups = parse_csv(controller_feature_groups)
        self.controller_lags = tuple(int(value) for value in controller_lags)
        self.controller_validity_mode = str(controller_validity_mode)
        self.controller_token_groups = parse_csv(controller_token_groups)
        self.controller_scale_granularity = str(controller_scale_granularity)
        self.controller_token_normalization = str(controller_token_normalization)
        self.controller_descriptor_hidden_dim = int(controller_descriptor_hidden_dim)
        self.controller_token_hidden_dim = int(controller_token_hidden_dim)
        self.controller_raw_descriptor_groups = controller_raw_descriptor_groups
        self.controller_context_mode = str(controller_context_mode)
        self.controller_token_pool = str(controller_token_pool)
        self.controller_decoder_token_scope = str(controller_decoder_token_scope)
        if self.controller_decoder_token_scope not in {"all_valid", "visible_only"}:
            raise ValueError(
                "Unsupported controller_decoder_token_scope: "
                f"{self.controller_decoder_token_scope}"
            )
        self.controller_architecture = str(controller_architecture)
        if self.controller_architecture == "token_only":
            if self.controller_mode != "compact":
                raise ValueError("token_only architecture requires controller_mode=compact")
            if self.controller_feature_groups:
                raise ValueError(
                    "token_only architecture requires empty controller feature groups"
                )
            if not self.controller_token_groups:
                raise ValueError(
                    "token_only architecture requires token_mean and/or token_std"
                )
            if self.controller_decoder_token_scope != "visible_only":
                raise ValueError(
                    "token_only architecture requires visible_only decoder context"
                )
            if str(self.pos_emb_type).lower() != "none" or str(
                self.decoder_pos_emb_type
            ).lower() != "none":
                raise ValueError(
                    "token_only architecture is locked to pure RoPE without "
                    "encoder or decoder APE"
                )
        self.controller_fusion_hidden_dim = int(controller_fusion_hidden_dim)
        self.adaptive_scope = str(adaptive_scope)
        if self.adaptive_scope not in {
            "encoder_decoder",
            "encoder_only",
            "decoder_only",
        }:
            raise ValueError(f"Unsupported adaptive_scope: {self.adaptive_scope}")
        self.controller_boundary_regularization = float(
            controller_boundary_regularization
        )
        self.controller_neutralize_groups = parse_csv(controller_neutralize_groups)
        self.controller_shuffle_groups = parse_csv(controller_shuffle_groups)
        self.controller_shuffle_offset = int(controller_shuffle_offset)
        self.transfer_adapter_ratio = float(transfer_adapter_ratio)
        self._rope_coordinate_cache = {}
        # Diagnostic-only rotary scale controls.  These parameters are absent
        # from ordinary checkpoints and are installed explicitly by the
        # oracle-scale probe CLI.  External overrides are intentionally kept
        # outside the state dict because they are per-evaluation interventions.
        self.register_parameter("rope_oracle_logits_encoder", None)
        self.register_parameter("rope_oracle_logits_decoder", None)
        self._rope_oracle_axis_index = None
        self._rope_scale_overrides = {"encoder": None, "decoder": None}
        self._fast_inference = False
        self._inference_grid_size = None
        self._inference_coordinate_cache = None
        self.rope_enabled = rope_mode != "none"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.decoder_num_heads = decoder_num_heads
        self.rope_theta_tuple = self._normalize_rope_theta(rope_theta)

        if "ComplexRotation" in pos_emb_type:
            patch_input_size = (4, 4, 4, 1)
        else:
            patch_input_size = (4, 4, 4, 2)
            patch_embed = video_vit.PatchEmbed_v2

        self.patch_embed = patch_embed(
            input_dim=patch_input_size[0]
            * patch_input_size[1]
            * patch_input_size[2]
            * patch_input_size[3],
            output_dim=embed_dim,
        )
        self.input_size = patch_input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = None
        if not self.rope_enabled or self.use_ape:
            self.pos_embed = UniversalPosEmbed(embed_dim, pos_emb_type=pos_emb_type)

        self.decoder_pos_embed = UniversalPosEmbed(
            decoder_embed_dim, pos_emb_type=decoder_pos_emb_type
        )

        encoder_block_cls = (
            video_vit.Block_v2_RoPE if self.rope_enabled else video_vit.Block_v2
        )
        if self.rope_enabled:
            encoder_attn_func = (
                video_vit.AxialRoPEAttention3D
                if self.attention_backbone == "axial"
                else video_vit.RoPEAttention3D
            )
        else:
            encoder_attn_func = video_vit.Attention
        self.blocks = nn.ModuleList(
            [
                encoder_block_cls(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=encoder_attn_func,
                )
                for _ in range(depth)
            ]
        )

        decoder_block_cls = (
            video_vit.Block_v2_RoPE if self.rope_enabled else video_vit.Block_v2
        )
        if self.rope_enabled:
            decoder_attn_func = (
                video_vit.AxialRoPEAttention3D
                if self.attention_backbone == "axial"
                else video_vit.RoPEAttention3D
            )
        else:
            decoder_attn_func = video_vit.Attention
        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block_cls(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    attn_func=decoder_attn_func,
                )
                for _ in range(decoder_depth)
            ]
        )

        self._init_rope_parameters(
            embed_dim=embed_dim,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
        )

        self.norm = norm_layer(embed_dim)
        self.transfer_adapter = None
        if self.transfer_adapter_ratio > 0:
            adapter_hidden = max(4, int(round(embed_dim * self.transfer_adapter_ratio)))
            self.transfer_adapter = nn.Sequential(
                norm_layer(embed_dim),
                nn.Linear(embed_dim, adapter_hidden),
                nn.GELU(),
                nn.Linear(adapter_hidden, embed_dim),
            )
            nn.init.zeros_(self.transfer_adapter[-1].weight)
            nn.init.zeros_(self.transfer_adapter[-1].bias)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 128, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        if self.transfer_adapter is not None:
            nn.init.zeros_(self.transfer_adapter[-1].weight)
            nn.init.zeros_(self.transfer_adapter[-1].bias)
        self._reset_controller_outputs()
        print(
            f"Initialized CSIModelMAE with rope_mode={self.rope_mode}, "
            f"controller_mode={self.controller_mode}, encoder_pe={self.pos_emb_type}, "
            f"decoder_pe={self.decoder_pos_emb_type}"
        )

    @staticmethod
    def _normalize_rope_theta(rope_theta):
        if isinstance(rope_theta, (float, int)):
            return (float(rope_theta), float(rope_theta), float(rope_theta))
        if isinstance(rope_theta, str):
            parts = [p.strip() for p in rope_theta.split(",") if p.strip()]
            if len(parts) == 1:
                value = float(parts[0])
                return (value, value, value)
            if len(parts) == 3:
                return tuple(float(part) for part in parts)
        if isinstance(rope_theta, (tuple, list)) and len(rope_theta) == 3:
            return tuple(float(value) for value in rope_theta)
        raise ValueError(
            "rope_theta must be a float or a comma-separated triplet such as 10,100,1000"
        )

    def _init_rope_parameters(
        self, embed_dim, num_heads, decoder_embed_dim, decoder_num_heads
    ):
        if not self.rope_enabled:
            return

        enc_head_dim = embed_dim // num_heads
        dec_head_dim = decoder_embed_dim // decoder_num_heads
        if enc_head_dim % 2 != 0 or dec_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if self.rope_axes == "1d":
            if self.rope_mode != "fixed":
                raise ValueError("1D RoPE currently supports fixed mode only")
            self.register_buffer("freqs", self._build_fixed_freqs_1d(enc_head_dim, num_heads, self.rope_theta_tuple[0]))
            self.register_buffer("dec_freqs", self._build_fixed_freqs_1d(dec_head_dim, decoder_num_heads, self.rope_theta_tuple[0]))
        elif self.rope_mode == "learnable":
            self.freqs = nn.Parameter(
                _init_random_3d_freqs_with_thetas(
                    enc_head_dim, num_heads, self.rope_theta_tuple
                ),
                requires_grad=True,
            )
            self.dec_freqs = nn.Parameter(
                _init_random_3d_freqs_with_thetas(
                    dec_head_dim, decoder_num_heads, self.rope_theta_tuple
                ),
                requires_grad=True,
            )
        elif self.rope_mode == "fixed":
            self.register_buffer(
                "freqs",
                self._build_fixed_freqs(enc_head_dim, num_heads, self.rope_theta_tuple),
            )
            self.register_buffer(
                "dec_freqs",
                self._build_fixed_freqs(
                    dec_head_dim, decoder_num_heads, self.rope_theta_tuple
                ),
            )
        elif self.rope_mode == "adaptive":
            self.enc_base_freqs = nn.Parameter(
                _init_random_3d_freqs_with_thetas(
                    enc_head_dim, num_heads, self.rope_theta_tuple
                ),
                requires_grad=True,
            )
            self.dec_base_freqs = nn.Parameter(
                _init_random_3d_freqs_with_thetas(
                    dec_head_dim, decoder_num_heads, self.rope_theta_tuple
                ),
                requires_grad=True,
            )
            if self.controller_mode == "mean_std":
                self.coherence_descriptor = None
                self.enc_rope_controller = DynamicRoPEController(
                    embed_dim, num_heads, enc_head_dim
                )
                self.dec_rope_controller = DynamicRoPEController(
                    decoder_embed_dim, decoder_num_heads, dec_head_dim
                )
            elif self.controller_mode in {"coherence", "coherence_meta"}:
                self.coherence_descriptor = MaskAwareCoherenceDescriptor(
                    patch_size=self.input_size[0],
                    include_metadata=self.controller_mode == "coherence_meta",
                )
                descriptor_dim = self.coherence_descriptor.output_dim
                if self.controller_mode == "coherence_meta":
                    controller_class = HybridCoherenceRoPEController
                else:
                    controller_class = CoherenceRoPEController
                if controller_class is HybridCoherenceRoPEController:
                    self.enc_rope_controller = controller_class(
                        descriptor_dim,
                        embed_dim,
                        num_heads,
                        enc_head_dim,
                        max_scale=self.controller_max_scale,
                        feature_groups=self.coherence_descriptor.feature_groups,
                        neutralize_groups=self.controller_neutralize_groups,
                        shuffle_groups=self.controller_shuffle_groups,
                        shuffle_offset=self.controller_shuffle_offset,
                    )
                    self.dec_rope_controller = controller_class(
                        descriptor_dim,
                        decoder_embed_dim,
                        decoder_num_heads,
                        dec_head_dim,
                        max_scale=self.controller_max_scale,
                        feature_groups=self.coherence_descriptor.feature_groups,
                        neutralize_groups=self.controller_neutralize_groups,
                        shuffle_groups=self.controller_shuffle_groups,
                        shuffle_offset=self.controller_shuffle_offset,
                    )
                else:
                    self.enc_rope_controller = controller_class(
                        descriptor_dim,
                        num_heads,
                        max_scale=self.controller_max_scale,
                        feature_groups=self.coherence_descriptor.feature_groups,
                        neutralize_groups=self.controller_neutralize_groups,
                        shuffle_groups=self.controller_shuffle_groups,
                        shuffle_offset=self.controller_shuffle_offset,
                    )
                    self.dec_rope_controller = controller_class(
                        descriptor_dim,
                        decoder_num_heads,
                        max_scale=self.controller_max_scale,
                        feature_groups=self.coherence_descriptor.feature_groups,
                        neutralize_groups=self.controller_neutralize_groups,
                        shuffle_groups=self.controller_shuffle_groups,
                        shuffle_offset=self.controller_shuffle_offset,
                    )
            elif self.controller_mode == "compact":
                descriptor_groups = self.controller_feature_groups
                self.coherence_descriptor = None
                descriptor_dim = 0
                descriptor_feature_groups = ()
                if self.controller_architecture != "token_only":
                    include_metadata = bool(
                        {"physics_meta", "size_meta"} & set(descriptor_groups)
                    )
                    self.coherence_descriptor = MaskAwareCoherenceDescriptor(
                        patch_size=self.input_size[0],
                        lags=self.controller_lags,
                        include_metadata=include_metadata,
                        feature_groups=descriptor_groups,
                        validity_mode=self.controller_validity_mode,
                    )
                    descriptor_dim = self.coherence_descriptor.output_dim
                    descriptor_feature_groups = self.coherence_descriptor.feature_groups
                common = dict(
                    descriptor_dim=descriptor_dim,
                    feature_groups=descriptor_feature_groups,
                    token_groups=self.controller_token_groups,
                    scale_granularity=self.controller_scale_granularity,
                    hidden_dim=self.controller_descriptor_hidden_dim,
                    token_hidden_dim=self.controller_token_hidden_dim,
                    token_normalization=self.controller_token_normalization,
                    token_pool=self.controller_token_pool,
                    architecture=self.controller_architecture,
                    fusion_hidden_dim=self.controller_fusion_hidden_dim,
                    max_scale=self.controller_max_scale,
                    raw_descriptor_groups=self.controller_raw_descriptor_groups,
                    context_mode=self.controller_context_mode,
                    neutralize_groups=self.controller_neutralize_groups,
                    shuffle_groups=self.controller_shuffle_groups,
                    shuffle_offset=self.controller_shuffle_offset,
                )
                self.enc_rope_controller = None
                self.dec_rope_controller = None
                if self.adaptive_scope in {"encoder_decoder", "encoder_only"}:
                    self.enc_rope_controller = CompactCoherenceRoPEController(
                        token_dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=enc_head_dim,
                        **common,
                    )
                if self.adaptive_scope in {"encoder_decoder", "decoder_only"}:
                    self.dec_rope_controller = CompactCoherenceRoPEController(
                        token_dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        head_dim=dec_head_dim,
                        **common,
                    )
            else:
                raise ValueError(f"Unsupported controller_mode: {self.controller_mode}")
        else:
            raise ValueError(f"Unsupported rope_mode: {self.rope_mode}")

        if self.rope_axes == "3d":
            encoder_frequency = (
                self.enc_base_freqs if self.rope_mode == "adaptive" else self.freqs
            )
            decoder_frequency = (
                self.dec_base_freqs if self.rope_mode == "adaptive" else self.dec_freqs
            )
            self.register_buffer(
                "initial_encoder_frequency", encoder_frequency.detach().clone(), persistent=True
            )
            self.register_buffer(
                "initial_decoder_frequency", decoder_frequency.detach().clone(), persistent=True
            )

    def _reset_controller_outputs(self):
        if self.rope_mode != "adaptive":
            return
        for controller in (
            getattr(self, "enc_rope_controller", None),
            getattr(self, "dec_rope_controller", None),
        ):
            if controller is not None:
                controller.reset_output()

    @staticmethod
    def _build_fixed_freqs(head_dim, num_heads, thetas):
        freqs_per_dim = []
        for theta in thetas:
            freq = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            freqs_per_dim.append(freq)
        return torch.stack(freqs_per_dim).unsqueeze(1).repeat(1, num_heads, 1)

    @staticmethod
    def _build_fixed_freqs_1d(head_dim, num_heads, theta):
        freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        return freq.unsqueeze(0).repeat(num_heads, 1)

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_cls_token, std=0.02)

        if "ComplexRotation" not in self.pos_emb_type:
            weight = self.patch_embed.proj.weight.data
            if self.trunc_init:
                torch.nn.init.trunc_normal_(weight)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)
        else:
            w_real = self.patch_embed.proj.weight_real.weight.data
            w_imag = self.patch_embed.proj.weight_imag.weight.data
            if self.trunc_init:
                torch.nn.init.trunc_normal_(w_real, std=0.02)
                torch.nn.init.trunc_normal_(w_imag, std=0.02)
                torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(w_real.view([w_real.shape[0], -1]))
                torch.nn.init.xavier_uniform_(w_imag.view([w_imag.shape[0], -1]))
                torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(module.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def random_masking(self, x, mask_ratio, token_length):
        batch_size, seq_len, channels = x.shape
        noise = torch.rand(batch_size, seq_len, device=x.device)
        col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        pad_mask = col_indices >= token_length.unsqueeze(1)
        noise[pad_mask] = 1e9

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = (token_length * (1 - mask_ratio)).long()
        row_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        keep_mask_sorted = row_indices < len_keep.unsqueeze(1)

        max_len_keep = len_keep.max().item()
        ids_keep = ids_shuffle[:, :max_len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, channels)
        )
        valid_keep_mask = torch.arange(
            max_len_keep, device=x.device
        ).unsqueeze(0) < len_keep.unsqueeze(1)
        x_masked = x_masked * valid_keep_mask.unsqueeze(-1).type_as(x_masked)

        mask_sorted = (~keep_mask_sorted).float()
        mask = torch.gather(mask_sorted, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def temporal_masking(self, x, input_size, mask_ratio=0.5):
        batch_size, max_length, channels = x.shape
        device = x.device
        t, k, u = [value.to(device) for value in input_size]
        t_blocks = t // 4
        patches_per_t = (k // 4) * (u // 4)
        t_keep = (t_blocks * (1 - mask_ratio)).long()
        arange = torch.arange(max_length, device=device)
        time_idx = arange.unsqueeze(0) // patches_per_t.unsqueeze(1)
        keep_mask = time_idx < t_keep.unsqueeze(1)
        mask = (~keep_mask).float()

        filler = torch.full((batch_size, max_length), max_length, device=device)
        ids_filled = torch.where(keep_mask, arange.unsqueeze(0), filler)
        ids_sorted, sort_indices = torch.sort(ids_filled, dim=1)
        ids_restore = torch.argsort(sort_indices, dim=1)

        max_keep = (t_keep * patches_per_t).max().long()
        ids_keep = ids_sorted[:, :max_keep]
        is_valid = ids_keep < max_length
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))
        x_masked = torch.gather(
            x, dim=1, index=ids_safe.unsqueeze(-1).expand(-1, -1, channels)
        )
        x_masked = x_masked * is_valid.unsqueeze(-1).to(x_masked.dtype)
        # Padded samples can have fewer valid structured-mask tokens than the
        # largest sample in the batch.  Keep their placeholder coordinates in
        # range; ``is_valid`` remains the authoritative attention mask.
        return x_masked, mask, ids_restore, ids_safe, is_valid

    def freq_masking(self, x, input_size, mask_ratio=0.5):
        batch_size, max_length, channels = x.shape
        device = x.device
        t, k, u = [value.to(device) for value in input_size]
        t_blocks, k_blocks, u_blocks = (
            (t // 4).long(),
            (k // 4).long(),
            (u // 4).long(),
        )
        global_idx = torch.arange(max_length, device=device).unsqueeze(0)
        k_idx = (global_idx // u_blocks.unsqueeze(1)) % k_blocks.unsqueeze(1)
        k_keep = torch.maximum(
            (k_blocks * (1 - mask_ratio)).round().long(), torch.ones_like(k_blocks)
        )
        valid_mask = global_idx < (t_blocks * k_blocks * u_blocks).unsqueeze(1)
        keep_mask = (k_idx < k_keep.unsqueeze(1)) & valid_mask
        mask = (~keep_mask).float()

        filler = torch.full((batch_size, max_length), max_length, device=device)
        ids_filled = torch.where(keep_mask, global_idx, filler)
        ids_sorted, sort_indices = torch.sort(ids_filled, dim=1)
        ids_restore = torch.argsort(sort_indices, dim=1)

        max_keep = (k_keep * t_blocks * u_blocks).max().long()
        ids_keep = ids_sorted[:, :max_keep]
        is_valid = ids_keep < max_length
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))
        x_masked = torch.gather(
            x, dim=1, index=ids_safe.unsqueeze(-1).expand(-1, -1, channels)
        )
        x_masked = x_masked * is_valid.unsqueeze(-1).to(x_masked.dtype)
        # See ``temporal_masking``: invalid padded slots must not carry the
        # out-of-range ``max_length`` sentinel into positional gathers.
        return x_masked, mask, ids_restore, ids_safe, is_valid

    def antenna_masking(self, x, input_size, mask_ratio=0.5):
        """Keep a leading contiguous antenna block and reconstruct the remainder."""
        batch_size, max_length, channels = x.shape
        device = x.device
        t, k, u = [value.to(device) for value in input_size]
        t_blocks, k_blocks, u_blocks = (
            (t // 4).long(),
            (k // 4).long(),
            (u // 4).long(),
        )
        global_idx = torch.arange(max_length, device=device).unsqueeze(0)
        u_idx = global_idx % u_blocks.unsqueeze(1)
        u_keep = torch.maximum(
            (u_blocks * (1 - mask_ratio)).round().long(), torch.ones_like(u_blocks)
        )
        valid_mask = global_idx < (t_blocks * k_blocks * u_blocks).unsqueeze(1)
        keep_mask = (u_idx < u_keep.unsqueeze(1)) & valid_mask
        mask = (~keep_mask).float()

        filler = torch.full((batch_size, max_length), max_length, device=device)
        ids_filled = torch.where(keep_mask, global_idx, filler)
        ids_sorted, sort_indices = torch.sort(ids_filled, dim=1)
        ids_restore = torch.argsort(sort_indices, dim=1)
        max_keep = (t_blocks * k_blocks * u_keep).max().long()
        ids_keep = ids_sorted[:, :max_keep]
        is_valid = ids_keep < max_length
        ids_safe = torch.where(is_valid, ids_keep, torch.zeros_like(ids_keep))
        x_masked = torch.gather(
            x, dim=1, index=ids_safe.unsqueeze(-1).expand(-1, -1, channels)
        )
        x_masked = x_masked * is_valid.unsqueeze(-1).to(x_masked.dtype)
        return x_masked, mask, ids_restore, ids_safe, is_valid

    @staticmethod
    def _compute_dynamic_cis(dynamic_freqs, t_t, t_f, t_a):
        angles_t = torch.einsum("n,bhd->bnhd", t_t, dynamic_freqs[:, 0])
        angles_f = torch.einsum("n,bhd->bnhd", t_f, dynamic_freqs[:, 1])
        angles_a = torch.einsum("n,bhd->bnhd", t_a, dynamic_freqs[:, 2])
        angles = (angles_t + angles_f + angles_a).float()
        return torch.polar(torch.ones_like(angles), angles)

    def configure_rope_oracle_scale(self, axis: str):
        """Install the 28-scalar Base-model oracle modulation probe.

        The probe shares the final Controller's axis-head granularity while
        leaving the underlying frequency bank and every content parameter
        untouched.  It is a mechanism diagnostic, not part of the method.
        """
        if not self.rope_enabled or self.rope_axes != "3d":
            raise ValueError("Oracle rotary scales require three-axis RoPE")
        if axis not in {"t", "k", "u"}:
            raise ValueError(f"Unsupported oracle axis: {axis}")
        self._rope_oracle_axis_index = {"t": 0, "k": 1, "u": 2}[axis]
        enc_reference = (
            self.enc_base_freqs if self.rope_mode == "adaptive" else self.freqs
        )
        dec_reference = (
            self.dec_base_freqs if self.rope_mode == "adaptive" else self.dec_freqs
        )
        if self.rope_oracle_logits_encoder is None:
            self.rope_oracle_logits_encoder = nn.Parameter(
                enc_reference.new_zeros(enc_reference.shape[1])
            )
        if self.rope_oracle_logits_decoder is None:
            self.rope_oracle_logits_decoder = nn.Parameter(
                dec_reference.new_zeros(dec_reference.shape[1])
            )

    def set_rope_scale_override(self, encoder=None, decoder=None):
        """Override Controller scales for a counterfactual evaluation pass.

        Each value must be ``[3,H]`` or ``[B,3,H]`` and contains *scales*, not
        logits.  The encoder and decoder are kept separate because they expose
        different numbers of attention heads.
        """
        self._rope_scale_overrides["encoder"] = encoder
        self._rope_scale_overrides["decoder"] = decoder

    def clear_rope_scale_override(self):
        self._rope_scale_overrides["encoder"] = None
        self._rope_scale_overrides["decoder"] = None

    def get_last_controller_scales(self):
        result = {}
        for mode, name in (
            ("encoder", "enc_rope_controller"),
            ("decoder", "dec_rope_controller"),
        ):
            controller = getattr(self, name, None)
            scale = getattr(controller, "last_scale", None)
            if scale is not None:
                result[mode] = scale.detach()
        return result

    def _diagnostic_dynamic_freqs(self, base_freqs, mode: str, batch_size: int):
        """Return overridden/probe frequencies, or ``None`` for normal flow."""
        override = self._rope_scale_overrides.get(mode)
        if override is not None:
            scale = torch.as_tensor(
                override, device=base_freqs.device, dtype=base_freqs.dtype
            )
            expected_heads = base_freqs.shape[1]
            if scale.ndim == 2:
                if tuple(scale.shape) != (3, expected_heads):
                    raise ValueError(
                        f"{mode} scale override must be [3,{expected_heads}], "
                        f"got {tuple(scale.shape)}"
                    )
                scale = scale.unsqueeze(0).expand(batch_size, -1, -1)
            elif scale.ndim == 3:
                if tuple(scale.shape[1:]) != (3, expected_heads):
                    raise ValueError(
                        f"{mode} scale override must be [B,3,{expected_heads}], "
                        f"got {tuple(scale.shape)}"
                    )
                if scale.shape[0] != batch_size:
                    raise ValueError(
                        f"{mode} override batch {scale.shape[0]} != {batch_size}"
                    )
            else:
                raise ValueError(f"Unsupported {mode} override rank: {scale.ndim}")
            if not torch.isfinite(scale).all() or torch.any(scale <= 0):
                raise ValueError(f"{mode} scale override must be finite and positive")
            return base_freqs.float().unsqueeze(0) * scale.float().unsqueeze(-1)

        logits = (
            self.rope_oracle_logits_encoder
            if mode == "encoder"
            else self.rope_oracle_logits_decoder
        )
        if logits is None:
            return None
        axis_index = self._rope_oracle_axis_index
        if axis_index is None:
            raise RuntimeError("Oracle scale parameters are installed without an axis")
        axis_scale = torch.exp(
            math.log(self.controller_max_scale) * torch.tanh(logits.float())
        )
        axis_selector = torch.nn.functional.one_hot(
            torch.tensor(axis_index, device=base_freqs.device), num_classes=3
        ).to(base_freqs.dtype)
        scale = 1.0 + axis_selector[:, None] * (axis_scale[None, :] - 1.0)
        scale = scale.unsqueeze(0).expand(batch_size, -1, -1)
        return base_freqs.float().unsqueeze(0) * scale.float().unsqueeze(-1)

    def _get_rope_coordinates(self, grid_size, device):
        if self._fast_inference and self._inference_grid_size == tuple(grid_size):
            if self._inference_coordinate_cache is None:
                self._inference_coordinate_cache = tuple(
                    coordinate.to(device=device) for coordinate in init_t_3d(*grid_size)
                )
            return self._inference_coordinate_cache
        device_index = -1 if device.index is None else int(device.index)
        key = (tuple(int(value) for value in grid_size), device.type, device_index)
        if key not in self._rope_coordinate_cache:
            self._rope_coordinate_cache[key] = tuple(
                coordinate.to(device=device) for coordinate in init_t_3d(*grid_size)
            )
        return self._rope_coordinate_cache[key]

    def set_fast_inference(self, enabled=True):
        """Enable output-equivalent inference mode with no diagnostic retention."""
        self._fast_inference = bool(enabled)
        for controller in (
            getattr(self, "enc_rope_controller", None),
            getattr(self, "dec_rope_controller", None),
        ):
            if controller is not None and hasattr(controller, "set_fast_inference"):
                controller.set_fast_inference(enabled)
        return self

    def set_inference_grid_size(self, grid_size):
        """Set an exact-shape deployment hint outside the compiled forward path."""
        grid_size = tuple(int(value) for value in grid_size)
        if len(grid_size) != 3 or any(value <= 0 for value in grid_size):
            raise ValueError("inference grid size must be three positive dimensions")
        self._inference_grid_size = grid_size
        self._inference_coordinate_cache = None
        return self

    def clear_inference_grid_size(self):
        self._inference_grid_size = None
        self._inference_coordinate_cache = None
        return self

    def _get_full_rope_frequencies(
        self, grid_size, x, mode="encoder", descriptor=None, token_mask=None
    ):
        if not self.rope_enabled:
            return None

        t_t, t_f, t_a = self._get_rope_coordinates(grid_size, x.device)

        if self.rope_axes == "1d":
            freq_param = self.freqs if mode == "encoder" else self.dec_freqs
            positions = torch.arange(t_t.numel(), device=x.device, dtype=freq_param.dtype)
            angles = positions[:, None, None] * freq_param[None, :, :]
            return torch.polar(torch.ones_like(angles), angles)

        if self.rope_mode in {"learnable", "fixed"}:
            freq_param = self.freqs if mode == "encoder" else self.dec_freqs
            axis_scale = freq_param.new_tensor(self.rope_frequency_scale).view(3, 1, 1)
            freq_param = freq_param * axis_scale
            diagnostic_freqs = self._diagnostic_dynamic_freqs(
                freq_param, mode, x.shape[0]
            )
            if diagnostic_freqs is not None:
                return self._compute_dynamic_cis(
                    diagnostic_freqs, t_t, t_f, t_a
                )
            return compute_mixed_cis_3d(freq_param, t_t, t_f, t_a)

        base_freqs = self.enc_base_freqs if mode == "encoder" else self.dec_base_freqs
        axis_scale = base_freqs.new_tensor(self.rope_frequency_scale).view(3, 1, 1)
        base_freqs = base_freqs * axis_scale
        diagnostic_freqs = self._diagnostic_dynamic_freqs(
            base_freqs, mode, x.shape[0]
        )
        if diagnostic_freqs is not None:
            return self._compute_dynamic_cis(diagnostic_freqs, t_t, t_f, t_a)
        controller = (
            self.enc_rope_controller if mode == "encoder" else self.dec_rope_controller
        )
        if controller is None:
            return compute_mixed_cis_3d(base_freqs, t_t, t_f, t_a)
        if self.controller_mode == "mean_std":
            dynamic_freqs = controller(x, base_freqs)
        else:
            if descriptor is None and not (
                self.controller_mode == "compact"
                and self.controller_architecture == "token_only"
            ):
                raise ValueError("Coherence controller requires a CSI descriptor")
            if self.controller_mode == "compact":
                dynamic_freqs = controller(
                    descriptor,
                    x,
                    base_freqs,
                    ablation=self.controller_ablation,
                    token_mask=token_mask,
                )
            elif self.controller_mode == "coherence_meta":
                dynamic_freqs = controller(
                    descriptor,
                    x,
                    base_freqs,
                    ablation=self.controller_ablation,
                )
            else:
                dynamic_freqs = controller(
                    descriptor, base_freqs, ablation=self.controller_ablation
                )
        return self._compute_dynamic_cis(dynamic_freqs, t_t, t_f, t_a)

    @staticmethod
    def _expand_decoder_rope(freqs_cis, batch_size, max_length):
        if freqs_cis.dim() == 3:
            return freqs_cis[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        return freqs_cis[:, :max_length, :, :]

    @staticmethod
    def _gather_encoder_rope(freqs_cis, ids_keep):
        batch_size, keep_len = ids_keep.shape
        if freqs_cis.dim() == 3:
            heads, half_dim = freqs_cis.shape[1], freqs_cis.shape[2]
            expanded = freqs_cis.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            heads, half_dim = freqs_cis.shape[2], freqs_cis.shape[3]
            expanded = freqs_cis
        gather_indices = ids_keep.view(batch_size, keep_len, 1, 1).expand(
            -1, -1, heads, half_dim
        )
        return torch.gather(expanded, 1, gather_indices)

    @staticmethod
    def _resolve_grid_dim(input_size_value):
        if torch.is_tensor(input_size_value):
            return int(input_size_value.max().item())
        if isinstance(input_size_value, (tuple, list)):
            values = []
            for value in input_size_value:
                if torch.is_tensor(value):
                    values.append(int(value.item()))
                else:
                    values.append(int(value))
            return max(values)
        return int(input_size_value)

    def _compute_grid_size(self, input_size):
        if self._fast_inference and self._inference_grid_size is not None:
            return self._inference_grid_size
        patch_t, patch_f, patch_a = self.input_size[:3]
        max_t = self._resolve_grid_dim(input_size[0])
        max_f = self._resolve_grid_dim(input_size[1])
        max_a = self._resolve_grid_dim(input_size[2])
        return (max_t // patch_t, max_f // patch_f, max_a // patch_a)

    def _full_grid_coordinates(self, grid_size, device):
        coords = self._get_rope_coordinates(grid_size, device)
        return torch.stack(coords, dim=-1).to(dtype=torch.long)

    def _patch_embed_inputs(self, x):
        if "ComplexRotation" in self.pos_emb_type:
            h_real, h_imag = self.patch_embed(x)
            return torch.cat([h_real, h_imag], dim=-1)

        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=-1)
        return self.patch_embed(x)

    def forward_encoder(
        self,
        x,
        token_length,
        input_size,
        mask_ratio,
        mask_strategy="random",
        grid_size=None,
        phys_meta=None,
    ):
        raw_patches = x
        x = self._patch_embed_inputs(x)
        batch_size, seq_len, _ = x.shape

        if mask_strategy == "random":
            x, mask, ids_restore, ids_keep = self.random_masking(
                x, mask_ratio, token_length
            )
            ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
                batch_size, seq_len
            )
            pad_mask_full = ids < token_length.unsqueeze(1)
            attn_mask = torch.gather(pad_mask_full, dim=1, index=ids_keep)
        elif mask_strategy == "temporal":
            x, mask, ids_restore, ids_keep, attn_mask = self.temporal_masking(
                x, input_size, mask_ratio
            )
        elif mask_strategy == "freq":
            x, mask, ids_restore, ids_keep, attn_mask = self.freq_masking(
                x, input_size, mask_ratio
            )
        elif mask_strategy == "antenna":
            x, mask, ids_restore, ids_keep, attn_mask = self.antenna_masking(
                x, input_size, mask_ratio
            )
        else:
            raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")

        if self.pos_embed is not None:
            x = self.pos_embed(x, grid_size, ids_keep=ids_keep)

        descriptor = None
        if (
            self.rope_mode == "adaptive"
            and self.controller_mode != "mean_std"
            and self.coherence_descriptor is not None
        ):
            token_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            valid_tokens = token_ids < token_length.unsqueeze(1)
            visible_tokens = (~mask.bool()) & valid_tokens
            descriptor = self.coherence_descriptor(
                raw_patches, input_size, visible_tokens, phys_meta=phys_meta
            )

        freqs_cis = self._get_full_rope_frequencies(
            grid_size,
            x,
            mode="encoder",
            descriptor=descriptor,
            token_mask=attn_mask,
        )
        if freqs_cis is not None:
            freqs_cis = self._gather_encoder_rope(freqs_cis, ids_keep)

        axial_coords = None
        if self.attention_backbone == "axial":
            full_coords = self._full_grid_coordinates(grid_size, x.device)
            expanded_coords = full_coords.unsqueeze(0).expand(batch_size, -1, -1)
            axial_coords = torch.gather(
                expanded_coords, 1, ids_keep.unsqueeze(-1).expand(-1, -1, 3)
            )

        for block in self.blocks:
            if self.rope_enabled:
                x = block(
                    x,
                    freqs_cis=freqs_cis,
                    attn_mask=attn_mask,
                    axial_coords=axial_coords,
                )
            else:
                x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        if self.transfer_adapter is not None:
            x = x + self.transfer_adapter(x)
        if self.cls_embed:
            x = x[:, 1:, :]
        return x, mask, ids_restore, descriptor

    def forward_decoder(
        self,
        x,
        ids_restore,
        token_length,
        grid_size=None,
        descriptor=None,
        reconstruction_mask=None,
    ):
        batch_size, _, _ = x.shape
        x = self.decoder_embed(x)
        channels = x.shape[-1]
        max_length = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(batch_size, max_length - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]),
        )
        x = x_.view(batch_size, max_length, channels)

        ids = torch.arange(max_length, device=x.device).unsqueeze(0).expand(
            batch_size, max_length
        )
        valid_token_mask = ids < token_length.unsqueeze(1)
        controller_token_mask = valid_token_mask
        if self.controller_decoder_token_scope == "visible_only":
            if reconstruction_mask is None:
                raise ValueError(
                    "visible_only decoder token statistics require reconstruction_mask"
                )
            if reconstruction_mask.shape != valid_token_mask.shape:
                raise ValueError(
                    "reconstruction_mask must match the restored decoder token grid: "
                    f"expected {tuple(valid_token_mask.shape)}, got "
                    f"{tuple(reconstruction_mask.shape)}"
                )
            controller_token_mask = (~reconstruction_mask.bool()) & valid_token_mask
        attn_mask = valid_token_mask
        if self.cls_embed:
            cls_mask = torch.ones(
                (batch_size, 1), dtype=attn_mask.dtype, device=attn_mask.device
            )
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)
            controller_token_mask = torch.cat(
                [cls_mask, controller_token_mask], dim=1
            )

        x = self.decoder_pos_embed(x, grid_size, ids_keep=None)
        if self.cls_embed:
            decoder_cls_tokens = self.decoder_cls_token.expand(batch_size, -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        freqs_cis = self._get_full_rope_frequencies(
            grid_size,
            x,
            mode="decoder",
            descriptor=descriptor,
            token_mask=controller_token_mask,
        )
        if freqs_cis is not None:
            freqs_cis = self._expand_decoder_rope(freqs_cis, batch_size, max_length)

        axial_coords = None
        if self.attention_backbone == "axial":
            full_coords = self._full_grid_coordinates(grid_size, x.device)[:max_length]
            axial_coords = full_coords.unsqueeze(0).expand(batch_size, -1, -1)

        for block in self.decoder_blocks:
            if self.rope_enabled:
                x = block(
                    x,
                    freqs_cis=freqs_cis,
                    attn_mask=attn_mask,
                    axial_coords=axial_coords,
                )
            else:
                x = block(x, attn_mask=attn_mask)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if self.cls_embed:
            x = x[:, 1:, :]
        return x

    @staticmethod
    def forward_loss(imgs, pred, mask, token_length):
        target = torch.cat([imgs.real, imgs.imag], dim=-1)
        batch_size, seq_len, _ = imgs.shape
        col_indices = torch.arange(seq_len, device=imgs.device).expand(batch_size, seq_len)
        mask_in_length = col_indices < token_length[:, None]
        mask_nmse = mask.bool() & mask_in_length
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask_nmse = mask_nmse.view(loss.shape)
        denom = mask_nmse.sum()
        if denom > 0:
            return (loss * mask_nmse).sum() / denom
        return loss.sum() * 0.0

    def forward(
        self,
        imgs,
        token_length,
        input_size=None,
        mask_ratio=0.5,
        mask_strategy="freq",
        phys_meta=None,
    ):
        grid_size = self._compute_grid_size(input_size)
        latent, mask, ids_restore, descriptor = self.forward_encoder(
            imgs,
            token_length,
            input_size,
            mask_ratio,
            mask_strategy=mask_strategy,
            grid_size=grid_size,
            phys_meta=phys_meta,
        )
        pred = self.forward_decoder(
            latent,
            ids_restore,
            token_length,
            grid_size=grid_size,
            descriptor=descriptor,
            reconstruction_mask=mask,
        )
        loss = self.forward_loss(imgs, pred, mask, token_length)
        if (
            self.controller_boundary_regularization > 0.0
            and self.rope_mode == "adaptive"
            and self.controller_mode == "compact"
        ):
            penalties = []
            for controller in (
                getattr(self, "enc_rope_controller", None),
                getattr(self, "dec_rope_controller", None),
            ):
                scale = getattr(controller, "last_scale_live", None)
                if scale is not None:
                    normalized = torch.abs(torch.log(scale.clamp_min(1e-8))) / math.log(
                        self.controller_max_scale
                    )
                    penalties.append(torch.relu(normalized - 0.9).square().mean())
            if penalties:
                loss = loss + self.controller_boundary_regularization * torch.stack(
                    penalties
                ).mean()
        return loss, pred, mask

    def get_last_controller_diagnostics(self):
        if self.rope_mode != "adaptive" or self.controller_mode == "mean_std":
            return None
        controller = getattr(self, "enc_rope_controller", None)
        if controller is None:
            controller = getattr(self, "dec_rope_controller", None)
        descriptor = getattr(controller, "last_descriptor", None)
        token_context = getattr(controller, "last_token_context", None)
        scale = getattr(controller, "last_scale", None)
        if scale is None or (descriptor is None and token_context is None):
            return None
        feature_names = ()
        if self.coherence_descriptor is not None:
            feature_names = self.coherence_descriptor.feature_names
        return {
            "feature_names": feature_names,
            "descriptor": descriptor,
            "token_groups": self.controller_token_groups,
            "token_context": token_context,
            "scale": scale,
        }

    def controller_standardizers(self):
        """Return the unique descriptor standardizers used by this model."""
        modules = []
        for controller_name in ("enc_rope_controller", "dec_rope_controller"):
            controller = getattr(self, controller_name, None)
            for attribute in ("standardizer", "token_standardizer"):
                standardizer = getattr(controller, attribute, None)
                if standardizer is not None and all(
                    standardizer is not item for item in modules
                ):
                    modules.append(standardizer)
        return modules

    def begin_controller_calibration(self):
        for standardizer in self.controller_standardizers():
            standardizer.begin_calibration()

    def finalize_controller_calibration(self):
        for standardizer in self.controller_standardizers():
            standardizer.finalize_calibration()


def csi_mae_base(**kwargs):
    return CSIModelMAE(
        embed_dim=768,
        num_heads=12,
        depth=8,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def csi_mae_base_depth_student(**kwargs):
    """Depth-only deployment student preserving all feature interfaces."""
    return CSIModelMAE(
        embed_dim=768,
        num_heads=12,
        depth=6,
        decoder_embed_dim=512,
        decoder_depth=3,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def csi_mae_small(**kwargs):
    return CSIModelMAE(
        embed_dim=512,
        num_heads=8,
        depth=6,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=2.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def csi_mae_wifo_base(**kwargs):
    """Architecture used by the reference WiFo Sionna checkpoint.

    This constructor is intentionally separate from ``csi_mae_small``:
    the public WiFo checkpoint uses an MLP ratio of 4.0 rather than 2.0.
    """
    return CSIModelMAE(
        embed_dim=512,
        num_heads=8,
        depth=6,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def csi_mae_tiny(**kwargs):
    return CSIModelMAE(
        embed_dim=192,
        num_heads=8,
        depth=2,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=2.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
