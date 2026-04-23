import math
from functools import partial

import torch
import torch.nn as nn

from util import video_vit
from util.logging import master_print as print
from util.pos_embed import UniversalPosEmbed
from util.rope import compute_mixed_cis_3d, init_t_3d


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
        rope_theta=10.0,
        use_ape=False,
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
        self.use_ape = use_ape
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
        encoder_attn_func = (
            video_vit.RoPEAttention3D if self.rope_enabled else video_vit.Attention
        )
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
        decoder_attn_func = (
            video_vit.RoPEAttention3D if self.rope_enabled else video_vit.Attention
        )
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 128, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()
        print(
            f"Initialized CSIModelMAE with rope_mode={self.rope_mode}, "
            f"encoder_pe={self.pos_emb_type}, decoder_pe={self.decoder_pos_emb_type}"
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

        if self.rope_mode == "learnable":
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
            self.enc_rope_controller = DynamicRoPEController(
                embed_dim, num_heads, enc_head_dim
            )
            self.dec_rope_controller = DynamicRoPEController(
                decoder_embed_dim, decoder_num_heads, dec_head_dim
            )
        else:
            raise ValueError(f"Unsupported rope_mode: {self.rope_mode}")

    @staticmethod
    def _build_fixed_freqs(head_dim, num_heads, thetas):
        freqs_per_dim = []
        for theta in thetas:
            freq = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            freqs_per_dim.append(freq)
        return torch.stack(freqs_per_dim).unsqueeze(1).repeat(1, num_heads, 1)

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
        return x_masked, mask, ids_restore, ids_keep, is_valid

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
        return x_masked, mask, ids_restore, ids_keep, is_valid

    @staticmethod
    def _compute_dynamic_cis(dynamic_freqs, t_t, t_f, t_a):
        angles_t = torch.einsum("n,bhd->bnhd", t_t, dynamic_freqs[:, 0])
        angles_f = torch.einsum("n,bhd->bnhd", t_f, dynamic_freqs[:, 1])
        angles_a = torch.einsum("n,bhd->bnhd", t_a, dynamic_freqs[:, 2])
        angles = (angles_t + angles_f + angles_a).float()
        return torch.polar(torch.ones_like(angles), angles)

    def _get_full_rope_frequencies(self, grid_size, x, mode="encoder"):
        if not self.rope_enabled:
            return None

        t_t, t_f, t_a = init_t_3d(*grid_size)
        t_t = t_t.to(x.device)
        t_f = t_f.to(x.device)
        t_a = t_a.to(x.device)

        if self.rope_mode in {"learnable", "fixed"}:
            freq_param = self.freqs if mode == "encoder" else self.dec_freqs
            return compute_mixed_cis_3d(freq_param, t_t, t_f, t_a)

        base_freqs = self.enc_base_freqs if mode == "encoder" else self.dec_base_freqs
        controller = (
            self.enc_rope_controller if mode == "encoder" else self.dec_rope_controller
        )
        dynamic_freqs = controller(x, base_freqs)
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
        patch_t, patch_f, patch_a = self.input_size[:3]
        max_t = self._resolve_grid_dim(input_size[0])
        max_f = self._resolve_grid_dim(input_size[1])
        max_a = self._resolve_grid_dim(input_size[2])
        return (max_t // patch_t, max_f // patch_f, max_a // patch_a)

    def _patch_embed_inputs(self, x):
        if "ComplexRotation" in self.pos_emb_type:
            h_real, h_imag = self.patch_embed(x)
            return torch.cat([h_real, h_imag], dim=-1)

        if torch.is_complex(x):
            x = torch.cat([x.real, x.imag], dim=-1)
        return self.patch_embed(x)

    def forward_encoder(
        self, x, token_length, input_size, mask_ratio, mask_strategy="random", grid_size=None
    ):
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
        else:
            raise ValueError(f"Unsupported mask_strategy: {mask_strategy}")

        if self.pos_embed is not None:
            x = self.pos_embed(x, grid_size, ids_keep=ids_keep)

        freqs_cis = self._get_full_rope_frequencies(grid_size, x, mode="encoder")
        if freqs_cis is not None:
            freqs_cis = self._gather_encoder_rope(freqs_cis, ids_keep)

        for block in self.blocks:
            if self.rope_enabled:
                x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
            else:
                x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        if self.cls_embed:
            x = x[:, 1:, :]
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, token_length, grid_size=None):
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
        attn_mask = ids < token_length.unsqueeze(1)
        if self.cls_embed:
            cls_mask = torch.ones(
                (batch_size, 1), dtype=attn_mask.dtype, device=attn_mask.device
            )
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)

        x = self.decoder_pos_embed(x, grid_size, ids_keep=None)
        if self.cls_embed:
            decoder_cls_tokens = self.decoder_cls_token.expand(batch_size, -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        freqs_cis = self._get_full_rope_frequencies(grid_size, x, mode="decoder")
        if freqs_cis is not None:
            freqs_cis = self._expand_decoder_rope(freqs_cis, batch_size, max_length)

        for block in self.decoder_blocks:
            if self.rope_enabled:
                x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
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
        del phys_meta
        grid_size = self._compute_grid_size(input_size)
        latent, mask, ids_restore = self.forward_encoder(
            imgs,
            token_length,
            input_size,
            mask_ratio,
            mask_strategy=mask_strategy,
            grid_size=grid_size,
        )
        pred = self.forward_decoder(latent, ids_restore, token_length, grid_size=grid_size)
        loss = self.forward_loss(imgs, pred, mask, token_length)
        return loss, pred, mask


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
