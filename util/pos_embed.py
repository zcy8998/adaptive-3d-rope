import torch
import torch.nn as nn
import numpy as np
from util.embed import (
    ComplexFactorizedPosEmbedding,
    Complex3DPosEmbedding_V2,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid_with_resolution
)

class UniversalPosEmbed(nn.Module):
    """
    统一位置编码模块，支持 PhysCSI 的多种模式。
    Modes:
      - 'ComplexRotation': 复数旋转 (核心推荐) - 包含 input->complex->rotate->concat->fusion
      - 'trivial': 传统的各种可学习参数相加
      - 'SinCos': 2D+1D 正弦编码
      - 'SinCos_3D': 3D 正弦编码
    """
    def __init__(self, 
                 embed_dim, 
                 pos_emb_type='trivial', 
                 max_t_len=4,    # 最大时间Patch数
                 max_freq_len=4, # 最大频率Patch数
                 max_ant_len=4):  # 最大天线Patch数
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_emb_type = pos_emb_type
        self._sincos_cache = {}
        
        if self.pos_emb_type == 'ComplexRotation':
            self.complex_enc = ComplexFactorizedPosEmbedding(embed_dim // 2)
            self.fusion = nn.Linear(embed_dim, embed_dim)
            self._init_fusion()
        
        elif self.pos_emb_type == 'ComplexRotation_3D':
            self.complex_enc = Complex3DPosEmbedding_V2(embed_dim // 2)
            self.fusion = nn.Linear(embed_dim, embed_dim)
            self._init_fusion()

        elif self.pos_emb_type == 'trivial':
            self.emb_temporal = nn.Parameter(torch.zeros(1, max_t_len, embed_dim))
            self.emb_freq = nn.Parameter(torch.zeros(1, max_freq_len, embed_dim))
            self.emb_ant = nn.Parameter(torch.zeros(1, max_ant_len, embed_dim))
            self._init_trivial()

        elif self.pos_emb_type in ['SinCos', 'SinCos_2D', 'SinCos_3D', 'SinCos_1D_3', 'None']:
            pass # 运行时动态计算

    def _init_fusion(self):
        torch.nn.init.xavier_uniform_(self.fusion.weight)
        torch.nn.init.constant_(self.fusion.bias, 0)

    def _init_trivial(self):
        torch.nn.init.trunc_normal_(self.emb_temporal, std=0.02)
        torch.nn.init.trunc_normal_(self.emb_freq, std=0.02)
        torch.nn.init.trunc_normal_(self.emb_ant, std=0.02)

    def _normalize_dim(self, value):
        if torch.is_tensor(value):
            return int(value.detach().item())
        return int(value)

    def _normalize_scale(self, scale, ndim):
        if scale is None:
            return tuple([1.0] * ndim)
        if torch.is_tensor(scale):
            scale = scale.detach().cpu().flatten().tolist()
        return tuple(float(v) for v in scale)

    def _cache_lookup(self, key, builder):
        cached = self._sincos_cache.get(key)
        if cached is None:
            cached = builder()
            self._sincos_cache[key] = cached
        return cached

    def forward(self, x, input_size, ids_keep=None, scale=None):
        if self.pos_emb_type == 'None':
            return x

        if self.pos_emb_type == 'ComplexRotation':
            return self._forward_complex(x, input_size, ids_keep)
        if self.pos_emb_type == 'ComplexRotation_3D':
            return self._forward_complex_3d(x, input_size, ids_keep)
        elif self.pos_emb_type == 'trivial':
            return self._forward_trivial(x, input_size, ids_keep)
        elif self.pos_emb_type == 'SinCos':
            return self._forward_sincos_1d(x, input_size, ids_keep)
        elif self.pos_emb_type == 'SinCos_2D':
            return self._forward_sincos_2d(x, input_size, ids_keep)
        elif self.pos_emb_type == 'SinCos_3D':
            return self._forward_sincos_3d(x, input_size, ids_keep, scale)
        elif self.pos_emb_type == 'SinCos_1D_3':
            return self._forward_sincos_1d_3(x, input_size, ids_keep, scale)
        return x

    def _forward_complex(self, x, input_size, ids_keep):
        # 1. 旋转 (x 变为复数)
        # 如果输入是实数，ComplexFactorizedPosEmbedding 内部会将其视为模长或实部
        x_complex = self.complex_enc(x, input_size, ids_keep=ids_keep)
        
        # 2. 拼接实虚部 [B, N, 2D]
        x_cat = torch.cat([x_complex.real, x_complex.imag], dim=-1)
        
        # 3. 融合投影 [B, N, D]
        return self.fusion(x_cat)

    def _forward_complex_3d(self, x, input_size, ids_keep):
        # 1. 旋转 (x 变为复数)
        # 如果输入是实数，ComplexFactorizedPosEmbedding 内部会将其视为模长或实部
        x_complex = self.complex_enc(x, input_size, ids_keep=ids_keep)
        
        # 2. 拼接实虚部 [B, N, 2D]
        x_cat = torch.cat([x_complex.real, x_complex.imag], dim=-1)
        
        # 3. 融合投影 [B, N, D]
        return self.fusion(x_cat)

    def _forward_trivial(self, x, input_size, ids_keep):
        T, H, W = input_size
        B = x.shape[0]
        
        # 1. 构造 Spatial (Freq + Ant) -> [1, H*W, D]
        # 注意：这里假设 H 对应 Antenna, W 对应 Freq (根据原代码逻辑推断)
        # 切片以适应当前输入尺寸 (主要是 Decoder 阶段或不同数据集)
        curr_ant = self.emb_ant[:, :H] 
        curr_freq = self.emb_freq[:, :W]
        
        # Broadcast sum: [1, H, 1, D] + [1, 1, W, D] -> [1, H, W, D] -> Flatten
        pos_spatial = curr_ant.unsqueeze(2) + curr_freq.unsqueeze(1)
        pos_spatial = pos_spatial.flatten(1, 2) 

        # 2. 构造 Temporal
        curr_temp = self.emb_temporal[:, :T]
        
        # 3. Total: [1, T, 1, D] + [1, 1, HW, D] -> [1, T, HW, D] -> Flatten
        pos_total = curr_temp.unsqueeze(2) + pos_spatial.unsqueeze(1)
        pos_total = pos_total.flatten(1, 2) # [1, L_total, D]

        # 4. Apply Mask if needed
        if ids_keep is not None:
            pos_total = pos_total.expand(B, -1, -1)
            pos_total = torch.gather(pos_total, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else:
            pos_total = pos_total.expand(B, -1, -1)
            
        return x + pos_total


    def _forward_sincos_1d(self, x, input_size, ids_keep):
        T, H, W = [self._normalize_dim(v) for v in input_size]
        B = x.shape[0]

        key = ('SinCos', str(x.device), self.embed_dim, T, H, W)

        def build():
            total_len = T * H * W
            grid = np.arange(total_len, dtype=np.float32)
            pos_total_np = get_1d_sincos_pos_embed_from_grid(self.embed_dim, grid)
            return torch.from_numpy(pos_total_np).float().to(x.device).unsqueeze(0)

        pos_total = self._cache_lookup(key, build)

        if ids_keep is not None:
            pos_total = pos_total.expand(B, -1, -1)
            pos_total = torch.gather(pos_total, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else:
            pos_total = pos_total.expand(B, -1, -1)

        return x + pos_total
    

    def _forward_sincos_2d(self, x, input_size, ids_keep):
        T, H, W = [self._normalize_dim(v) for v in input_size]
        B = x.shape[0]
        key = ('SinCos_2D', str(x.device), self.embed_dim, T, H, W)

        def build():
            pos_spatial_np = get_2d_sincos_pos_embed(self.embed_dim, grid_size=H, grid_size2=W)
            pos_temporal_np = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.arange(T, dtype=np.float32))
            pos_sp = torch.from_numpy(pos_spatial_np).float().to(x.device).unsqueeze(0)
            pos_tm = torch.from_numpy(pos_temporal_np).float().to(x.device).unsqueeze(0)
            pos_total_local = pos_tm.unsqueeze(2) + pos_sp.unsqueeze(1)
            return pos_total_local.flatten(1, 2)

        pos_total = self._cache_lookup(key, build)
        
        if ids_keep is not None:
            pos_total = pos_total.expand(B, -1, -1)
            pos_total = torch.gather(pos_total, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else:
            pos_total = pos_total.expand(B, -1, -1)
        return x + pos_total

    def _forward_sincos_3d(self, x, input_size, ids_keep, scale):
        T, H, W = [self._normalize_dim(v) for v in input_size]
        B = x.shape[0]
        scale = self._normalize_scale(scale, 3)
        key = ('SinCos_3D', str(x.device), self.embed_dim, T, H, W, scale)

        def build():
            t, h, w = torch.arange(T), torch.arange(H), torch.arange(W)
            tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

            ED = self.embed_dim
            div = ED // 3
            ED1, ED2, ED3 = div, div, ED - 2 * div

            emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED1, tt.flatten(), scale[0])
            emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED2, hh.flatten(), scale[1])
            emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED3, ww.flatten(), scale[2])

            pos_np = np.concatenate([emb_t, emb_h, emb_w], axis=1)
            return torch.from_numpy(pos_np).float().to(x.device).unsqueeze(0)

        pos_total = self._cache_lookup(key, build)
        
        if ids_keep is not None:
            pos_total = pos_total.expand(B, -1, -1)
            pos_total = torch.gather(pos_total, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else:
            pos_total = pos_total.expand(B, -1, -1)
        return x + pos_total
    
    def _forward_sincos_1d_3(self, x, input_size, ids_keep, scale):
        T, H, W = [self._normalize_dim(v) for v in input_size]
        B = x.shape[0]
        scale = self._normalize_scale(scale, 3)
        key = ('SinCos_1D_3', str(x.device), self.embed_dim, T, H, W, scale)

        def build():
            t, h, w = torch.arange(T), torch.arange(H), torch.arange(W)
            tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')

            ED = self.embed_dim
            emb_t = get_1d_sincos_pos_embed_from_grid_with_resolution(ED, tt.flatten(), scale[0])
            emb_h = get_1d_sincos_pos_embed_from_grid_with_resolution(ED, hh.flatten(), scale[1])
            emb_w = get_1d_sincos_pos_embed_from_grid_with_resolution(ED, ww.flatten(), scale[2])

            pos_np = emb_t + emb_h + emb_w
            return torch.from_numpy(pos_np).float().to(x.device).unsqueeze(0)

        pos_total = self._cache_lookup(key, build)
        
        if ids_keep is not None:
            pos_total = pos_total.expand(B, -1, -1)
            pos_total = torch.gather(pos_total, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        else:
            pos_total = pos_total.expand(B, -1, -1)
            
        return x + pos_total
    

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
