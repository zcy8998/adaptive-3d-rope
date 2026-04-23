import torch
import torch.nn as nn
import numpy as np
import math

# ==========================================
# 核心组件: 因子分解复数位置编码 (Scheme B)
# ==========================================
class ComplexFactorizedPosEmbedding(nn.Module):
    """
    PhysCSI 核心组件: 因子分解三维复数位置编码
    公式: H_out = H_in * exp(i * (w_t*t + w_f*f + w_s*s + theta))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 定义三个维度的频率参数 (可学习) [1, 1, D]
        # 初始化较小值模拟低频，允许网络慢慢学习高频特征
        self.omega_t = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.omega_h = nn.Parameter(torch.randn(1, 1, dim) * 0.02) # Freq / Height
        self.omega_w = nn.Parameter(torch.randn(1, 1, dim) * 0.02) # Antenna / Width
        
        # 初始相位
        self.theta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, input_size, ids_keep=None):
        """
        x: [B, N, D] (可以是实数或复数)
        input_size: (T, H, W) 原始网格大小
        ids_keep: [B, N_keep] 掩码索引
        """
        B = x.shape[0]
        T, H, W = input_size
        device = x.device
        
        # 1. 生成三维网格坐标 [T, H, W]
        # 使用 indexing='ij' 以匹配 meshgrid 的直观逻辑
        grid_t, grid_h, grid_w = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        flat_t = grid_t.flatten().view(1, -1, 1) # [1, L_total, 1]
        flat_h = grid_h.flatten().view(1, -1, 1)
        flat_w = grid_w.flatten().view(1, -1, 1)
        
        # 2. 处理 Mask (Gather) 或 全量 (Expand)
        if ids_keep is not None:
            # ids_keep: [B, N_keep] -> [B, N_keep, 1]
            gather_idx = ids_keep.unsqueeze(-1)
            
            # 扩展到 Batch 维度并 Gather
            batch_t = torch.gather(flat_t.expand(B, -1, -1), 1, gather_idx)
            batch_h = torch.gather(flat_h.expand(B, -1, -1), 1, gather_idx)
            batch_w = torch.gather(flat_w.expand(B, -1, -1), 1, gather_idx)
        else:
            batch_t = flat_t.expand(B, -1, -1)
            batch_h = flat_h.expand(B, -1, -1)
            batch_w = flat_w.expand(B, -1, -1)

        # 3. 计算相位角度 (广播机制: [B,N,1] * [1,1,D] -> [B,N,D])
        angle = (batch_t * self.omega_t) + \
                (batch_h * self.omega_h) + \
                (batch_w * self.omega_w) + \
                self.theta
        
        # 4. 生成旋转因子
        rotator = torch.exp(1j * angle)
        
        # 5. 转换为复数并旋转
        if not x.is_complex():
            # 假设输入为振幅或实部
            x = x.to(torch.complex64)
            
        return x * rotator
    

class Complex3DPosEmbedding_V2(nn.Module):
    """
    3D 通道切分型复数位置编码 (Channel Slicing / Concatenation)
    逻辑: 将 Embed Dim 切分为三份，分别独立编码 T, H, W
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 1. 确定每一段的长度
        # 假设 dim = 768 -> d_t=256, d_h=256, d_w=256
        self.d_t = dim // 3
        self.d_h = dim // 3
        self.d_w = dim - self.d_t - self.d_h # 处理不能整除的情况
        
        # 2. 分别定义三个维度的频率参数
        # 注意：这里每个 parameter 的维度变小了，只负责自己的那一部分通道
        self.omega_t = nn.Parameter(torch.randn(1, 1, self.d_t) * 0.02)
        self.omega_h = nn.Parameter(torch.randn(1, 1, self.d_h) * 0.02)
        self.omega_w = nn.Parameter(torch.randn(1, 1, self.d_w) * 0.02)
        
        # 初始相位也分开定义 (可选，为了严谨)
        self.theta_t = nn.Parameter(torch.zeros(1, 1, self.d_t))
        self.theta_h = nn.Parameter(torch.zeros(1, 1, self.d_h))
        self.theta_w = nn.Parameter(torch.zeros(1, 1, self.d_w))

    def forward(self, x, input_size, ids_keep=None):
        B = x.shape[0]
        T, H, W = input_size
        device = x.device
        
        # --- 1. 生成坐标 (与之前相同) ---
        grid_t, grid_h, grid_w = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        flat_t = grid_t.flatten().view(1, -1, 1)
        flat_h = grid_h.flatten().view(1, -1, 1)
        flat_w = grid_w.flatten().view(1, -1, 1)
        
        # --- 2. Mask 处理 (与之前相同) ---
        if ids_keep is not None:
            gather_idx = ids_keep.unsqueeze(-1)
            batch_t = torch.gather(flat_t.expand(B, -1, -1), 1, gather_idx)
            batch_h = torch.gather(flat_h.expand(B, -1, -1), 1, gather_idx)
            batch_w = torch.gather(flat_w.expand(B, -1, -1), 1, gather_idx)
        else:
            batch_t = flat_t.expand(B, -1, -1)
            batch_h = flat_h.expand(B, -1, -1)
            batch_w = flat_w.expand(B, -1, -1)

        # --- 3. 分别计算各部分的相位 (核心修改) ---
        # 注意：这里不再相加，而是分别计算，形状较小
        angle_t = (batch_t * self.omega_t) + self.theta_t # [B, N, d_t]
        angle_h = (batch_h * self.omega_h) + self.theta_h # [B, N, d_h]
        angle_w = (batch_w * self.omega_w) + self.theta_w # [B, N, d_w]
        
        # --- 4. 在通道维度拼接相位 ---
        # [B, N, d_t] + [B, N, d_h] + [B, N, d_w] -> [B, N, D]
        angle_total = torch.cat([angle_t, angle_h, angle_w], dim=-1)
        
        # --- 5. 生成旋转因子并旋转 ---
        rotator = torch.exp(1j * angle_total)
        
        if not x.is_complex():
            x = x.to(torch.complex64)
            
        return x * rotator

# ==========================================
# 基础数据嵌入层
# ==========================================
class DataEmbedding(nn.Module):
    """
    将原始 CSI Patch 数据投影到高维特征空间
    """
    def __init__(self, c_in, d_model, args=None):
        super(DataEmbedding, self).__init__()
        # 计算 Patch 的总维度
        patch_dim = c_in * args.patch_size * args.patch_size * args.t_patch_size
        self.input_proj = nn.Linear(patch_dim, d_model)
        
    def forward(self, x):
        # x: [N, L, patch_vol * 2] (已经过 patchify)
        return self.input_proj(x)

# ==========================================
# 工具函数: SinCos 编码生成器
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size2=None, cls_token=False):
    if grid_size2 is None: grid_size2 = grid_size
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size2, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size2])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid_with_resolution(embed_dim, pos, resolution=1.0):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1) * resolution
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb