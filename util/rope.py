import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 3D 混合频率初始化 (Time, Frequency, Antenna)
def init_random_3d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    初始化 3D 频率: [3, dim//2] -> 对应 (Time, Freq, Antenna)
    """
    # 生成基准幅度
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_list = []
    
    # 为三个维度 (Time, Freq, Antenna) 生成频率
    # 论文中 Mixed 模式是随机初始化的角度
    for _ in range(3): # 3 dimensions
        freqs_dim = []
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
            # 生成混合了 cos 和 sin 的频率分量
            f = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            freqs_dim.append(f)
        freqs_list.append(torch.stack(freqs_dim, dim=0))
    
    # Shape: [3, num_heads, dim//2]
    freqs = torch.stack(freqs_list, dim=0)
    return freqs


def init_standard_3d_freqs_aligned(dim: int, num_heads: int, theta: float = 10000.0):
    """
    保持原有 [3, num_heads, dim//2] 形状的标准化频率初始化。
    消除了随机角度和负值，使用纯粹的标准指数衰减频率。
    """
    # 1. 生成标准的 RoPE 基准频率 (严格为正)
    # 对应标准公式: 1 / (theta ** (2i / dim))
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # 2. 扩展为需要的形状 [3, num_heads, dim//2]
    # 标准初始化中，各 Head 通常共享同一组频率（除非你有特殊的 Head-scaling 需求）
    freqs = inv_freq.view(1, 1, -1).expand(3, num_heads, -1)
    
    # 返回 clone 以确保在作为 Parameter 注册或修改时内存连续且独立
    return freqs.clone()


# 2. 生成 3D 坐标网格
def init_t_3d(T: int, F: int, A: int):
    """
    生成 Time, Freq, Antenna 的网格坐标
    假设 Token 展平顺序是 Time -> Freq -> Antenna
    """
    # 总 Token 数
    total_tokens = T * F * A
    grid = torch.arange(total_tokens, dtype=torch.float32)
    
    # 反解坐标 (假设行优先展平: T, F, A)
    # index = t * (F * A) + f * (A) + a
    t_a = grid % A
    t_f = torch.div(grid, A, rounding_mode='floor') % F
    t_t = torch.div(grid, F * A, rounding_mode='floor')

    return t_t, t_f, t_a

# 3. 计算 3D 混合复数指数 (Complex Exponential)
def compute_mixed_cis_3d(freqs: torch.Tensor, t_t: torch.Tensor, t_f: torch.Tensor, t_a: torch.Tensor):
    """计算混合频率的旋转向量"""
    # freqs: [3, num_heads, D/2]
    # t: [N]
    t_t = t_t.unsqueeze(-1).unsqueeze(1) # [N, 1, 1]
    t_f = t_f.unsqueeze(-1).unsqueeze(1)
    t_a = t_a.unsqueeze(-1).unsqueeze(1)
    
    # freqs: [1, num_heads, D/2] (unsqueeze dim 0 for broadcast)
    angle_t = t_t * freqs[0].unsqueeze(0) 
    angle_f = t_f * freqs[1].unsqueeze(0)
    angle_a = t_a * freqs[2].unsqueeze(0)
    
    # 混合相位
    angle_mixed = angle_t + angle_f + angle_a # [N, num_heads, D/2]
    freqs_cis = torch.polar(torch.ones_like(angle_mixed), angle_mixed)
    return freqs_cis


def init_decoupled_freqs(dim: int, num_heads: int, theta: float = 10000.0):
    """
    生成分离的频率列表。
    input dim: 模型的 Head Dimension (例如 64)
    output: List[Tensor, Tensor, Tensor] 
            例如 [[num_heads, 10], [num_heads, 11], [num_heads, 11]]
    """
    half_dim = dim // 2
    # 计算切分 (例如 32 -> 10, 11, 11)
    splits = [half_dim // 3] * 3
    for i in range(half_dim % 3):
        splits[-(i+1)] += 1
    
    freqs_list = []
    for s_dim in splits:
        # 标准 RoPE 频率生成方式: theta^(-2i/d)
        freqs = 1.0 / (theta ** (torch.arange(0, s_dim, 1).float() / s_dim))
        # 扩展到多头
        freqs = freqs.unsqueeze(0).repeat(num_heads, 1) # [num_heads, s_dim]
        freqs_list.append(freqs)
        
    # [Critical Change] 
    # 不要 stack，因为 s_dim 可能不相等 (比如 10, 11, 11)
    # 直接返回 list
    return freqs_list


def compute_decoupled_cis_3d(freqs, t_t: torch.Tensor, t_f: torch.Tensor, t_a: torch.Tensor):
    """
    计算分离维度的旋转向量 (Factorized RoPE)
    freqs: nn.ParameterList 或 List[Tensor]，包含 3 个不同形状的 Tensor
    """
    # 1. 扩展时间/位置索引维度以支持广播 [N] -> [N, 1, 1]
    t_t = t_t.unsqueeze(-1).unsqueeze(1) 
    t_f = t_f.unsqueeze(-1).unsqueeze(1)
    t_a = t_a.unsqueeze(-1).unsqueeze(1)

    # 2. 分别计算各个维度的角度
    # freqs[0], freqs[1], freqs[2] 分别是 Time, Freq, Antenna 的频率参数
    # 它们的形状可能是 [num_heads, 10], [num_heads, 11], [num_heads, 11]
    
    # [N, 1, 1] * [1, num_heads, dim_i] -> [N, num_heads, dim_i]
    angle_t = t_t * freqs[0].unsqueeze(0)  
    angle_f = t_f * freqs[1].unsqueeze(0)  
    angle_a = t_a * freqs[2].unsqueeze(0)  

    # 3. 分别生成各自的复数旋转向量 (cis)
    cis_t = torch.polar(torch.ones_like(angle_t), angle_t)
    cis_f = torch.polar(torch.ones_like(angle_f), angle_f)
    cis_a = torch.polar(torch.ones_like(angle_a), angle_a)

    # 4. 在最后一个维度拼接 (Concatenate)
    # 结果形状: [N, num_heads, total_dim/2]
    freqs_cis = torch.cat([cis_t, cis_f, cis_a], dim=-1)

    return freqs_cis


# 4. 辅助函数: 广播形状调整
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
         shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):
         # 匹配 [N, H, D/2] 到 [B, N, H, D/2]
         shape = [1, x.shape[1], x.shape[2], x.shape[3]] # Assume x is [B, N, H, D/2]
         # 这里简化处理，直接用 view 广播
         return freqs_cis.unsqueeze(0) # [1, N, H, D/2]
    
    return freqs_cis

# 5. 应用旋转 (保持不变，这是通用的)
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq: [B, N, H, D]
    # reshape to [B, N, H, D/2, 2] -> view as complex [B, N, H, D/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # freqs_cis: [N, H, D/2]
    # 广播 freqs_cis 到 [B, N, H, D/2]
    if len(freqs_cis.shape) == 3:
        freqs_cis = freqs_cis.unsqueeze(0) # [1, N, H, D/2]
        
    # print(xq.shape)
    # print(freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

######
def compute_splits(half_dim: int):
    base = half_dim // 3
    splits = [base, base, base]
    rem = half_dim - base * 3
    for i in range(rem):
        splits[-(i+1)] += 1
    return splits  # e.g. [10,11,11]

def init_bank_params(head_dim: int, num_heads: int, theta: float, device=None, learnable=True):
    """返回 nn.ParameterList of 3 tensors (time,freq,ant) each [num_heads, s_i]"""
    half = head_dim // 2
    splits = compute_splits(half)
    params = nn.ParameterList()
    for s in splits:
        idx = torch.arange(0, s, dtype=torch.float32, device=device)
        base = 1.0 / (theta ** (idx / float(s)))  # [s]
        base = base.unsqueeze(0).repeat(num_heads, 1)  # [num_heads, s]
        if learnable:
            params.append(nn.Parameter(base))
        else:
            params.append(base)
    return params

def compute_cis_from_params(freqs_params, phase_params, t_idx, f_idx, a_idx):
    """
    freqs_params: nn.ParameterList length 3, each [num_heads, s]
    phase_params: list length 3 of same shapes or None (per-head phase offsets)
    t_idx/f_idx/a_idx: [N] float tensors on correct device
    returns: complex64 tensor shape [N, num_heads, half_dim]
    """
    N = t_idx.shape[0]
    num_heads = freqs_params[0].shape[0]

    t = t_idx.unsqueeze(1).unsqueeze(-1)  # [N,1,1]
    f = f_idx.unsqueeze(1).unsqueeze(-1)
    a = a_idx.unsqueeze(1).unsqueeze(-1)

    parts = []
    for param, pos, phase in zip(freqs_params, (t, f, a), phase_params):
        # param: [num_heads, s]
        freq = param.unsqueeze(0)  # [1, num_heads, s]
        angle = pos * freq  # [N, num_heads, s]
        if phase is not None:
            angle = angle + phase.unsqueeze(0)  # broadcast [1,num_heads,s]
        real = torch.cos(angle)
        imag = torch.sin(angle)
        cis = torch.complex(real, imag)  # complex64
        parts.append(cis)
    cis_cat = torch.cat(parts, dim=-1)  # [N, num_heads, half_dim]
    return cis_cat


def init_lta_3d_freqs(dim: int, num_heads: int, theta: float = 10000.0):
    """
    VideoRoPE LTA (Low-frequency Time Allocation) 频率初始化策略。
    
    核心逻辑：
    1. 生成整个特征维度 (dim//2) 的全局频率谱。
    2. 将 高频部分 分配给 空间维度 (Freq, Antenna)。
    3. 将 低频部分 (Tail) 分配给 时间维度 (Time)。
    
    Args:
        dim: 模型的 head_dim (e.g. 64)
        num_heads: 注意力头数
        theta: 基准频率 (Default: 10000.0)
    
    Returns:
        List[Tensor]: [freqs_t, freqs_f, freqs_a]，每个形状为 [num_heads, split_dim]
        注意：返回顺序调整为 T, F, A 以便后续计算，但内部数值来源于频谱的不同位置。
    """
    half_dim = dim // 2
    
    # 1. 计算分割方案 (Splits)
    # 我们需要将 half_dim 分给 T, F, A
    # 假设均分，或者根据具体需求调整。这里采用原始代码的逻辑进行均分处理余数。
    base = half_dim // 3
    splits = [base, base, base] # [s1, s2, s3]
    rem = half_dim - base * 3
    for i in range(rem):
        splits[-(i+1)] += 1
    
    # 这里的 splits 对应物理维度的分配大小，通常我们定义顺序。
    # 假设我们想把三个块分别给 F, A, T (顺序不重要，重要的是下面的分配逻辑)
    s_f, s_a, s_t = splits[0], splits[1], splits[2]
    
    # 2. 生成全局频率谱 (Global Spectrum)
    # 频率公式: theta ^ (-2i / d)
    # index 0 -> High Freq (1.0)
    # index max -> Low Freq (1/theta)
    global_indices = torch.arange(0, half_dim, dtype=torch.float32)
    global_freqs = 1.0 / (theta ** (global_indices / half_dim))
    
    # 3. LTA 分配策略 (Critical Step)
    # Space (Freq & Antenna) <- Indices (High & Mid Freq) -> 频谱头部
    # Time                 <- Indices (Low Freq)      -> 频谱尾部
    
    # 切片索引
    idx_f_end = s_f
    idx_a_end = s_f + s_a
    
    # 提取频率
    # Freq 维度使用最高频 (索引 0 ~ s_f)
    freqs_f_raw = global_freqs[0 : idx_f_end]
    
    # Antenna 维度使用中频 (索引 s_f ~ s_f+s_a)
    freqs_a_raw = global_freqs[idx_f_end : idx_a_end]
    
    # Time 维度强制使用最低频 (索引 s_f+s_a ~ end)
    freqs_t_raw = global_freqs[idx_a_end : ]
    
    # 4. 扩展到多头并打包
    # 返回列表，顺序对应 compute_lta_cis_3d 中的拼接顺序
    freqs_list = []
    
    # 扩展 [s_dim] -> [num_heads, s_dim]
    f_t = freqs_t_raw.unsqueeze(0).repeat(num_heads, 1)
    f_f = freqs_f_raw.unsqueeze(0).repeat(num_heads, 1)
    f_a = freqs_a_raw.unsqueeze(0).repeat(num_heads, 1)
    
    # 返回顺序: Time, Freq, Antenna (需与 compute 函数一致)
    return [f_t, f_f, f_a]

def compute_lta_cis_3d(freqs_list, t_t: torch.Tensor, t_f: torch.Tensor, t_a: torch.Tensor):
    """
    计算基于 LTA 策略的旋转位置编码 (Factorized/Decoupled)。
    
    Args:
        freqs_list: init_lta_3d_freqs 返回的列表 [freqs_t, freqs_f, freqs_a]
                    其中 freqs_t 包含的是最低频分量。
        t_t, t_f, t_a: 展平后的坐标索引 [N]
    """
    # 解包频率
    freqs_t, freqs_f, freqs_a = freqs_list[0], freqs_list[1], freqs_list[2]
    
    # 1. 扩展时间/位置索引维度以支持广播 [N] -> [N, 1, 1]
    t_t = t_t.unsqueeze(-1).unsqueeze(1)
    t_f = t_f.unsqueeze(-1).unsqueeze(1)
    t_a = t_a.unsqueeze(-1).unsqueeze(1)

    # 2. 计算角度 (Outer Product)
    # [N, 1, 1] * [1, num_heads, s_dim] -> [N, num_heads, s_dim]
    angle_t = t_t * freqs_t.unsqueeze(0)
    angle_f = t_f * freqs_f.unsqueeze(0)
    angle_a = t_a * freqs_a.unsqueeze(0)

    # 3. 转复数 (Polar)
    cis_t = torch.polar(torch.ones_like(angle_t), angle_t)
    cis_f = torch.polar(torch.ones_like(angle_f), angle_f)
    cis_a = torch.polar(torch.ones_like(angle_a), angle_a)

    # 4. 拼接 (Concatenate)
    # 注意：这里的拼接顺序必须与你模型期望的特征维度物理含义一致。
    # 即使 Time 使用的是低频，拼接后的 cis 向量在 dim 维度上通常还是按照 t, f, a 或者 f, a, t 排列。
    # 此处我们按照 init_lta_3d_freqs 的切片逻辑：
    # 原始全谱是 [Freq_part, Ant_part, Time_part] (High -> Low)
    # 如果我们要还原这个全谱结构 (Diagonal Layout)，我们需要按此顺序拼接。
    
    # 但由于 Transformer 的 hidden dim 是无序的，只要 Encoder 和 Decoder 一致即可。
    # 为了物理可解释性，通常建议把 Time 放在特征向量的末尾（对应频谱末尾）。
    
    # Option A: 严格还原 High -> Low 频谱顺序 (推荐 LTA 做法)
    # 拼接顺序: Freq(High) -> Ant(Mid) -> Time(Low)
    freqs_cis = torch.cat([cis_f, cis_a, cis_t], dim=-1)
    
    # Option B: 如果你的模型其他部分(如初始化)强依赖于 T, F, A 的顺序
    # freqs_cis = torch.cat([cis_t, cis_f, cis_a], dim=-1) 

    return freqs_cis


def init_mrope_interleaved_3d_freqs(dim: int, num_heads: int, theta: float = 10000.0):
    """
    M-RoPE 风格的交错式频率初始化 (Interleaved 3D Frequencies).
    
    逻辑：
    1. 生成全维度的基准频率谱。
    2. 创建三个全零向量。
    3. 按照 Round-Robin (轮询) 的方式，将基准频率分配给 T, F, A。
       - Index % 3 == 0 -> Time
       - Index % 3 == 1 -> Freq
       - Index % 3 == 2 -> Antenna
    4. 未被分配的位置保持为 0。
    
    这利用了 compute_mixed_cis_3d 中的加法性质：
    Angle = (t * freq_t) + (f * freq_f) + (a * freq_a)
    如果 freq_t 在某一位是 0，那么 t 就不影响该位的旋转，该位完全由 f 或 a 控制。
    
    Args:
        dim: head_dim (e.g. 64)
        num_heads: 头数
        theta: 基准频率
        
    Returns:
        Tensor [3, num_heads, dim//2]
    """
    half_dim = dim // 2
    
    # 1. 生成完整的基准频率谱 (0 到 half_dim)
    # 这里的 indices 是全局的，对应整个特征向量的每一个通道
    freqs_base = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    
    # 2. 初始化三个全零容器
    freqs_t = torch.zeros_like(freqs_base)
    freqs_f = torch.zeros_like(freqs_base)
    freqs_a = torch.zeros_like(freqs_base)
    
    # 3. 交错分配 (Interleave)
    # Time 占据 0, 3, 6...
    freqs_t[0::3] = freqs_base[0::3]
    
    # Freq 占据 1, 4, 7...
    freqs_f[1::3] = freqs_base[1::3]
    
    # Antenna 占据 2, 5, 8...
    freqs_a[2::3] = freqs_base[2::3]
    
    # 4. 扩展到多头 [num_heads, half_dim]
    # 注意：所有头共享相同的交错模式，这符合 Transformer 的标准设计
    ft = freqs_t.unsqueeze(0).repeat(num_heads, 1)
    ff = freqs_f.unsqueeze(0).repeat(num_heads, 1)
    fa = freqs_a.unsqueeze(0).repeat(num_heads, 1)
    
    # 5. 堆叠返回 [3, num_heads, half_dim]
    # 这样可以直接兼容你原来的 nn.Parameter(freqs) 初始化方式
    return torch.stack([ft, ff, fa], dim=0)