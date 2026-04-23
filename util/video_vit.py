# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# import util.logging as logging
import pdb
import os

import numpy as np
import torch
import torch.nn as nn
from timm_utils.models.layers import to_2tuple
from timm_utils.models.vision_transformer import DropPath, Mlp
from einops import rearrange

from util.rope import apply_rotary_emb
# logger = logging.get_logger(__name__)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=(48,64),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=16,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)
        # print(kernel_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape 
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # pdb.set_trace()
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class PatchEmbed_v2(nn.Module):
    def __init__(
        self,
        input_dim=128,
        output_dim=768,
    ):
        super().__init__()
        self.proj= nn.Linear(
            in_features=input_dim,  # 输入特征维度
            out_features=output_dim, # 输出特征维度
            bias=True  # 包含偏置项（可选）
        )
    def forward(self, x):
        return self.proj(x) 


class CSIPatchEmbed_Complex(nn.Module):
    def __init__(self, input_dim=64, output_dim=768):
        super().__init__()
        self.proj = ComplexLinear(input_dim, output_dim // 2)

    def forward(self, csi_input):        
        if torch.is_complex(csi_input):
            x_real = csi_input.real
            x_imag = csi_input.imag
        else:
            # 假设最后一维 0是实部，1是虚部
            x_real = csi_input[..., 0]
            x_imag = csi_input[..., 1]

        return self.proj(x_real, x_imag)
    

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 权重分为实部 (A) 和 虚部 (B)
        self.weight_real = nn.Linear(in_features, out_features, bias=bias)
        self.weight_imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):        
        real_part = self.weight_real(x_real) - self.weight_imag(x_imag)
        imag_part = self.weight_real(x_imag) + self.weight_imag(x_real)
        
        return real_part, imag_part
    

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 4, 4),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x, attn_mask=None, save_attn=False):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            key_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,N]
            query_mask = ~attn_mask.unsqueeze(1).unsqueeze(3) # [B,1,N,1]
            combined_mask = key_mask | query_mask             # [B,1,N,N]
            # 使用大负数代替 -inf，数值更稳定
            # attn = attn.masked_fill(key_mask, float(-60000.0))
            dtype_min = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(key_mask, dtype_min) 

        attn = attn.softmax(dim=-1)
        # self.last_attn_map = attn.detach().cpu() 
        # if attn_mask is not None:
        #     # 把属于 padding 的 query 的 attention 行置 0（避免其输出有意义）
        #     attn = attn.masked_fill(query_mask, 0.0)

        # 保存注意力权重
        # if save_attn:
        #     return self._save_attention(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x
    
    def _save_attention(self, attn):
        """保存注意力权重到文件"""
        # 分离计算图并转为numpy
        attn_np = attn.detach().cpu().numpy()
        
        # 创建保存目录
        os.makedirs("attention_weights", exist_ok=True)
        
        pdb.set_trace()
        # 保存为.npy文件
        for b in range(attn_np.shape[0]):
            for h in range(self.num_heads):
                filename = f"attention_weights/attn_b{b}_h{h}_.npy"
                np.save(filename, attn_np[b, h])
        
        return filename


class RoPEAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=None, # 保留接口但不一定用
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, attn_mask=None, save_attn=False):
        B, N, C = x.shape
        # [B, N, num_heads, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)
        
        # --- 核心修改: 应用 RoPE ---
        # permute 放在后面做，apply_rotary_emb 期望 [B, N, H, D]
        if freqs_cis is not None:
            # 如果存在 Class Token，RoPE 通常跳过 index 0
            # 假设你的 CSI 数据全是 Patch Token，直接应用
            # 如果有 CLS Token，需要像这样:
            # q[:, 1:, ...], k[:, 1:, ...] = apply_rotary_emb(q[:, 1:, ...], k[:, 1:, ...], freqs_cis)
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # 转置为 [B, H, N, D] 以进行 Attention 计算
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            key_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,N]
            query_mask = ~attn_mask.unsqueeze(1).unsqueeze(3) # [B,1,N,1]
            combined_mask = key_mask | query_mask             # [B,1,N,N]
            # 使用大负数代替 -inf，数值更稳定
            # attn = attn.masked_fill(key_mask, float(-60000.0))
            dtype_min = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(key_mask, dtype_min) 

        attn = attn.softmax(dim=-1)
        # self.last_attn_map = attn.detach().cpu() 

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Linear_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = ((k.transpose(-2, -1) * self.scale).softmax(dim=-1)) @ v

        # attn = attn.softmax(dim=-1)

        # x = ((q.softmax(dim=-1)) @ attn).transpose(1, 2).reshape(B, N, C)
        x = ((q.softmax(dim=-1)) @ attn).reshape(B, N, C)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x

class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class Block_v2(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_mask=None, save_attn=False):
        if save_attn:
            return self.attn(self.norm1(x), attn_mask=attn_mask, save_attn=save_attn)
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask, save_attn=save_attn))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_v2_RoPE(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=RoPEAttention3D,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, freqs_cis=None, attn_mask=None, save_attn=False):
        # 传递 freqs_cis
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask, save_attn=save_attn))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class Linear_Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Linear_Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

if __name__ == '__main__':
    # input = torch.rand(2,9,512,512)
    # input = torch.unsqueeze(input,dim=1)   #torch.Size([2, 1, 10, 512, 512]) B,T,C,H,W
    # patch_embed = PatchEmbed(img_size=512,in_chans=1,frames=9,t_patch_size=3)
    # output = patch_embed(input)
    x = torch.rand(2,196,768)
    model = Attention()
    output = model(x)
    # print()


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# # import util.logging as logging
# import pdb
# import os

# import numpy as np
# import torch
# import torch.nn as nn
# from timm_utils.models.layers import to_2tuple
# from timm_utils.models.vision_transformer import DropPath, Mlp
# from einops import rearrange

# # logger = logging.get_logger(__name__)


# class PatchEmbed(nn.Module):
#     """Image to Patch Embedding"""

#     def __init__(
#         self,
#         img_size=(48,64),
#         patch_size=16,
#         in_chans=3,
#         embed_dim=768,
#         # temporal related:
#         frames=16,
#         t_patch_size=4,
#     ):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         assert img_size[1] % patch_size[1] == 0
#         assert img_size[0] % patch_size[0] == 0
#         assert frames % t_patch_size == 0
#         num_patches = (
#             (img_size[1] // patch_size[1])
#             * (img_size[0] // patch_size[0])
#             * (frames // t_patch_size)
#         )
#         self.input_size = (
#             frames // t_patch_size,
#             img_size[0] // patch_size[0],
#             img_size[1] // patch_size[1],
#         )
#         print(
#             f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
#         )
#         self.img_size = img_size
#         self.patch_size = patch_size

#         self.frames = frames
#         self.t_patch_size = t_patch_size

#         self.num_patches = num_patches

#         self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.t_grid_size = frames // t_patch_size

#         kernel_size = [t_patch_size] + list(patch_size)
#         # print(kernel_size)
#         self.proj = nn.Conv3d(
#             in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
#         )

#     def forward(self, x):
#         B, C, T, H, W = x.shape 
#         assert (
#             H == self.img_size[0] and W == self.img_size[1]
#         ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         # pdb.set_trace()
#         assert T == self.frames
#         x = self.proj(x).flatten(3)
#         x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
#         return x


# class PatchEmbed_v2(nn.Module):
#     def __init__(
#         self,
#         input_dim=128,
#         output_dim=768,
#     ):
#         super().__init__()
#         self.proj= nn.Linear(
#             in_features=input_dim,  # 输入特征维度
#             out_features=output_dim, # 输出特征维度
#             bias=True  # 包含偏置项（可选）
#         )
#     def forward(self, x):
#         return self.proj(x) 
    

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         input_size=(4, 14, 14),
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         assert attn_drop == 0.0  # do not use
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.input_size = input_size
#         assert input_size[1] == input_size[2]

#     def forward(self, x, attn_mask=None, save_attn=False):
#         B, N, C = x.shape
#         q = (
#             self.q(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         k = (
#             self.k(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         v = (
#             self.v(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         if attn_mask is not None:
#             key_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,N]
#             query_mask = ~attn_mask.unsqueeze(1).unsqueeze(3) # [B,1,N,1]
#             combined_mask = key_mask | query_mask             # [B,1,N,N]
#             # 使用大负数代替 -inf，数值更稳定
#             # attn = attn.masked_fill(key_mask, float(-60000.0))
#             dtype_min = torch.finfo(attn.dtype).min
#             attn = attn.masked_fill(key_mask, dtype_min) 

#         attn = attn.softmax(dim=-1)
#         # if attn_mask is not None:
#         #     # 把属于 padding 的 query 的 attention 行置 0（避免其输出有意义）
#         #     attn = attn.masked_fill(query_mask, 0.0)

#         # 保存注意力权重
#         if save_attn:
#             return self._save_attention(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = x.view(B, -1, C)
#         return x
    
#     def _save_attention(self, attn):
#         """保存注意力权重到文件"""
#         # 分离计算图并转为numpy
#         attn_np = attn.detach().cpu().numpy()
        
#         # 创建保存目录
#         os.makedirs("attention_weights", exist_ok=True)
        
#         pdb.set_trace()
#         # 保存为.npy文件
#         for b in range(attn_np.shape[0]):
#             for h in range(self.num_heads):
#                 filename = f"attention_weights/attn_b{b}_h{h}_.npy"
#                 np.save(filename, attn_np[b, h])
        
#         return filename
    

# class Linear_Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         input_size=(4, 14, 14),
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5

#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         assert attn_drop == 0.0  # do not use
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.input_size = input_size
#         assert input_size[1] == input_size[2]

#     def forward(self, x):
#         B, N, C = x.shape
#         q = (
#             self.q(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         k = (
#             self.k(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )
#         v = (
#             self.v(x)
#             .reshape(B, N, self.num_heads, C // self.num_heads)
#             .permute(0, 2, 1, 3)
#         )

#         # attn = (q @ k.transpose(-2, -1)) * self.scale

#         attn = ((k.transpose(-2, -1) * self.scale).softmax(dim=-1)) @ v

#         # attn = attn.softmax(dim=-1)

#         # x = ((q.softmax(dim=-1)) @ attn).transpose(1, 2).reshape(B, N, C)
#         x = ((q.softmax(dim=-1)) @ attn).reshape(B, N, C)
#         # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = x.view(B, -1, C)
#         return x

# class Block(nn.Module):
#     """
#     Transformer Block with specified Attention function
#     """

#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         attn_func=Attention,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = attn_func(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
    

# class Block_v2(nn.Module):
#     """
#     Transformer Block with specified Attention function
#     """

#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         attn_func=Attention,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = attn_func(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#     def forward(self, x, attn_mask=None, save_attn=False):
#         if save_attn:
#             return self.attn(self.norm1(x), attn_mask=attn_mask, save_attn=save_attn)
#         x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask, save_attn=save_attn))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
    

# class Linear_Block(nn.Module):
#     """
#     Transformer Block with specified Attention function
#     """

#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=False,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         attn_func=Linear_Attention,
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = attn_func(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#         )
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
    

# if __name__ == '__main__':
#     # input = torch.rand(2,9,512,512)
#     # input = torch.unsqueeze(input,dim=1)   #torch.Size([2, 1, 10, 512, 512]) B,T,C,H,W
#     # patch_embed = PatchEmbed(img_size=512,in_chans=1,frames=9,t_patch_size=3)
#     # output = patch_embed(input)
#     x = torch.rand(2,196,768)
#     model = Attention()
#     output = model(x)
#     # print()

