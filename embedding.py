from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_shape: Tuple, embed_dim: int, patch_size: int, num_patches: int, dropout: float, in_channels: int, learned_pos_embed: bool):
        super().__init__()
        self.patcher = nn.Sequential(
        nn.Conv2d(
        in_channels=in_channels,
        out_channels=embed_dim,
        kernel_size=patch_size, # assumes kernel to by square
        stride=patch_size
        ),
        nn.Flatten(2))

        grid_size = (img_shape[0] // patch_size, img_shape[1] // patch_size)
        num_patches = grid_size[0] * grid_size[1]

        if learned_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        else:
            pe = sinusoidal_2d_pos_embed(embed_dim, *grid_size, cls=False)
            self.register_buffer("pos_embed", pe, persistent=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.patcher(x).permute(0, 2, 1)
        x = self.pos_embed + x
        x = self.dropout(x)
        return x

def sinusoidal_2d_pos_embed(embed_dim: int, H: int, W: int, *, cls: bool = False, device=None, dtype=None):
    """
    Returns [1, H*W(+1), D] 2D sin/cos PE. `embed_dim` must be divisible by 4.
    """
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be a multiple of 4 for 2D sincos PE.")
    dtype = dtype or torch.float32

    dim_h = embed_dim // 2
    dim_w = embed_dim // 2
    pe_h = get_1d_sincos_pos_embed(dim_h, H)  # [H, dim_h]
    pe_w = get_1d_sincos_pos_embed(dim_w, W)  # [W, dim_w]

    rows = []
    for i in range(H):
        row = torch.cat((pe_h[i].unsqueeze(0).repeat(W, 1), pe_w), dim=1)  # [W, D]
        rows.append(row)
    pe = torch.stack(rows, dim=0)          # ✅ [H, W, D] (use stack, not torch.tensor([...]))
    pe = pe.view(H * W, embed_dim)         # [H*W, D]

    if cls:
        cls_tok = torch.zeros(1, embed_dim, device=device, dtype=dtype)
        pe = torch.cat([cls_tok, pe], dim=0)  # [H*W+1, D]
    return pe.unsqueeze(0)


def get_1d_sincos_pos_embed(dim, length):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def sinusoidal_time_embedding(t: torch.Tensor, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0, "dim must be even"
    t = t.float().unsqueeze(1)                    # (B,1)
    half = dim // 2
    # log-spaced frequencies
    exponents = torch.arange(half, device=t.device, dtype=t.dtype) / half
    freqs = torch.exp(-math.log(max_period) * exponents)  # (half,)
    angles = t * freqs                                    # (B,half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)  # (B,dim)
    return emb

class SimpleTimeEmbedding(nn.Module):
    def __init__(self, T, t_embed_dim, embed_dim):
        super().__init__()
        self.embedding=nn.Embedding(T, t_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(t_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, t):
        return self.mlp(self.embedding(t))

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, t_embed_dim: int, embed_dim: int):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(t_embed_dim, embed_dim), 
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = sinusoidal_time_embedding(t, self.t_embed_dim)  # (B, dim_t)
        return self.mlp(h)                            # (B, out)