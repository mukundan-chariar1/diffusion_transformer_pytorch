import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from embedding import PatchEmbedding, SimpleTimeEmbedding, SinusoidalTimeEmbedding
from diffusion import *


# ----------------------- Building blocks -----------------------

class FFN(nn.Module):
    def __init__(self, dim: int, hidden_units: int, drop: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_units, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_units, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_units: int,
                 attn_drop: float = 0.0, drop: float = 0.0, drop_path: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=True, bias=bias
        )
        self.proj_drop = nn.Dropout(drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=False)
        self.mlp = FFN(embed_dim, hidden_units=hidden_units, drop=drop, bias=bias)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # FIX: modulate the feature dim (embed_dim), condition dim = embed_dim
        self.ada_attn = AdaLNMod(channels=embed_dim, cond_dim=embed_dim, with_gate=True, zero_init=True)
        self.ada_mlp  = AdaLNMod(channels=embed_dim, cond_dim=embed_dim, with_gate=True, zero_init=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Attention branch
        h1 = self.norm1(x)                 # (B,N,C)
        h1, g1 = self.ada_attn(h1, c)      # (B,N,C), (B,N,C)-broadcastable
        attn_out = self.attn(h1, h1, h1)[0]  # FIX: pass (Q,K,V); take output
        attn_out = self.proj_drop(attn_out)  # use proj_drop
        x = x + self.drop_path1(g1 * attn_out)

        # MLP branch
        h2 = self.norm2(x)
        h2, g2 = self.ada_mlp(h2, c)
        mlp_out = self.mlp(h2)
        x = x + self.drop_path2(g2 * mlp_out)

        return x

class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask

class AdaLNMod(nn.Module):
    """Modulate a *pre-normalized* tensor: y = (1+gamma(c))*x + beta(c)"""
    def __init__(self, channels: int, cond_dim: int, with_gate: bool = False, zero_init: bool = True):
        super().__init__()
        self.channels, self.with_gate = channels, with_gate
        out_dim = 2*channels + (channels if with_gate else 0)
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_dim))
        if zero_init:
            nn.init.zeros_(self.to_mod[1].weight); nn.init.zeros_(self.to_mod[1].bias)

    def forward(self, x, c):
        B = x.size(0)
        mod = self.to_mod(c)
        if self.with_gate:
            gamma, beta, gate = torch.split(mod, [self.channels]*3, dim=-1)
        else:
            gamma, beta = torch.split(mod, [self.channels, self.channels], dim=-1); gate = None
        shape = [B] + [1]*(x.dim()-2) + [self.channels]  # broadcast over tokens/spatial
        y = (1 + gamma).view(*shape) * x + beta.view(*shape)
        return (y, gate.view(*shape)) if self.with_gate else y

class DiT(nn.Module):
    def __init__(
        self,
        img_shape: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        n_layers: int = 6,
        embed_dim: int = 128,
        num_heads: int = 4,
        T: int = 1000,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        s: float=0.008,
        schedule_type: str="cosine",
        t_embed_dim: int = 128,
        hidden_units: int = 64,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
        learned_pos_embed: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        H, W = img_shape
        assert H % patch_size == 0 and W % patch_size == 0, "H/W must be divisible by patch size"
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        grid_h, grid_w = H // patch_size, W // patch_size
        num_patches = grid_h * grid_w
        self.num_patches=num_patches
        self.data_shape=(in_chans, *img_shape)
        
        self.diffusion = Diffusion(
            T = T,
            b_0 = b_0,
            b_T = b_T,
            n_data_dims = 3,
            s=s,
            schedule_type=schedule_type,
            )

        self.patch_embed = PatchEmbedding(
            img_shape=img_shape,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_patches=num_patches,
            dropout=drop_rate,
            in_channels=in_chans,
            learned_pos_embed=learned_pos_embed,
        )

        self.time_embed = SinusoidalTimeEmbedding(t_embed_dim=t_embed_dim, embed_dim=embed_dim)

        dpr = torch.linspace(0, drop_path_rate, steps=n_layers).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, hidden_units=hidden_units,
                    attn_drop=attn_drop_rate, drop=drop_rate, drop_path=dpr[i], bias=bias)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.ada_head = AdaLNMod(channels=embed_dim, cond_dim=embed_dim, with_gate=False, zero_init=True)
        self.head = nn.Linear(embed_dim, in_chans * patch_size * patch_size, bias=True)

    # ---- helpers ----
    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, PP = patches.shape
        P = self.patch_size
        C = self.in_chans
        H, W = self.img_shape
        gh, gw = H // P, W // P
        assert N == gh * gw and PP == C * P * P
        x = patches.view(B, gh, gw, P, P, C)              # [B, gh, gw, P, P, C]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()      # [B, C, gh, P, gw, P]
        x = x.view(B, C, H, W)                            # [B, C, H, W]
        return x

    def forward(self, x: torch.FloatTensor, t: torch.LongTensor) -> torch.FloatTensor:
        tokens = self.patch_embed(x)                 # [B, N, D]
        t_emb  = self.time_embed(t)                  # [B, D]

        # FIX: pass conditioning c=t_emb into each block
        for blk in self.blocks:
            tokens = blk(tokens, t_emb)
                
        tokens = self.norm(tokens)             # pre-normalize
        tokens = self.ada_head(tokens, t_emb)  # <-- adaptive modulation (no extra LN)
        patches = self.head(tokens)            # (B, N, C*P*P)
        eps_hat = self._unpatchify(patches)    # (B, C, H, W)
        eps_hat = self._unpatchify(patches)          # [B, C, H, W]
        return eps_hat
    
    @torch.inference_mode()
    def generate(self, n_samples: int, device: str = 'cuda') -> torch.FloatTensor:
        """
        Returns a stack of all denoised states:
        shape = (T+1, n_samples, C, H, W)
        """
        self.eval().to(device)
        dtype = next(self.parameters()).dtype
        C = self.in_chans
        H, W = self.img_shape

        # start from pure noise
        x = torch.randn((n_samples, C, H, W), device=device, dtype=dtype)
        xs = [x]

        # reuse t-batch tensor each step
        t_batch = torch.empty(n_samples, device=device, dtype=torch.long)

        for t in range(self.diffusion.T - 1, -1, -1):
            t_batch.fill_(t)
            eps_theta = self(x, t_batch)               # predict noise
            x = self.diffusion.reverse(x, t, eps_theta)
            xs.append(x)

        return torch.stack(xs)
