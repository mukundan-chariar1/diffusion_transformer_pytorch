import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from embedding import PatchEmbedding, SimpleTimeEmbedding, SinusoidalTimeEmbedding
from diffusion import *

class Block(nn.Module):
    def __init__(self, hidden_dim: int=256, num_heads: int=6, attn_drop: float=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn=nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=True, bias=True
        )
        self.norm2=nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.mlp=nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim),
                               nn.GELU(),
                               nn.Linear(4*hidden_dim, hidden_dim))
        
        self.adaLN=nn.Sequential(nn.SiLU(),
                                 nn.Linear(hidden_dim, 6*hidden_dim))
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (pre_attn_shift, pre_attn_scale, post_attn_scale, pre_mlp_shift, pre_mlp_scale, post_mlp_scale)=self.adaLN(c).chunk(6, dim=1)
        out=x
        attn_norm_output=(self.norm1(out)*(1+pre_attn_scale.unsqueeze(1))+pre_attn_shift.unsqueeze(1))
        out=out+post_attn_scale.unsqueeze(1)*self.attn(attn_norm_output, attn_norm_output, attn_norm_output)[0]
        mlp_norm_output=(self.norm2(out)* (1+pre_mlp_scale.unsqueeze(1))+pre_mlp_shift.unsqueeze(1))
        out=out+post_mlp_scale.unsqueeze(1)*self.mlp(mlp_norm_output)
        
        return out
    
class DiTBackbone(nn.Module):
    def __init__(self, input_size: tuple=(32, 32), in_channels: int=4, patch_size: int=16, embed_dim: int=256, hidden_dim: int=256, num_heads: int=6, num_layers: int=6):
        super().__init__()
        self.input_size=input_size
        self.in_channels=in_channels
        
        self.patch_size=patch_size
        
        self.grid_size=(input_size[0]//patch_size, input_size[1]//patch_size)
        self.num_patches=self.grid_size[0]*self.grid_size[1]
        
        self.patcher=PatchEmbedding(input_size, in_channels, hidden_dim, patch_size)
        
        self.time_embedding=SinusoidalTimeEmbedding(embed_dim, hidden_dim)
        
        self.norm=nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        
        self.adaLNzero=nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        
        self.blocks=nn.ModuleList([Block(hidden_dim, num_heads) for _ in range(num_layers)])
        
        self.final_layer=nn.Linear(hidden_dim, patch_size*patch_size*in_channels)

        for b in self.blocks:  # the last Linear in those stacks
            m=b.adaLN[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        m=self.adaLNzero[-1]
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, PP = x.shape
        P = self.patch_size
        C = self.in_channels
        H, W = self.input_size
        gh, gw = H // P, W // P
        assert N == gh * gw and PP == C * P * P
        x = x.view(B, gh, gw, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x
        
    def forward(self, x: torch.Tensor, c: torch.LongTensor) -> torch.Tensor:
        out=self.patcher(x)
        t_emb=self.time_embedding(c)
        
        for block in self.blocks:
            out=block(out, t_emb)
            
        pre_mlp_shift, pre_mlp_scale = self.adaLNzero(t_emb).chunk(2, dim=1)
        out = (self.norm(out)*(1+pre_mlp_scale.unsqueeze(1))+pre_mlp_shift.unsqueeze(1))
        
        out=self.final_layer(out)
        
        return self.unpatchify(out)
        
class DiT(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (32, 32),
        patch_size: int = 16,
        num_layers: int = 6,
        embed_dim: int = 128,
        num_heads: int = 4,
        T: int = 1000,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        s: float=0.008,
        schedule_type: str="cosine",
        hidden_dim: int = 64,
        in_chans: int = 4,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        grid_h, grid_w = input_size[0] // patch_size, input_size[0] // patch_size
        num_patches = grid_h * grid_w
        self.num_patches=num_patches
        self.data_shape=(in_chans, *input_size)
        
        self.diffusion = Diffusion(
            T = T,
            b_0 = b_0,
            b_T = b_T,
            n_data_dims = 3,
            s=s,
            schedule_type=schedule_type,
            )
        
        self.backbone=DiTBackbone(input_size, 
                                  in_chans, 
                                  patch_size, 
                                  embed_dim, 
                                  hidden_dim, 
                                  num_heads, 
                                  num_layers
                                  )
        
    def forward(self, x: torch.FloatTensor, t: torch.LongTensor) -> torch.FloatTensor:
        return self.backbone(x, t)
    
    @torch.inference_mode()
    def generate(self, n_samples: int, device: str = 'cuda') -> torch.FloatTensor:
        self.eval().to(device)
        dtype = next(self.parameters()).dtype
        C = self.in_chans
        H, W = self.input_size

        x = torch.randn((n_samples, C, H, W), device=device, dtype=dtype)
        xs = [x]

        t_batch = torch.empty(n_samples, device=device, dtype=torch.long)

        for t in range(self.diffusion.T - 1, -1, -1):
            t_batch.fill_(t)
            eps_theta = self(x, t_batch)
            x = self.diffusion.reverse(x, t, eps_theta)
            xs.append(x)

        return torch.stack(xs)