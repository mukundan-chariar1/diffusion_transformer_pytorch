from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_shape: tuple, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patcher=nn.Sequential(nn.Conv2d(
                                            in_channels=in_channels,
                                            out_channels=embed_dim,
                                            kernel_size=patch_size, # assumes kernel to by square
                                            stride=patch_size
                                            ),
                                    nn.Flatten(2))

        grid_size=(img_shape[0]//patch_size, img_shape[1]//patch_size)
        self.num_patches=grid_size[0]*grid_size[1]
        grid=torch.arange(grid_size[0])
        grid_mesh=torch.meshgrid(grid, grid, indexing="ij")
        grid_mesh=torch.stack(grid_mesh)
        
        grid_h=grid_mesh[0].reshape(-1)
        grid_w=grid_mesh[1].reshape(-1)
        
        factor = 10000 ** ((torch.arange(
                            start=0,
                            end=embed_dim // 4) / (embed_dim // 4))
                            )
        
        grid_h_emb = grid_h[:, None].repeat(1, embed_dim // 4) / factor
        grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)

        grid_w_emb = grid_w[:, None].repeat(1, embed_dim // 4) / factor
        grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
        pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)
        
        self.register_buffer("pos_emb", pos_emb)
        
    def forward(self, x):
        x=self.patcher(x).permute(0, 2, 1)
        x=self.pos_emb+x
        return x

def sinusoidal_time_embedding(t: torch.Tensor, dim: int, *, max_period: float=10000.0) -> torch.Tensor:
    assert dim%2==0, "dim must be even"
    factor = 10000 ** ((torch.arange(
                                    start=0, end=dim // 2, dtype=torch.float32, device=t.device) / (dim // 2))
                                    )
    
    t_emb = t[:, None].repeat(1, dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb

class SimpleTimeEmbedding(nn.Module):
    def __init__(self, T, t_embed_dim, embed_dim):
        super().__init__()
        self.embedding=nn.Embedding(T, t_embed_dim)
        self.mlp=nn.Sequential(
            nn.Linear(t_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, t):
        return self.mlp(self.embedding(t))

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, t_embed_dim: int, embed_dim: int):
        super().__init__()
        self.t_embed_dim=t_embed_dim
        self.mlp=nn.Sequential(
            nn.Linear(t_embed_dim, embed_dim), 
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h=sinusoidal_time_embedding(t, self.t_embed_dim)
        return self.mlp(h)