from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

class Diffusion(nn.Module):

    def __init__(
        self,
        T: int=1000,
        b_0: float=1e-4,
        b_T: float=2e-2,
        n_data_dims: int=3,
        s: float=0.008,
        schedule_type: str='quadratic'
        ):
        super().__init__()
        self.T=T
        self.s=s

        if schedule_type=='quadratic':
            beta=torch.linspace(b_0**0.5, b_T**0.5, T)**2
            beta=beta.view(T, *([1]*n_data_dims))
            alpha=1-beta
            alpha_bar=alpha.cumprod(dim=0)
        elif schedule_type=='cosine':
            f=lambda t: torch.cos((t/T+s)/(1+s)*torch.pi/2)**2
            alpha_bar=f(torch.arange(T+1))/f(torch.tensor([0.]))
            alpha=alpha_bar[1:]/alpha_bar[:-1]
            beta=1-alpha
            alpha_bar=alpha_bar[1:]
        elif schedule_type=="linear":
            beta=torch.linspace(b_0, b_T, T)
            beta=beta.view(T, *([1]*n_data_dims))
            alpha=1-beta
            alpha_bar=alpha.cumprod(dim=0)
        else: NotImplementedError
        
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta', beta)

    @torch.no_grad()
    def forward(
            self,
            x_0: torch.Tensor,
            t: torch.Tensor,
            )-> Tuple[torch.Tensor, torch.Tensor]:

        alpha_bar_t=self.alpha_bar[t].view(-1, *([1]*(x_0.dim()-1)))
        
        mu=torch.sqrt(alpha_bar_t)*x_0

        std=torch.sqrt(1-alpha_bar_t)

        eps_q=torch.randn_like(x_0)
        x_t=mu+std*eps_q

        return x_t, eps_q
    
    @torch.inference_mode()
    def reverse(
            self,
            x_t: torch.Tensor,
            t: int,
            eps_theta: torch.Tensor,
            ) -> torch.Tensor:

        alpha_t=self.alpha[t].view(1, *([1]*(x_t.dim()-1)))
        alpha_bar_t=self.alpha_bar[t].view(1, *([1]*(x_t.dim()-1)))
        beta_t=self.beta[t].view(1, *([1]*(x_t.dim()-1)))

        mu=(1/torch.sqrt(alpha_t))*(x_t-(beta_t/torch.sqrt(1-alpha_bar_t))*eps_theta)

        std=torch.sqrt(beta_t)

        eps_p=torch.zeros_like(x_t) if t==0 else torch.randn_like(x_t)
        x_t_1=mu+std*eps_p

        return x_t_1