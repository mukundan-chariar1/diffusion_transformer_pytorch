from typing import Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

class Diffusion(nn.Module):

    def __init__(
        self,
        T: int = 1000, # total number of diffusion steps,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        n_data_dims: int = 3, # number of data dimensions. For example, colored image data has 3 (channel, height, width)
        s: float=0.008,
        schedule_type: str='quadratic'
        ):
        super().__init__()
        self.T = T
        self.s=s

        if schedule_type=='quadratic':
            # calculate the 1D tensor for beta containing the values for each diffusion step
            # using quadratic schedule
            beta = torch.linspace(b_0**0.5, b_T**0.5, T)**2
            beta=beta.view(T, *([1]*n_data_dims))
            alpha = 1. - beta
            alpha_bar = alpha.cumprod(dim=0)
        elif schedule_type=='cosine':
            f = lambda t: torch.cos((t/T+s)/(1+s) * torch.pi/2) ** 2
            alpha_bar = f(torch.arange(T+1))/f(torch.tensor([0.]))
            alpha = alpha_bar[1:]/alpha_bar[:-1]
            beta = 1 - alpha
            alpha_bar = alpha_bar[1:]
        else: NotImplementedError

        # based on n_data_dims, make the shape of beta broadcastable to batched data
        
            
        # calculate alpha and alpha_bar from beta
        # both alpha and alpha_bar have T elements as well, one for each diffusion step
        

        # register the tensors as buffers to be saved with the model
        # and to be moved to the right device when calling .to(device)
        # You can access them like a normal attribute, like self.alpha
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta', beta)

    @torch.no_grad()
    def forward(
            self,
            x_0: torch.FloatTensor, # (batch_size, *data_shape),
            t: torch.LongTensor, # (batch_size,),
            )-> Tuple[torch.FloatTensor, torch.FloatTensor]: # noisy data and the epsilon used to corrupt it
        """
        for each data sample in the batch, draw a sample from q(x_t|x_0, t)
        according to the schedule and the corresponding diffusion step of each data sample.

        You can index alpha, alpha_bar, or beta with the tensor t directly,
        and get a batch of alpha, alpha_bar, or beta values.

        Returns:
        x_t: torch.FloatTensor, the corrupted batch
        eps_q: torch.FloatTensor, the noise used to corrupt the data
        """

        alpha_bar_t=self.alpha_bar[t].view(-1, *([1]*(x_0.dim()-1)))
        
        # mean of q(x_t|x_0, t)
        mu = torch.sqrt(alpha_bar_t)*x_0

        # std of q(x_t|x_0, t)
        std = torch.sqrt(1-alpha_bar_t)

        # sample from q using the reparameterization trick
        eps_q = torch.randn_like(x_0)
        x_t = mu+std*eps_q

        return x_t, eps_q
    
    @torch.inference_mode()
    def reverse(
            self,
            x_t: torch.FloatTensor, # (batch_size, *data_shape),
            t: int,
            eps_theta: torch.FloatTensor, # (batch_size, *data_shape),
            ):
        """
        for a batch of corrupted data x_t and using the estimated noise eps_theta, 
        sample from p(x_{t-1}|x_t, t)
        
        Here, t is the same for all samples in the batch.

        Returns:
        x_t_1: torch.FloatTensor, a single-step denoised batch of data
        """

        alpha_t=self.alpha[t].view(1, *([1]*(x_t.dim()-1)))
        alpha_bar_t=self.alpha_bar[t].view(1, *([1]*(x_t.dim()-1)))
        beta_t=self.beta[t].view(1, *([1]*(x_t.dim()-1)))

        # mean of p(x_{t-1}|x_t, t)
        mu = (1/torch.sqrt(alpha_t))*(x_t-(beta_t/torch.sqrt(1-alpha_bar_t))*eps_theta)

        # std of p(x_{t-1}|x_t, t)
        std = torch.sqrt(beta_t)

        # sample from p using the reparameterization trick
        # NOTE: no noise is added at the final denoising step (t=0 -> eps_p=0)
        eps_p = torch.zeros_like(x_t) if t==0 else torch.randn_like(x_t)
        x_t_1 = mu+std*eps_p

        return x_t_1