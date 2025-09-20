import torch
from torch import nn
from torch.nn import functional as F

def D_real_loss_fn(
        D_real: torch.FloatTensor, # (batch_size, 1)
        ) -> torch.FloatTensor: # ()
    """
    D_real is D(x), the discriminator's output when fed with real images
    We want this to be close to 1, because the discriminator should recognize real images
    """

    target=torch.ones_like(D_real)#*0.99
    return F.binary_cross_entropy(D_real, target)


def D_fake_loss_fn(
        D_fake: torch.FloatTensor, # (batch_size, 1)
        ) -> torch.FloatTensor: # ()
    """
    D_fake is D(G(z)), the discriminator's output when fed with generated images
    We want this to be close to 0, because the discriminator should not be fooled
    """
    # return F.binary_cross_entropy(D_fake, torch.zeros_like(D_fake))
    # target=torch.ones_like(D_fake)*0.01
    target=torch.zeros_like(D_fake)
    return F.binary_cross_entropy(D_fake, target)


def G_loss_fn(
        D_fake: torch.FloatTensor, # (batch_size, 1)
        ) -> torch.FloatTensor: # ()
    """
    D_fake is D(G(z)), the discriminator's output when fed with generated images
    We want this to be close to 1, because the generator wants to fool the discriminator
    """
    target=torch.ones_like(D_fake)#*0.99
    return F.binary_cross_entropy(D_fake, target)

def D_KL(
        mu: torch.FloatTensor, # shape (batch_size, latent_size),
        logvar: torch.FloatTensor, # shape (batch_size, latent_size),
        ) -> torch.FloatTensor: # shape ()
    """
    Compute the KL divergence that you derived earlier, elementwise.
    Then, average over the batch dimension and latent dimension.

    mu: mean of q(z|x)
    logvar: Logarithm of variance of q(z|x).
    """
    # NotImplemented
    var=torch.exp(logvar)
    kl=torch.mean(var+mu**2-1-logvar)/2
    # kl=torch.mean(torch.sum((var+mu**2-1-logvar)/2, dim=-1))

    return kl

# loss.py
def vae_loss(x_recon, x_real, mu, logvar, step, kl_max=1e-2, warmup=10000, sigma=0.1):
    # recon term ~ NLL of Gaussian with fixed variance
    w_recon = 1.0 / (2.0 * (sigma ** 2))   # e.g. sigma=0.1 -> 50
    recon = F.mse_loss(x_recon, x_real) * w_recon

    # KL with linear warmup
    beta = min(kl_max, kl_max * (step / warmup))
    var  = torch.exp(logvar)
    kl   = 0.5 * torch.mean(var + mu**2 - 1.0 - logvar)
    return recon + beta * kl, dict(recon=recon.detach(), kl=(beta*kl).detach(), beta=beta)

def total_variation(x):
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w
