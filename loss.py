import torch
from torch import nn
from torch.nn import functional as F

def D_real_loss_fn(
        D_real: torch.FloatTensor,
        ) -> torch.FloatTensor:

    target=torch.ones_like(D_real)
    return F.binary_cross_entropy(D_real, target)


def D_fake_loss_fn(
        D_fake: torch.FloatTensor,
        ) -> torch.FloatTensor:
    
    # target=torch.ones_like(D_fake)*0.01
    target=torch.zeros_like(D_fake)
    return F.binary_cross_entropy(D_fake, target)


def G_loss_fn(
        D_fake: torch.FloatTensor,
        ) -> torch.FloatTensor:
    
    target=torch.ones_like(D_fake)#*0.99
    return F.binary_cross_entropy(D_fake, target)

def D_KL(
        mu: torch.FloatTensor,
        logvar: torch.FloatTensor,
        ) -> torch.FloatTensor:
    
    var=torch.exp(logvar)
    kl=torch.mean(var+mu**2-1-logvar)/2
    return kl

