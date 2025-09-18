import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from typing import Sequence, Tuple

class ResBlock(nn.Module):

    def __init__(self, in_channels, growth_factor):
        super().__init__()
        self.conv1=nn.Sequential(nn.BatchNorm2d(in_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, 4*growth_factor, 1, 1, 0, bias=False))

        self.conv2=nn.Sequential(nn.BatchNorm2d(4*growth_factor),
                                 nn.ReLU(),
                                 nn.Conv2d(4*growth_factor, growth_factor, 3, 1, 1, bias=False))
    def forward(self, x):
        out=x+self.conv2(self.conv1(x))

        return out

class Encoder(nn.Module):
    """
    Probabilistic encoder. Output: mu and logvar of q(z|x).
    """
    def __init__(
            self,
            input_size: int,
            latent_size: int,
            hidden_sizes: Sequence[int],
            activation: str = 'ReLU',
            ):
        super().__init__()

        act=nn.__getattribute__(activation)

        layers=[nn.Linear(input_size, hidden_sizes[0])]
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act())
            layers.append(nn.Dropout(p=0.2))

        self.layers=nn.Sequential(*layers[:-2])
        # self.layers=nn.Sequential(*layers[:-1])
        # self.fc_mu=nn.Linear(hidden_sizes[-1], latent_size)
        # self.fc_logvar=nn.Linear(hidden_sizes[-1], latent_size)

        self.fc=nn.Linear(hidden_sizes[-1], latent_size*2)

    def forward(
            self, 
            y: torch.FloatTensor, # shape (batch_size, input_size),
            ) -> Tuple[torch.FloatTensor, torch.FloatTensor]: # mu and logvar of q(z|x), both of shape (batch_size, latent_size)
        
        # mu=self.fc_mu(self.layers(y))
        # logvar=self.fc_logvar(self.layers(y))

        mu_logvar=self.fc(self.layers(y))

        mu, logvar=torch.chunk(mu_logvar, 2, -1)

        return (mu, logvar)
    

class Decoder(nn.Module):
    """
    Treat this as a normal decoder.
    """
    def __init__(
            self,
            latent_size: int,
            output_size: int,
            hidden_sizes: Sequence[int],
            activation: str = 'ReLU',
            ):
        super().__init__()

        act=nn.__getattribute__(activation)

        layers=[nn.Linear(latent_size, hidden_sizes[0])]
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(act())
            layers.append(nn.Dropout(p=0.2))

        # layers=layers[:-2]
        layers=layers[:-1]
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers=nn.Sequential(*layers)
        self.latent_size=latent_size


    def forward(
            self, 
            z: torch.FloatTensor, # shape (batch_size, latent_size),
            ) -> torch.FloatTensor: # y_hat of shape (batch_size, output_size)
        
        return self.layers(z)
    

class VAE(nn.Module):

    def __init__(
            self,
            input_size: int,
            latent_size: int,
            hidden_sizes_encoder: Sequence[int],
            hidden_sizes_decoder: Sequence[int],
            activation: str = 'ReLU',
            ):
        super().__init__()
        """
        use the encoder and decoder classes you defined above, like:
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        """

        self.encoder=Encoder(input_size, latent_size, hidden_sizes_encoder, activation)
        self.decoder=Decoder(latent_size, input_size, hidden_sizes_decoder, activation)
        self.apply(init_weights)

    def forward(
            self, 
            y: torch.FloatTensor, # shape (batch_size, input_size),
            ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: # y_hat, mu, logvar
        """
        - forward pass of the encoder, get mu and logvar
        - sample z from the output of the encoder
        - forward pass of the decoder to get y_hat (reconstruction)
        return y_hat, mu, logvar
        """
        mu, logvar=self.encoder(y)
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        z=mu+eps*std

        y_hat=self.decoder(z)

        return y_hat, mu, logvar
    
    @torch.inference_mode()
    def generate(
            self,
            n_samples: int,
            seed: int = 0,
            device: str = 'cuda',
            ) -> torch.FloatTensor: # shape (n_samples, input_size)
        
        torch.manual_seed(seed)
        """
        Set the decoder to evaluation mode and move it to the device.
        sample from p(z) with the correct shape and device and dtype
        decode them to generate new samples
        """
        
        self.decoder.to(device)
        self.decoder.eval()

        z=torch.randn(n_samples, self.decoder.latent_size, device=device)

        y_hat=self.decoder(z)
        return y_hat
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)