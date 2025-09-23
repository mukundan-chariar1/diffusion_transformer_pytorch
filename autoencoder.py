from typing import Sequence, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import PatchEmbedding

class ResBlock(nn.Module):
    def __init__(self, in_channels: int=64, out_channels: int=32, stride: int=1, activation: str='ReLU', batchnorm: bool=True):
        super().__init__()
        act=nn.__getattribute__(activation)
        self.batchnorm=batchnorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        if self.batchnorm: self.bn1   = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if self.batchnorm: self.bn2   = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.act=act()
        
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x)

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)

        out += shortcut
        out = self.act(out)
        return out
    
class UpResBlock(nn.Module):
    def __init__(self, in_channels: int=64, out_channels: int=32, activation: str='ReLU', batchnorm: bool=True):
        super().__init__()
        self.batchnorm=batchnorm
        act=nn.__getattribute__(activation)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        if self.batchnorm: self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.batchnorm: self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.skip = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.act=act()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)

        out = self.deconv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)

        out += shortcut
        out = self.act(out)
        return out
    
class LatentEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, img_shape: tuple = (256, 256), base: int = 64, z_ch: int = 4, activation: str='ReLU', num_layers: int=2, variational: bool=False):
        super().__init__()
        act=nn.__getattribute__(activation)
        self.variational=variational
        ratio = img_shape[0] // 32
        assert (ratio & (ratio - 1)) == 0, "img_shape/32 must be a power of 2"

        layers = [nn.Conv2d(in_ch, base, 3, padding=1), act()]
        ch = base
        for _ in range(int(math.log2(ratio))):
            layers += [
                ResBlock(ch, ch//2, stride=2, activation=activation, batchnorm=True),
                *[ResBlock(ch//2, ch//2, stride=1, activation=activation, batchnorm=True)]*num_layers,
            ]
            ch //= 2
        
        self.body = nn.Sequential(*layers)
        self.mu_net = nn.Conv2d(ch, z_ch, 1)
        if variational: self.logvar_net = nn.Conv2d(ch, z_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.body(x)
        if self.variational:
            return self.mu_net(x), self.logvar_net(x) 
        else: 
            return self.mu_net(x)

class LatentDecoder(nn.Module):
    def __init__(self, out_ch: int = 3, img_shape: tuple = (256, 256), base: int = 64, z_ch: int = 4,
                activation: str='ReLU', num_layers: int=2):
        super().__init__()
        act=nn.__getattribute__(activation)
        ratio = img_shape[0] // 32

        n_up = int(math.log2(ratio))
        start_ch=base // (2 ** n_up)
        ch_schedule=[start_ch//2]
        ch_schedule.extend([start_ch * (2 ** i) for i in range(n_up)])
        
        self.from_z = nn.Sequential(nn.Conv2d(z_ch, ch_schedule[0], 1), act())

        dec = []
        for i in range(n_up):
            c_in = ch_schedule[i]
            c_out = ch_schedule[i + 1]
            dec += [
                UpResBlock(c_in, c_out, activation, batchnorm=False),
                *[ResBlock(c_out, c_out, stride=1, activation=activation, batchnorm=False)]*num_layers
            ]
        dec += [nn.Conv2d(ch_schedule[-1], out_ch, kernel_size=3, padding=1)]
        self.body = nn.Sequential(*dec)
        
        self.tanh=nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_z(z)
        x = self.body(x)
        x=self.tanh(x)
        return x

class ResNetVAE(nn.Module):
    def __init__(self, img_shape: tuple=(256, 256), latent_shape: tuple=(32, 32), in_channels: int=3, latent_channels: int=4, activation: str='ReLU', base: int=64, num_layers: int=2):
        super().__init__()
        self.img_shape=img_shape
        self.latent_shape=latent_shape
        self.in_channels=in_channels
        self.latent_channels=latent_channels
        
        self.latent_size=(self.latent_channels, *self.latent_shape)
        
        self.encoder=LatentEncoder(in_channels, img_shape, z_ch=latent_channels, activation=activation, base=base, num_layers=num_layers, variational=True)
        self.decoder=LatentDecoder(in_channels, img_shape, z_ch=latent_channels, activation=activation, base=base, num_layers=num_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, n_samples: int=40) -> torch.Tensor:
        z = torch.randn(n_samples, *(self.latent_channels, *self.latent_shape))
        return self.decoder(z)
    
class ResNetAE(nn.Module):
    def __init__(self, img_shape: tuple=(256, 256), latent_shape: tuple=(32, 32), in_channels: int=3, latent_channels: int=4, activation: str='ReLU', base: int=64, num_layers: int=2):
        super().__init__()
        self.img_shape=img_shape
        self.latent_shape=latent_shape
        self.in_channels=in_channels
        self.latent_channels=latent_channels
        
        self.latent_size=(self.latent_channels, *self.latent_shape)
        
        self.encoder=LatentEncoder(in_channels, img_shape, z_ch=latent_channels, activation=activation, base=base, num_layers=num_layers, variational=False)
        self.decoder=LatentDecoder(in_channels, img_shape, z_ch=latent_channels, activation=activation, base=base, num_layers=num_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decoder(z)

    def generate(self, n_samples: int=40) -> torch.Tensor:
        z = torch.randn(n_samples, *(self.latent_channels, *self.latent_shape))
        return self.decoder(z)
    
class Discriminator(nn.Module):

    def __init__(
            self,
            input_size: tuple=(256, 256),
            in_channels: int=3,
            patch_size: int=16,
            hidden_sizes: Sequence[int]=[256, 128],
            batchnorm: bool=True,
            activation: str='ReLU',
            ):
        super().__init__()

        act=nn.__getattribute__(activation)
        
        layers=[PatchEmbedding(input_size, in_channels, hidden_sizes[0], patch_size)]
        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if batchnorm: layers.append(nn.LayerNorm(out_size))
            layers.append(act())
            layers.append(nn.Dropout(p=0.2))

        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())

        self.layers=nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor,
            ) -> torch.Tensor:
        
        return self.layers(x)