import torch
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler

import lpips

from torchsummary import summary

from tqdm import tqdm

from dataloader import *
from diffusion import *
from transformer import *
from autoencoder import *
from loss import *
from utils import *

def train_GAN(
    generator,
    discriminator,
    train_dataset,
    device='cuda',
    plot_freq=100,
    optimizer_name_G="Adam",
    optimizer_config_G=dict(lr=1e-3),
    lr_scheduler_name_G=None,
    lr_scheduler_config_G=dict(),
    optimizer_name_D="Adam",
    optimizer_config_D=dict(lr=1e-3),
    lr_scheduler_name_D=None,
    lr_scheduler_config_D=dict(),
    n_iters=10000,
    batch_size=64,
    recon_weight=1.0,
    kl_weight=0.01,
    adv_weight=0.001,
    lpips_weight=1,
    beta_schedule: bool=True
):
    generator.to(device)
    discriminator.to(device)
    tracker = VAEGAN_Tracker(n_iters, plot_freq)

    optimizer_G = getattr(optim, optimizer_name_G)(generator.parameters(), **optimizer_config_G)
    optimizer_D = getattr(optim, optimizer_name_D)(discriminator.parameters(), **optimizer_config_D)

    scheduler_G = getattr(lr_scheduler, lr_scheduler_name_G)(optimizer_G, **lr_scheduler_config_G) if lr_scheduler_name_G else None
    scheduler_D = getattr(lr_scheduler, lr_scheduler_name_D)(optimizer_D, **lr_scheduler_config_D) if lr_scheduler_name_D else None

    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    iter_pbar = tqdm(range(n_iters), desc="Training", unit="iter")
    iter = 0
    
    lpips_fn=lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    if beta_schedule: kl_weights=torch.linspace(0, kl_weight, n_iters//1000)
    else: kl_weights=[kl_weight]*(n_iters//1000)

    while iter < n_iters:
        for x_real in train_loader:
            if iter >= n_iters:
                break

            x_real = x_real.to(device)
            
            generator.eval()
            discriminator.train()

            optimizer_D.zero_grad()
            d_real = discriminator(x_real)
            d_real_loss = D_real_loss_fn(d_real)

            with torch.no_grad():
                x_recon, mu, logvar = generator(x_real)
                z = torch.randn(x_real.size(0), *(generator.latent_channels, *generator.latent_shape), device=device)
                x_fake = generator.decode(z)

            d_fake_recon = discriminator(x_recon.detach())
            d_fake_gen = discriminator(x_fake.detach())
            d_fake_loss = (D_fake_loss_fn(d_fake_recon) + D_fake_loss_fn(d_fake_gen)) / 2

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if scheduler_D:
                scheduler_D.step()
                
            generator.train()
            discriminator.eval()

            optimizer_G.zero_grad()
            x_recon, mu, logvar = generator(x_real)
            recon_loss = F.mse_loss(x_recon, x_real)
            kl_loss = D_KL(mu, logvar)
            lpips_loss=lpips_fn(x_recon, x_real).mean()
            d_recon = discriminator(x_recon)
            adv_loss = G_loss_fn(d_recon)

            total_loss = recon_weight * recon_loss + kl_weights[iter//1000] * kl_loss + adv_weight * adv_loss + lpips_weight * lpips_loss
            total_loss.backward()
            optimizer_G.step()

            if scheduler_G:
                scheduler_G.step()

            with torch.no_grad():
                z_sample = torch.randn(40, *generator.latent_size, device=device)
                gen_samples = generator.decode(z_sample)

                idx = np.random.randint(0, x_real.size(0))
                sample_real = x_real[idx]
                sample_recon = x_recon[idx]
                sample_z = mu[idx]

            tracker.update_vaegan(
                real_score=d_real.mean().item(),
                fake_score=d_fake_recon.mean().item(),
                D_loss=d_loss.item(),
                G_loss=total_loss.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_loss.item(),
                lpips_loss=lpips_loss.item(),
                x_real=sample_real.unsqueeze(0),
                x_recon=sample_recon.unsqueeze(0),
            )

            if iter % plot_freq == 0:
                tracker.get_samples(gen_samples)
                
            iter += 1
            iter_pbar.update(1)
            iter_pbar.set_postfix_str(f'loss: {total_loss.item():.6f}')
            
    tracker.close()
            
@torch.enable_grad()
def train_VAE(
        model,
        train_dataset,
        device = 'cuda',
        plot_freq: int = 100,
        alpha=100000, 
        beta: float = 1.,
        gamma=0.5,
        rec_loss_fn: nn.Module = nn.MSELoss(),
        optimizer_name: str = 'Adam',
        optimizer_config: dict = dict(),
        lr_scheduler_name: Union[str, None] = None,
        lr_scheduler_config: dict = dict(),
        running_avg_window: int = 20,
        n_iters: int = 1000,
        batch_size: int = 64,
        variational: bool=False,
        eps: float=1e-4
        ):

    assert beta >= 0
    assert alpha>=0
    assert gamma>=0
    
    model.train().to(device)

    tracker = VAE_Tracker(
        n_iters = n_iters, 
        plot_freq = plot_freq,
        )
    
    lpips_fn=lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.__getattribute__(optimizer_name)(**optimizer_config)
    if lr_scheduler_name is not None:
        scheduler = lr_scheduler.__getattribute__(lr_scheduler_name)(optimizer, **lr_scheduler_config)

    iter = 0
    iter_pbar = tqdm(range(n_iters), desc='Iters', unit='iter', leave=True)

    while iter < n_iters:
        for y in train_loader:
            y = y.to(device)
            optimizer.zero_grad()
            
            if variational:
                y_hat, mu, logvar = model(y)
                rec_loss = rec_loss_fn(y_hat, y)
                prior_loss = D_KL(mu, logvar)
                lpips_loss=lpips_fn(y_hat, y).mean()
                loss = alpha*rec_loss+beta*prior_loss+gamma*lpips_loss

                loss.backward()
                optimizer.step()

                tracker.update(rec_loss.item(), prior_loss.item(), loss.item(), lpips_loss.item())
            else: 
                y_hat=model(y)
                rec_loss = rec_loss_fn(y_hat, y)
                tracker.update(rec_loss.item(), 0, rec_loss.item(), 0)
            
            running_avg_loss = np.mean(tracker.total_losses[-running_avg_window:])
        
            if lr_scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(running_avg_loss)
            elif lr_scheduler_name is not None:
                scheduler.step()
                
            if iter % plot_freq == 0:
                with torch.no_grad():
                    z_sample = torch.randn(40, *model.latent_size, device=device)
                    gen_samples = model.decode(z_sample)
                tracker.get_samples(gen_samples)

            if variational: iter_pbar.set_postfix_str(f'L_rec: {rec_loss.item():.6f}, L_prior: {prior_loss.item():.6f}, L_total: {loss.item():.6f}, LPIPS: {lpips_loss.item():.6f}')
            else: iter_pbar.set_postfix_str(f'L_rec: {rec_loss.item():.6f}')
            iter_pbar.update(1)
            iter += 1
            if iter >= n_iters:
                break
            if iter%100==0:
                beta=beta+eps
            
    tracker.close()
            
            
if __name__=="__main__":
    DATA_DIR="data"  
    
    img_size=(64, 64)

    train_transforms=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    train_dataset=ImageDataset(DATA_DIR, train_transforms)
    
    # Sanity check
    sanity_check(ImageDataset(DATA_DIR, 
                                   transforms=torchvision.transforms.Compose([
                                          torchvision.transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                                          torchvision.transforms.ToTensor(),])),)
    
    generator=ResNetVAE(img_shape=img_size, activation='LeakyReLU', base=32, num_layers=1).to('cuda')
    discriminator=Discriminator(input_size=img_size, hidden_sizes=[64, 64, 64], patch_size=8).to('cuda')
    
    summary(generator, (3, *img_size))
    summary(discriminator, (3, *img_size))

    import pdb; pdb.set_trace()
    
    
    train_GAN(
            generator, 
            discriminator, 
            train_dataset, 
            n_iters=10000, 
            
            # Generator
            optimizer_name_G = 'Adam',
            optimizer_config_G = dict(lr=1e-3, weight_decay=1e-5,),
            lr_scheduler_name_G = None, #'CosineAnnealingLR',
            lr_scheduler_config_G = dict(T_max=50000, eta_min=1e-5, last_epoch=-1),

            # Discriminator
            optimizer_name_D = 'Adam',
            optimizer_config_D = dict(lr=5e-4, weight_decay=1e-5,),
            lr_scheduler_name_D = None, #'CosineAnnealingLR',
            lr_scheduler_config_D = dict(T_max=50000, eta_min=1e-5, last_epoch=-1),
            beta_schedule=False)
    
    train_VAE(generator, 
              train_dataset, 
              n_iters=500, 
              beta=1,
              alpha=1,
              gamma=1, 
              optimizer_name='Adam', 
              optimizer_config={
                                    "params": [
                                        {"params": generator.encoder.parameters(), "lr": 2e-3},
                                        {"params": generator.decoder.parameters(), "lr": 2e-3},
                                    ],
                                    "betas": (0.9, 0.999),
                                    "weight_decay": 0.01,
                                },
              lr_scheduler_name = 'ReduceLROnPlateau',
              lr_scheduler_config = dict(mode='min', factor=0.1, patience=10),
              rec_loss_fn=nn.MSELoss(reduction='mean'))
    
    z_sample = torch.randn(25, *generator.latent_size, device='cuda')
    gen_samples = generator.decode(z_sample)
    
    sanity_check_tensor(unnormalize(gen_samples))
    
    import pdb; pdb.set_trace()