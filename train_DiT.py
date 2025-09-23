import torch
import torchvision
from torch import nn, optim

from tqdm import tqdm

from dataloader import *
from diffusion import *
from transformer import *
from utils import *

from diffusers.models import AutoencoderKL

@torch.enable_grad()
def train_via_epoch(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device='cuda',
        plot_freq: int=100,

        optimizer_name: str='Adam',
        optimizer_config: dict=dict(),
        lr_scheduler_name: Union[str, None]=None,
        lr_scheduler_config: dict=dict(),

        epochs: int=5000,
        batch_size: int=32,
        ):

    model.train().to(device)
    
    tracker=Diffusion_Tracker(
        n_iters=epochs, 
        plot_freq=plot_freq,
        )

    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer: torch.optim.Optimizer=torch.optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_config)
    if lr_scheduler_name is not None:
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler=torch.optim.lr_scheduler.__getattribute__(lr_scheduler_name)(optimizer, **lr_scheduler_config)

    iter_pbar=tqdm(range(epochs), desc='Iters', unit='iter', leave=True)

    for epoch in range(epochs):
        loss=train_one_epoch(train_loader, model, optimizer, lr_scheduler)

        iter_pbar.update(1)
        iter_pbar.set_postfix_str(f'loss: {loss.item():.6f}')
        if epoch%plot_freq==0:
            gen_samples=model.generate(1)[-1]
            tracker.get_samples(gen_samples)
        tracker.update(loss.item())

def train_one_epoch(train_loader, model, optimizer, lr_scheduler=None, device='cuda'):
    for x in train_loader:
        model.train()

        x=x.to(device)
        
        t=torch.randint(0, model.diffusion.T, (x.shape[0],), device=device)

        x_t, eps_q=model.diffusion.forward(x, t)

        eps_theta=model(x_t, t)

        loss=F.mse_loss(eps_theta, eps_q)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if type(lr_scheduler).__name__=='ReduceLROnPlateau':
            lr_scheduler.step(loss.item())
        elif type(lr_scheduler).__name__ is not None:
            lr_scheduler.step()
        
    return loss
    
@torch.enable_grad()
def train_via_iter(
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device='cuda',
        plot_freq: int=100,

        optimizer_name: str='Adam',
        optimizer_config: dict=dict(),
        lr_scheduler_name: Union[str, None]=None,
        lr_scheduler_config: dict=dict(),

        n_iters: int=5000,
        batch_size: int=32,
        num_gen: int=5,
        config_string: str='default'
        ):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
    vae.eval()
    
    model.train().to(device)
    
    tracker=Diffusion_Tracker(
        n_iters=n_iters, 
        plot_freq=plot_freq,
        num_gen=num_gen,
        config_string=config_string
        )

    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer: torch.optim.Optimizer=torch.optim.__getattribute__(optimizer_name)(model.parameters(), **optimizer_config)
    if lr_scheduler_name is not None:
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler=torch.optim.lr_scheduler.__getattribute__(lr_scheduler_name)(optimizer, **lr_scheduler_config)

    iter_pbar=tqdm(range(n_iters), desc='Iters', unit='iter', leave=True)
    iters=0

    while iters<n_iters:
        for x, y in train_loader:
            model.train()

            x=x.to(device)
            
            t=torch.randint(0, model.diffusion.T, (x.shape[0],), device=device)

            x_t, eps_q=model.diffusion.forward(x, t)

            eps_theta=model(x_t, t)

            loss=F.mse_loss(eps_theta, eps_q)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if lr_scheduler_name=='ReduceLROnPlateau':
                lr_scheduler.step(loss.item())
            elif lr_scheduler_name is not None:
                lr_scheduler.step()

            iters+=1
            iter_pbar.update(1)
            iter_pbar.set_postfix_str(f'loss: {loss.item():.6f}')
            if iters%plot_freq==0:
                gen_samples=vae.decode(model.generate(num_gen)[-1].detach().cpu() / 0.18215).sample
                tracker.get_samples(gen_samples, y[:num_gen])
                del gen_samples
            tracker.update(loss.item())
            if iters>=n_iters:
                break
            
    tracker.close()
                
if __name__=="__main__":
    DATA_DIR="data"  
    
    img_size=(32, 32)

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
    
    model=DiT(
        input_size=img_size, 
        patch_size=4,
        num_layers=12, 
        embed_dim=128, 
        num_heads=8,
        T=100, 
        b_0=1e-4,
        b_T=2e-2,
        hidden_dim=128,
        in_chans=3, 
        schedule_type='linear'
        )

    import pdb; pdb.set_trace()


    
    train_via_iter(
        model, 
        train_dataset, 
        n_iters=4000, 
        plot_freq=100, 
        optimizer_name='AdamW', 
        optimizer_config={"lr": 1e-5, "weight_decay": 1e-6},
        lr_scheduler_name='CosineAnnealingLR',
        lr_scheduler_config={"T_max": 5000},
        batch_size=16
        )

    gen_samples=model.generate(3)
    plot_reverse_diffusion(model, gen_samples, n_steps=10)
    
    
    import pdb; pdb.set_trace()