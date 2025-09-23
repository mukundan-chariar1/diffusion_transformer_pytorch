import torch
import torchvision
from torchsummary import summary
from torch import nn

from diffusers.models import AutoencoderKL

from dataloader import *

import random
import numpy as np

from tqdm import tqdm

if __name__=='__main__':
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device chosen: {device}')

    seed=42
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    DATA_DIR="data"
    img_size=(256, 256)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    train_transforms=torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    
    train_dataset=ImageDatasetCompression(DATA_DIR, train_transforms)
    
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    for x, name in tqdm(train_loader):
        x.to(device)
        
        y=vae.encode(x.to(device)).latent_dist.sample().mul_(0.18215).cpu().detach()
        
        torch.save(y.squeeze(0), f"latent_data/{name[0]}.pth")
        
    # sanity check
    
    test_dataset=train_dataset=ImageDatasetTransformer(DATA_DIR, train_transforms)
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    for y, x in tqdm(test_loader):
        print(y.shape) # should be (1, 4, 32, 32)
        
        break

    import pdb; pdb.set_trace()



