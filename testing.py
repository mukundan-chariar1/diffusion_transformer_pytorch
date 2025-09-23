import torch
import torchvision

from cleanfid import fid
from diffusers.models import AutoencoderKL

import numpy as np
import cv2

import os
import json
import random
from tqdm import tqdm

import shutil

from transformer import *
from dataloader import *

def calculate_fid(config_str: str='8_4_1_100'):
    latent_size=(32, 32)
    latent_channels=4
    
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device chosen: {device}')

    seed=42

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    with open(f"weights/{config_str}/{config_str}.json") as f:
        model_config=json.load(f)
        
    model=DiT(
        input_size=latent_size,
        embed_dim=384,
        b_0=1e-4,
        b_T=2e-2,
        hidden_dim=384,
        in_chans=latent_channels,
        schedule_type='linear',
        **model_config
        )
    
    weights=torch.load(f"weights/{config_str}/{config_str}.pth")
    model.load_state_dict(weights)
    model.eval().to(device)
    
    if not os.path.exists(f"weights/{config_str}/generated"): 
        print('No images generated for this configuration, generating')
        os.mkdir(f"weights/{config_str}/generated")
    
        with torch.no_grad():
            for i in tqdm(range(100)):
                x=vae.decode(model.generate(1)[-1] / 0.18215).sample
                
                img=(x.squeeze().cpu().numpy().transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
                img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"weights/{config_str}/generated/{i:05d}.jpg", img)
    else:
        print('Images already generated for this configuration, moving to testing')
                
    del vae, model
    
    score = fid.compute_fid("test_data", f"weights/{config_str}/generated", mode="clean")
    print("FID:", score)
    
    with open(f"weights/{config_str}/score.json", 'w') as f:
        json.dump({'score': score}, f, indent=2)
        
    print(f'Saved score at weights/{config_str}/score.json')
    
def concat_images(config_str: str='8_4_1_100'):
    images=sorted(os.listdir(f"weights/{config_str}/generated"))
    img_row=[]
    img_columns=[]
    
    for i, img in enumerate(images):
        image=cv2.imread(f"weights/{config_str}/generated/{img}")
        img_row.append(image)
        
        if (i+1)%10==0:
            img_columns.append(np.concatenate(img_row))
            img_row=[]

    img_columns=np.concatenate(img_columns, 1)
    cv2.imwrite(f"weights/{config_str}/generated_img.png", img_columns)

if __name__=='__main__':  
    config_str='2_8_6_1000'  
    calculate_fid(config_str)
    concat_images(config_str)
    
    import pdb; pdb.set_trace()


