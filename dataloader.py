from typing import Tuple
import torch
import torchvision

from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

import os

def getMeanStd(dataset, return_sums: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    psum=torch.tensor([0.0, 0.0, 0.0])
    psum_sq=torch.tensor([0.0, 0.0, 0.0])

    for inputs in tqdm(dataset):
        psum+=inputs.sum(axis=[1, 2])
        psum_sq+=(inputs**2).sum(axis=[1, 2])

    count=dataset.__len__()*dataset[0].shape[1]*dataset[0].shape[2]
    total_mean=psum/count
    total_var=(psum_sq/count)-(total_mean**2)
    total_std=torch.sqrt(total_var)

    if return_sums: return psum, psum_sq
    else: return total_mean, total_std

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir=data_dir
        self.transforms=transforms

        self.img_paths=list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]).convert("RGB"))
    
class ImageDatasetCompression(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir=data_dir
        self.transforms=transforms

        self.img_paths=list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image=Image.open(self.img_paths[idx]).convert("RGB")
        return self.transforms(image), self.img_paths[idx][5:-4]
    
class ImageDatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.data_dir=data_dir
        self.transforms=transforms

        self.img_paths=list(map(lambda fname: os.path.join(self.data_dir[-4:], f"{fname[:-4]}.jpg"), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image=Image.open(self.img_paths[idx]).convert("RGB")
        latent=torch.load(f"latent_{self.img_paths[idx][:-4]}.pth")
        return latent, self.transforms(image)
    
    
if __name__=="__main__":
    DATA_DIR="data"  
    
    # train_dataset_unnormalized  =ImageDataset(DATA_DIR, 
    #                                             transforms=torchvision.transforms.Compose([
    #                                                     torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
    #                                                     torchvision.transforms.ToTensor(),
    #     ]))
    
    # mean, std=getMeanStd(train_dataset_unnormalized)
    
    mean=torch.tensor([0.4437, 0.4710, 0.4613])         # calculated from above function call
    std=torch.tensor([0.2680, 0.2537, 0.2975])          # calculated from above function call

    train_transforms=torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
        ])

    train_dataset=ImageDataset(DATA_DIR, train_transforms)
    
    # Sanity check
    r, c=[5, 5]
    fig, ax=plt.subplots(r, c, figsize= (15, 15))

    k=0
    dtl=torch.utils.data.DataLoader(
        dataset=ImageDataset(DATA_DIR, 
                                   transforms=torchvision.transforms.Compose([
                                          torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                                          torchvision.transforms.ToTensor(),])),
        batch_size=64,
        shuffle=True)

    for data in dtl:
        x=data

        for i in range(r):
            for j in range(c):
                img=x[k].numpy().transpose(1, 2, 0)
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    fig.show()
    import pdb; pdb.set_trace()