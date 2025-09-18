import torch
import numpy as np
import matplotlib.pyplot as plt

import PIL
from typing import Union

def sanity_check(dataset):
    r, c    = [5, 5]
    fig, ax = plt.subplots(r, c, figsize= (15, 15))

    k       = 0
    dtl     = torch.utils.data.DataLoader(
        dataset     = dataset,
        batch_size  = 64,
        shuffle     = True)

    for data in dtl:
        x = data

        for i in range(r):
            for j in range(c):
                img = x[k].numpy().transpose(1, 2, 0)
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    fig.show()
    
def plot_reverse_diffusion(model, 
        xs: torch.FloatTensor, # (T+1, n_samples, *data_shape)
        n_steps: int = 10,
        ):
    n_samples = xs.shape[1]
    ts = list(range(0, model.diffusion.T+1, model.diffusion.T//n_steps))
    fig, axes = plt.subplots(n_samples, n_steps+1, figsize=(8, 8), sharex=True, sharey=True)
    for sample_idx, ax in enumerate(axes):
        for t_idx, a in enumerate(ax):
            a.axis('off')
            a.imshow(unnormalize(xs[ts[t_idx], sample_idx]).squeeze().detach().cpu().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            if sample_idx == 0:
                a.set_title(f't={model.diffusion.T-ts[t_idx]}')
    plt.tight_layout()
    plt.show()
    
class Diffusion_Tracker:
    """
    Logs and plots different loss terms of a GAN during training.
    """
    def __init__(
            self, 
            n_iters: int,
            plot_freq: Union[int, None] = None, # plot every plot_freq iterations
            ):
        
        self.losses = []

        self.plot = plot_freq is not None
        self.iter = 0
        self.n_iters = n_iters
		
        if self.plot:
            self.plot_freq = plot_freq
            self.plot_results()


    def plot_results(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))

        # Score plot:
        self.loss_curve, = self.ax.plot(
            range(1, self.iter+1),
            self.losses,
            )

        self.ax.set_xlim(0, self.n_iters+1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Diffusion Learning Curve')
        self.ax.grid(linestyle='--')

        self.samples_fig, self.samples_axes = plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
        self.sample_axes = self.samples_axes.flat
        self.samples = []
        for ax in self.sample_axes:
            ax.axis('off')
            self.samples.append(ax.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1))


    def update(
            self, 
            loss: float,
            ):
        self.losses.append(loss)
        self.iter += 1
		
        if self.plot and self.iter % self.plot_freq == 0:

            # score plot:
            self.loss_curve.set_data(range(1, self.iter+1), self.losses)
            self.ax.relim()
            self.ax.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()

    
    def get_samples(
            self, 
            samples: torch.FloatTensor, # (n_samples, *output_shape)
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
            
def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1], keeps shape (B,C,H,W) or (C,H,W)."""
    return x.add(1).mul_(0.5).clamp_(0, 1)