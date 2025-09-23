import torch
import numpy as np
import matplotlib.pyplot as plt

import PIL
from typing import Union

from IPython import get_ipython
from IPython.display import display, clear_output

def sanity_check(dataset):
    r, c=[5, 5]
    fig, ax=plt.subplots(r, c, figsize=(15, 15))

    k=0
    dtl=torch.utils.data.DataLoader(
        dataset=dataset,
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
    
def sanity_check_tensor(x):
    r, c=[5, 5]
    fig, ax=plt.subplots(r, c, figsize=(15, 15))

    k=0
    for i in range(r):
        for j in range(c):
            img=x[k].detach().cpu().numpy().transpose(1, 2, 0)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')
            k+=1

    fig.show()
    
def plot_reverse_diffusion(model, 
        xs: torch.Tensor,
        vae,
        n_steps: int=10,
        ):
    n_samples=xs.shape[1]
    ts=list(range(0, model.diffusion.T+1, model.diffusion.T//n_steps))
    fig, axes=plt.subplots(n_samples, n_steps+1, figsize=(8, 8), sharex=True, sharey=True)
    for sample_idx, ax in enumerate(axes):
        for t_idx, a in enumerate(ax):
            a.axis('off')
            img=vae.decode(xs[ts[t_idx], sample_idx].unsqueeze(0) / 0.18215).sample
            a.imshow(unnormalize(img).squeeze().detach().cpu().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
            if sample_idx==0:
                a.set_title(f't={model.diffusion.T-ts[t_idx]}')
    plt.tight_layout()

    fig.savefig('./generated.png')
    
    plt.show()
    
class Diffusion_Tracker:
    def __init__(
            self, 
            n_iters: int,
            plot_freq: Union[int, None]=None, 
            num_gen: int=5,
            config_string: str="default"
            ):
        
        self.losses=[]

        self.plot=plot_freq is not None
        self.iter=0
        self.n_iters=n_iters
        self.config_string=config_string
		
        if self.plot:
            self.plot_freq=plot_freq
            self.num_gen=num_gen
            self.plot_results()

    def plot_results(self):
        self.fig, self.ax=plt.subplots(figsize=(12, 4))

        self.loss_curve,=self.ax.plot(
            range(1, self.iter+1),
            self.losses,
            )

        self.ax.set_xlim(0, self.n_iters+1)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Diffusion Learning Curve')
        self.ax.grid(linestyle='--')

        self.samples_fig, self.samples_axes=plt.subplots(2, self.num_gen, figsize=(8, 5), sharex=True, sharey=True)
        self.sample_axes=self.samples_axes.flat
        self.samples=[]
        for ax in self.sample_axes:
            ax.axis('off')
            self.samples.append(ax.imshow(np.zeros((256, 256)), cmap='gray', vmin=0, vmax=1))


    def update(
            self, 
            loss: float,
            ):
        self.losses.append(loss)
        self.iter+=1
		
        if self.plot and self.iter%self.plot_freq==0:

            # score plot:
            self.loss_curve.set_data(range(1, self.iter+1), self.losses)
            self.ax.relim()
            self.ax.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter} (top) v/s reals (bottom)')

            self.fig.canvas.draw()
            
            if get_ipython() is not None:
                clear_output(wait=True)
                display(self.fig)
                display(self.samples_fig)

    
    def get_samples(
            self, 
            samples: torch.Tensor, 
            reals: torch.Tensor
            ):
        for sample, sample_img in zip(samples, self.samples[:self.num_gen]):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
        for sample, sample_img in zip(reals, self.samples[self.num_gen:]):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
    def close(self):
        self.fig.savefig(f"weights/{self.config_string}/{self.config_string}_losses.png")
        self.samples_fig.savefig(f"weights/{self.config_string}/{self.config_string}_generated.png")
        
        self.fig.show()
        self.samples_fig.show()
            
            
def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return x.add(1).mul_(0.5).clamp_(0, 1)

class GAN_Tracker:
    def __init__(
            self, 
            n_iters: int,
            plot_freq: Union[int, None] = None,
            ):
        self.real_scores = []
        self.fake_scores = []
        self.D_losses = []
        self.G_losses = []

        self.plot = plot_freq is not None
        self.iter = 0
        self.n_iters = n_iters
		
        if self.plot:
            self.plot_freq = plot_freq
            self.plot_results()


    def plot_results(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(13, 3), sharex=True)

        self.real_score_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.real_scores,
			label = r'$D(x)$',
            )
        self.fake_score_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.fake_scores,
            label = r'$D(G(z))$',
            )

        self.ax1.set_xlim(0, self.n_iters+1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Discriminator Score')
        self.ax1.set_title('Discriminator Score')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

        self.D_loss_curve, = self.ax2.plot(
            range(1, self.iter+1),
            self.D_losses,
			label = 'D',
            )
        self.G_loss_curve, = self.ax2.plot(
            range(1, self.iter+1),
            self.G_losses,
            label = 'G',
            )
        self.ax2.set_xlim(0, self.n_iters+1)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Learning Curve')
        self.ax2.grid(linestyle='--')
        self.ax2.legend()

        self.samples_fig, self.samples_axes=plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
        self.sample_axes=self.samples_axes.flat
        self.samples=[]
        for ax in self.sample_axes:
            ax.axis('off')
            self.samples.append(ax.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1))

    def update(
            self, 
            real_score: float,
            fake_score: float,
            D_loss: float,
            G_loss: float,
            ):
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)
        self.D_losses.append(D_loss)
        self.G_losses.append(G_loss)
        self.iter += 1
		
        if self.plot and self.iter % self.plot_freq == 0:
            self.real_score_curve.set_data(range(1, self.iter+1), self.real_scores)
            self.fake_score_curve.set_data(range(1, self.iter+1), self.fake_scores)
            self.ax1.relim()
            self.ax1.autoscale_view()

            self.D_loss_curve.set_data(range(1, self.iter+1), self.D_losses)
            self.G_loss_curve.set_data(range(1, self.iter+1), self.G_losses)
            self.ax2.relim()
            self.ax2.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()
            
            if get_ipython() is not None:
                clear_output(wait=True)
                display(self.fig)
                display(self.samples_fig)

    def get_samples(
            self, 
            samples: torch.Tensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))

class VAEGAN_Tracker(GAN_Tracker):
    def __init__(self, n_iters: int, plot_freq: Union[int, None] = None):
        super().__init__(n_iters, plot_freq)
        self.recon_losses = []
        self.kl_losses = []
        self.lpips_losses=[]

        if self.plot:
            self.plot_freq = plot_freq

            self.fig, self.axs = plt.subplots(1, 2, figsize=(18, 5))

            self.ax1 = self.axs[0]
            self.real_score_curve, = self.ax1.plot([], [], label=r'$D(x)$')
            self.fake_score_curve, = self.ax1.plot([], [], label=r'$D(G(z))$')
            self._setup_score_plot()

            self.ax2 = self.axs[1]
            self.D_loss_curve, = self.ax2.plot([], [], label='D Loss')
            self.G_loss_curve, = self.ax2.plot([], [], label='G Loss')
            self.recon_loss_curve, = self.ax2.plot([], [], label='Reconstruction Loss')
            self.kl_loss_curve, = self.ax2.plot([], [], label='KL Loss')
            self.lpips_loss_curve, = self.ax2.plot([], [], label='LPIPS Loss')
            self._setup_loss_plot()

            self.samples_fig, self.samples_axes=plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
            self.sample_axes=self.samples_axes.flat
            self.samples=[]
            for ax in self.sample_axes:
                ax.axis('off')
                self.samples.append(ax.imshow(np.zeros((256, 256, 3)), vmin=0, vmax=1))

    def _setup_score_plot(self):
        self.ax1.set_xlim(0, self.n_iters + 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Discriminator Score')
        self.ax1.set_title('Discriminator Scores')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

    def _setup_loss_plot(self):
        self.ax2.set_xlim(0, self.n_iters + 1)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Training Losses')
        self.ax2.grid(linestyle='--')
        self.ax2.legend()

    def update_vaegan(
            self,
            real_score: float,
            fake_score: float,
            D_loss: float,
            G_loss: float,
            recon_loss: float,
            kl_loss: float,
            lpips_loss: float,
            x_real: torch.Tensor = None,
            x_recon: torch.Tensor = None
    ):
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)
        self.D_losses.append(D_loss)
        self.G_losses.append(G_loss)
        self.recon_losses.append(recon_loss)
        self.kl_losses.append(kl_loss)
        self.lpips_losses.append(lpips_loss)

        self.iter += 1

        if self.plot and self.iter % self.plot_freq == 0:
            self.real_score_curve.set_data(range(1, self.iter + 1), self.real_scores)
            self.fake_score_curve.set_data(range(1, self.iter + 1), self.fake_scores)
            self.ax1.relim()
            self.ax1.autoscale_view()

            self.D_loss_curve.set_data(range(1, self.iter + 1), self.D_losses)
            self.G_loss_curve.set_data(range(1, self.iter + 1), self.G_losses)
            self.recon_loss_curve.set_data(range(1, self.iter + 1), self.recon_losses)
            self.kl_loss_curve.set_data(range(1, self.iter + 1), self.kl_losses)
            self.lpips_loss_curve.set_data(range(1, self.iter+1), self.lpips_losses)
            self.ax2.relim()
            self.ax2.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()
            
            if get_ipython() is not None:
                clear_output(wait=True)
                display(self.fig)
                display(self.samples_fig)

    def get_samples(
            self, 
            samples: torch.Tensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
    def close(self):
        self.fig.show()
        self.samples_fig.show()
            
class VAE_Tracker:
    def __init__(
            self, 
            n_iters: int,
            plot_freq: int = 0,
            ):
        self.rec_losses = []
        self.prior_losses = []
        self.total_losses = []
        self.lpips_loss=[]
        self.plot = plot_freq is not None
        self.iter = 0
        self.n_iters = n_iters
        self.plot_freq = plot_freq
        if self.plot_freq > 0:
            self.plot_results()
            
        self.samples_fig, self.samples_axes=plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
        self.sample_axes=self.samples_axes.flat
        self.samples=[]
        for ax in self.sample_axes:
            ax.axis('off')
            self.samples.append(ax.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1))

    def plot_results(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
        self.rec_loss_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.rec_losses,
			label = 'Rec Loss',
            )
        self.prior_loss_curve, = self.ax2.plot(
            range(1, self.iter+1),
            self.prior_losses,
			label = 'Prior Loss',
            )
        self.total_loss_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.total_losses,
			label = 'Total Loss',
            )
        self.lpips_loss_curve, = self.ax1.plot(
            range(1, self.iter+1),
            self.total_losses,
			label = 'LPIPS Loss',
            )
        self.ax1.set_xlim(0, self.n_iters+1)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Reconstruction and Total Loss Learning Curve')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

        self.ax2.set_xlim(0, self.n_iters+1)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Prior Loss and LPIPS loss Learning Curve')
        self.ax2.grid(linestyle='--')


    def update(
            self, 
            rec_loss: float,
            prior_loss: float,
            total_loss: float,
            lpips_loss: float,
            ):
        self.rec_losses.append(rec_loss)
        self.prior_losses.append(prior_loss)
        self.total_losses.append(total_loss)
        self.lpips_loss.append(lpips_loss)
        self.iter += 1
		
        if self.plot_freq > 0 and self.iter % self.plot_freq == 0:
            self.rec_loss_curve.set_data(range(1, self.iter+1), self.rec_losses)
            self.prior_loss_curve.set_data(range(1, self.iter+1), self.prior_losses)
            self.total_loss_curve.set_data(range(1, self.iter+1), self.total_losses)
            self.lpips_loss_curve.set_data(range(1, self.iter+1), self.lpips_loss)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax1.set_ylim(bottom=0.0, top=None)
            self.ax2.relim()
            self.ax2.autoscale_view()
            plt.tight_layout()
            self.fig.canvas.draw()
            
            if get_ipython() is not None:
                clear_output(wait=True)
                display(self.fig)
                display(self.samples_fig)
            
    def get_samples(
            self, 
            samples: torch.Tensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
    def close(self):
        self.fig.show()
