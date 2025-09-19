import torch
import numpy as np
import matplotlib.pyplot as plt

import PIL
from typing import Union

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
    
def sanity_check_tensor(dataset):
    r, c=[5, 5]
    fig, ax=plt.subplots(r, c, figsize=(15, 15))

    k=0
    for data in dataset:
        x=data

        for i in range(r):
            for j in range(c):
                img=x.detach().cpu().numpy().transpose(1, 2, 0)
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                k+=1
        break

    fig.show()
    
def plot_reverse_diffusion(model, 
        xs: torch.FloatTensor,
        n_steps: int=10,
        ):
    n_samples=xs.shape[1]
    ts=list(range(0, model.diffusion.T+1, model.diffusion.T//n_steps))
    fig, axes=plt.subplots(n_samples, n_steps+1, figsize=(8, 8), sharex=True, sharey=True)
    for sample_idx, ax in enumerate(axes):
        for t_idx, a in enumerate(ax):
            a.axis('off')
            a.imshow(unnormalize(xs[ts[t_idx], sample_idx]).squeeze().detach().cpu().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
            if sample_idx==0:
                a.set_title(f't={model.diffusion.T-ts[t_idx]}')
    plt.tight_layout()
    plt.show()
    
class Diffusion_Tracker:
    def __init__(
            self, 
            n_iters: int,
            plot_freq: Union[int, None]=None, 
            ):
        
        self.losses=[]

        self.plot=plot_freq is not None
        self.iter=0
        self.n_iters=n_iters
		
        if self.plot:
            self.plot_freq=plot_freq
            self.plot_results()


    def plot_results(self):
        self.fig, self.ax=plt.subplots(figsize=(12, 4))

        # Score plot:
        self.loss_curve,=self.ax.plot(
            range(1, self.iter+1),
            self.losses,
            )

        self.ax.set_xlim(0, self.n_iters+1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Diffusion Learning Curve')
        self.ax.grid(linestyle='--')

        self.samples_fig, self.samples_axes=plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
        self.sample_axes=self.samples_axes.flat
        self.samples=[]
        for ax in self.sample_axes:
            ax.axis('off')
            self.samples.append(ax.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1))


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

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()

    
    def get_samples(
            self, 
            samples: torch.FloatTensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
            
def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1], keeps shape (B,C,H,W) or (C,H,W)."""
    return x.add(1).mul_(0.5).clamp_(0, 1)

class GAN_Tracker:
    """
    Logs and plots different loss terms of a GAN during training.
    """
    def __init__(
            self, 
            n_iters: int,
            plot_freq: Union[int, None] = None, # plot every plot_freq iterations
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

        # Score plot:
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

        # Loss plot:
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

            # score plot:
            self.real_score_curve.set_data(range(1, self.iter+1), self.real_scores)
            self.fake_score_curve.set_data(range(1, self.iter+1), self.fake_scores)
            self.ax1.relim()
            self.ax1.autoscale_view()

            # loss plot:
            self.D_loss_curve.set_data(range(1, self.iter+1), self.D_losses)
            self.G_loss_curve.set_data(range(1, self.iter+1), self.G_losses)
            self.ax2.relim()
            self.ax2.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()

    
    def get_samples(
            self, 
            samples: torch.FloatTensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))

class VAEGAN_Tracker(GAN_Tracker):
    """
    Extended tracker for VAEGAN training with additional visualizations
    for reconstruction loss, KL divergence, and adversarial loss.
    """

    def __init__(self, n_iters: int, plot_freq: Union[int, None] = None):
        super().__init__(n_iters, plot_freq)

        # Additional VAEGAN metrics
        self.recon_losses = []
        self.kl_losses = []

        if self.plot:
            self.plot_freq = plot_freq

            # Create figure with 3 subplots (GAN loss, Discriminator scores, VAE loss)
            self.fig, self.axs = plt.subplots(1, 2, figsize=(18, 5))

            # Subplot 1: Discriminator Scores
            self.ax1 = self.axs[0]
            self.real_score_curve, = self.ax1.plot([], [], label=r'$D(x)$')
            self.fake_score_curve, = self.ax1.plot([], [], label=r'$D(G(z))$')
            self._setup_score_plot()

            # Subplot 2: GAN Losses (D_loss & G_loss)
            self.ax2 = self.axs[1]
            self.D_loss_curve, = self.ax2.plot([], [], label='D Loss')
            self.G_loss_curve, = self.ax2.plot([], [], label='G Loss')
            self.recon_loss_curve, = self.ax2.plot([], [], label='Reconstruction Loss')
            self.kl_loss_curve, = self.ax2.plot([], [], label='KL Loss')
            self._setup_loss_plot()

            # Sample visualization
            self.samples_fig, self.samples_axes=plt.subplots(5, 8, figsize=(8, 5), sharex=True, sharey=True)
            self.sample_axes=self.samples_axes.flat
            self.samples=[]
            for ax in self.sample_axes:
                ax.axis('off')
                self.samples.append(ax.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1))

    def _setup_score_plot(self):
        """Configures discriminator score subplot."""
        self.ax1.set_xlim(0, self.n_iters + 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Discriminator Score')
        self.ax1.set_title('Discriminator Scores')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

    def _setup_loss_plot(self):
        """Configures loss subplot."""
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
            x_real: torch.Tensor = None,
            x_recon: torch.Tensor = None
    ):
        """
        Update tracker with new training data.
        """
        # Append to lists
        self.real_scores.append(real_score)
        self.fake_scores.append(fake_score)
        self.D_losses.append(D_loss)
        self.G_losses.append(G_loss)
        self.recon_losses.append(recon_loss)
        self.kl_losses.append(kl_loss)

        self.iter += 1

        if self.plot and self.iter % self.plot_freq == 0:
            # Update score plot
            self.real_score_curve.set_data(range(1, self.iter + 1), self.real_scores)
            self.fake_score_curve.set_data(range(1, self.iter + 1), self.fake_scores)
            self.ax1.relim()
            self.ax1.autoscale_view()

            # Update loss plot
            self.D_loss_curve.set_data(range(1, self.iter + 1), self.D_losses)
            self.G_loss_curve.set_data(range(1, self.iter + 1), self.G_losses)
            self.recon_loss_curve.set_data(range(1, self.iter + 1), self.recon_losses)
            self.kl_loss_curve.set_data(range(1, self.iter + 1), self.kl_losses)
            self.ax2.relim()
            self.ax2.autoscale_view()

            self.samples_fig.suptitle(f'Generated Samples at Iteration {self.iter}')

            self.fig.canvas.draw()

    def get_samples(
            self, 
            samples: torch.FloatTensor, 
            ):
        for sample, sample_img in zip(samples, self.samples):
            sample_img.set_data(unnormalize(sample).clip(0, 1).detach().squeeze().cpu().numpy().transpose(1, 2, 0))
            
    def close(self):
        self.fig.show()
        self.samples_fig.show()
            
class VAE_Tracker:
    """
    Logs and plots different loss terms of a VAE during training.
    """
    def __init__(
            self, 
            n_iters: int,
            plot_freq: int = 0, # plot every plot_freq iterations
            ):
        self.rec_losses = []
        self.prior_losses = []
        self.total_losses = []
        self.plot = plot_freq is not None
        self.iter = 0
        self.n_iters = n_iters
        self.plot_freq = plot_freq
        if self.plot_freq > 0:
            self.plot_results()

    def plot_results(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

        # Loss plot:
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
        self.ax1.set_xlim(0, self.n_iters+1)
        # self.ax1.set_ylim(0, 0.002)
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Reconstruction and Total Loss Learning Curve')
        self.ax1.grid(linestyle='--')
        self.ax1.legend()

        self.ax2.set_xlim(0, self.n_iters+1)
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Prior Loss Learning Curve')
        self.ax2.grid(linestyle='--')


    def update(
            self, 
            rec_loss: float,
            prior_loss: float,
            total_loss: float,
            ):
        self.rec_losses.append(rec_loss)
        self.prior_losses.append(prior_loss)
        self.total_losses.append(total_loss)
        self.iter += 1
		
        if self.plot_freq > 0 and self.iter % self.plot_freq == 0:

            # loss plot:
            self.rec_loss_curve.set_data(range(1, self.iter+1), self.rec_losses)
            self.prior_loss_curve.set_data(range(1, self.iter+1), self.prior_losses)
            self.total_loss_curve.set_data(range(1, self.iter+1), self.total_losses)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax1.set_ylim(bottom=0.0, top=None)
            self.ax2.relim()
            self.ax2.autoscale_view()
            plt.tight_layout()
            self.fig.canvas.draw()
            
    def close(self):
        self.fig.show()
