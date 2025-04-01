import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from typing import Optional, Union, Tuple

from ..models.unet import UNet

class DDPM(nn.Module):
    def __init__(
        self,
        model: UNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.device = device

        # Register buffers for noise schedule parameters
        self.register_buffer("betas", self.get_betas(beta_start, beta_end, beta_schedule))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1. - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / self.alphas_cumprod - 1))

        # Calculate q(x_{t-1} | x_t, x_0) variance
        self.register_buffer("posterior_variance", (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        ))
        # Clipped because posterior variance is 0 at beginning of diffusion chain
        self.register_buffer("posterior_log_variance_clipped", torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        ))
        self.register_buffer("posterior_mean_coef1", (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        ))
        self.register_buffer("posterior_mean_coef2", (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        ))

    def get_betas(self, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
        """Get the noise schedule for the diffusion process."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.timesteps, device=self.device)
        elif schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            s = 0.008
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps, device=self.device)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_t | x_0) (forward diffusion process)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract coefficients for given timestep t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t) (reverse diffusion process)."""
        pred_noise = self.model(x, t)
        
        # Calculate x_0 from x_t and predicted noise
        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        x0 = sqrt_recip_alphas_cumprod_t * x - sqrt_recipm1_alphas_cumprod_t * pred_noise
        
        if clip_denoised:
            x0 = torch.clamp(x0, -self.clip_sample_range, self.clip_sample_range)
        
        # Calculate mean and variance of q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * x0 +
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x.shape)
        
        # No noise when t == 0
        noise = torch.randn_like(x) if t_index > 0 else torch.zeros_like(x)
        
        return posterior_mean + noise * torch.exp(0.5 * posterior_log_variance_clipped)

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        return_all_timesteps: bool = False,
    ) -> Union[torch.Tensor, list]:
        """Generate samples from noise by iteratively applying the reverse process."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        imgs = [img]
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            imgs.append(img)
        
        return torch.stack(imgs) if return_all_timesteps else img

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        channels: int = 3,
        height: int = 64,
        width: int = 64,
        return_all_timesteps: bool = False,
    ) -> Union[torch.Tensor, list]:
        """Generate samples from the model."""
        return self.p_sample_loop(
            shape=(batch_size, channels, height, width),
            return_all_timesteps=return_all_timesteps,
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate loss for given x_start and timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy, noise_pred = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        # Simple MSE loss between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def forward(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        return self.p_losses(x, t, noise)