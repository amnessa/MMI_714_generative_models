import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from conditional_unet_depth import ConditionalUNetDepth


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """Helper to gather coefficients at batch indices t and reshape to (B, 1, 1, 1).

    Assumes `a` is a 1D buffer on the same device as the model / inputs.
    """
    b = t.shape[0]
    # ensure t is on same device and long
    t = t.to(a.device).long()
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class GaussianDiffusion(nn.Module):
    """Simple DDPM-style Gaussian diffusion wrapper for ConditionalUNetDepth.

    This class handles the forward noising process (q_sample), training loss
    (p_losses), and sampling (reverse process), assuming an epsilon-prediction
    objective on depth values in [-1, 1].
    """

    def __init__(
        self,
        model: Optional[ConditionalUNetDepth] = None,
        *,
        config: DiffusionConfig = DiffusionConfig(),
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.model = model if model is not None else ConditionalUNetDepth()
        self.config = config
        self.device = device

        self.timesteps = config.timesteps

        # Linear beta schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            alphas_cumprod[:-1],
        ])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # useful precomputes
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    # ---------------------------------------------------------------------
    # q(x_t | x_0) - forward process
    # ---------------------------------------------------------------------

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Sample x_t given x_0 and timestep t using the closed form.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise, noise

    # ---------------------------------------------------------------------
    # Training objective
    # ---------------------------------------------------------------------

    def p_losses(
        self,
        batch,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute epsilon-prediction MSE loss for a batch.

        Expects batch dict with keys:
          - "pixel_values": (B, 1, H, W) ground truth depth x0
          - "conditioning": (B, 4, H, W) RGB (3ch) + corrupted depth (1ch)
        """
        x_start = batch["pixel_values"]
        cond = batch["conditioning"]

        b = x_start.shape[0]
        device = x_start.device

        if noise is None:
            noise = torch.randn_like(x_start)

        t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)

        x_noisy, eps = self.q_sample(x_start=x_start, t=t, noise=noise)

        # model predicts noise given noisy depth and conditioning
        model_out = self.model(x_noisy, cond, t)

        loss = torch.mean((eps - model_out) ** 2)
        return loss

    # ---------------------------------------------------------------------
    # Sampling - p(x_0 | x_T, cond)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step x_t -> x_{t-1}."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = torch.sqrt(1.0 / extract(self.alphas, t, x.shape))

        # predict noise
        model_mean = self.model(x, cond, t)

        # DDPM epsilon parameterization: compute mean of posterior of x_{t-1}
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t / sqrt_one_minus_alphas_cumprod_t * model_mean
        )

        # sample noise, but not at last step
        posterior_var_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        nonzero_mask = (t != 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
        return x_prev

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, shape, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate a sample given conditioning RGB+corrupted-depth.

        shape: (B, 1, H, W) for the depth to be generated
        cond : (B, 4, H, W)
        """
        if device is None:
            device = self.device or cond.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, t)
        return img


if __name__ == "__main__":
    # minimal sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNetDepth().to(device)
    diffusion = GaussianDiffusion(model=model).to(device)

    b, h, w = 2, 64, 64
    x0 = torch.randn(b, 1, h, w, device=device)
    cond = torch.randn(b, 4, h, w, device=device)

    batch = {"pixel_values": x0, "conditioning": cond}
    loss = diffusion.p_losses(batch)
    print("loss:", float(loss))

    # quick sampling test
    sample_depth = diffusion.sample(cond=cond, shape=(b, 1, h, w), device=device)
    print("sample shape:", sample_depth.shape)
