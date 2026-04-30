"""Boundary-aware noise scheduler for trajectory diffusion.

Implements the forward (q-sample) process of a DDPM with either a cosine or
linear ``beta`` schedule. The schedule is *clamped* so that the cumulative
``alpha_bar`` never collapses below a minimum signal-retention level. This
ensures that even at the noisiest timestep, the start- and goal-position
boundary conditions remain recoverable through the reverse process.

Math (see ``docs/math.md``):

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    alpha_bar_t  = clamp(schedule(t), min_alpha, 1.0)
    beta_t       = 1 - alpha_bar_t / alpha_bar_{t-1}    (clamped to (1e-8, 0.999))
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BoundaryAwareNoiseScheduler(nn.Module):
    """DDPM noise schedule with bounded ``alpha_bar`` for boundary preservation.

    Parameters
    ----------
    T:
        Number of diffusion steps.
    schedule_type:
        ``"cosine"`` (Nichol & Dhariwal, 2021) or ``"linear"``.
    max_noise_ratio:
        Caps the noise-to-signal ratio at the noisiest step. The minimum
        ``alpha_bar`` is set to ``1 / (1 + max_noise_ratio**2)``.
    """

    def __init__(
        self,
        T: int = 1000,
        schedule_type: str = "cosine",
        max_noise_ratio: float = 3.0,
    ) -> None:
        super().__init__()
        self.T = T
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            steps = torch.arange(T, dtype=torch.float32)
            s = 0.008
            alphas_cum = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cum = alphas_cum / alphas_cum[0]
            min_alpha = 1.0 / (1.0 + max_noise_ratio**2)
            alphas_cum = torch.clamp(alphas_cum, min=min_alpha, max=1.0)
        elif schedule_type == "linear":
            betas = torch.linspace(1e-5, 2e-3, T)
            alphas = 1 - betas
            alphas_cum = torch.cumprod(alphas, dim=0)
            alphas_cum = torch.clamp(alphas_cum, min=0.7)
        else:
            raise ValueError(f"Unknown schedule_type={schedule_type!r}")

        # Recover per-step betas from the (clamped) cumulative alphas.
        betas = torch.zeros(T)
        betas[0] = 1 - alphas_cum[0]
        betas[1:] = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)
        alphas = 1 - betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.register_buffer("sqrt_ac", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_1mac", torch.sqrt(1 - alphas_cum))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion step ``q(x_t | x_0)``.

        Parameters
        ----------
        x0:
            Clean trajectory tensor of shape ``[B, Na, 2, T]``.
        t:
            Per-batch timestep indices, shape ``[B]``.
        noise:
            Optional pre-sampled standard-normal noise (same shape as ``x0``).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        if x0.dim() != 4:
            raise ValueError(f"Expected x0 of shape [B,Na,2,T], got {tuple(x0.shape)}")
        a = self.sqrt_ac[t].view(-1, 1, 1, 1)
        am = self.sqrt_1mac[t].view(-1, 1, 1, 1)
        x_noisy = a * x0 + am * noise
        return x_noisy, noise


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for diffusion timestep ``t``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * (torch.arange(half, device=device) / max(half - 1, 1)))
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)
