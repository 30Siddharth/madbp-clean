"""Per-agent 1-D U-Net trajectory denoiser.

Operates on each agent's per-timestep feature stream independently — the
*inter-agent* coupling has already been performed by the axial pre-processor
upstream. A shared context vector ``c`` (from ``ContextEncoder``) plus the
diffusion-time embedding is injected at every residual block via FiLM
modulation.

Shapes:
    feat : [B, Na, T, feat_dim]   — output of the axial pre-processor
    ctx  : [B, ctx_dim]           — env+pose+count+time embedding
    out  : [B, Na, T, state_dim]  — predicted noise (same shape as x_t)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualBlock(nn.Module):
    """1-D residual block with FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, ctx_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.film = nn.Linear(ctx_dim, out_ch * 2)
        self.skip = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = self.norm1(self.conv1(x))
        scale, shift = self.film(ctx).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.dropout(self.act(h))
        h = self.norm2(self.conv2(h))
        return self.act(h + skip)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ctx_dim: int) -> None:
        super().__init__()
        self.res = _ResidualBlock(in_ch, out_ch, ctx_dim)
        self.down = nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, ctx)
        return self.down(h), h


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, ctx_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = _ResidualBlock(out_ch + skip_ch, out_ch, ctx_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        if h.size(-1) != skip.size(-1):
            h = F.interpolate(h, size=skip.size(-1), mode="linear", align_corners=False)
        h = torch.cat([h, skip], dim=1)
        return self.res(h, ctx)


class _MiddleBlock(nn.Module):
    def __init__(self, ch: int, ctx_dim: int) -> None:
        super().__init__()
        self.res1 = _ResidualBlock(ch, ch, ctx_dim)
        self.attn = nn.MultiheadAttention(ch, num_heads=8, batch_first=True)
        self.res2 = _ResidualBlock(ch, ch, ctx_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.res1(x, ctx)
        a = h.permute(0, 2, 1)
        a, _ = self.attn(a, a, a)
        h = h + a.permute(0, 2, 1)
        return self.res2(h, ctx)


class TrajectoryUNet(nn.Module):
    """Per-agent 1-D U-Net denoiser that predicts ``epsilon`` of shape ``[B,Na,T,2]``."""

    def __init__(
        self,
        max_agents: int,
        T: int,
        state_dim: int,
        feat_dim: int,
        hid_dim: int,
        ctx_dim: int,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.T = T
        self.state_dim = state_dim

        self.init_conv = nn.Conv1d(feat_dim, hid_dim, 3, padding=1)
        self.down1 = _DownBlock(hid_dim, hid_dim * 2, ctx_dim)
        self.down2 = _DownBlock(hid_dim * 2, hid_dim * 4, ctx_dim)
        self.down3 = _DownBlock(hid_dim * 4, hid_dim * 8, ctx_dim)
        self.middle = _MiddleBlock(hid_dim * 8, ctx_dim)
        self.up1 = _UpBlock(hid_dim * 8, hid_dim * 8, hid_dim * 4, ctx_dim)
        self.up2 = _UpBlock(hid_dim * 4, hid_dim * 4, hid_dim * 2, ctx_dim)
        self.up3 = _UpBlock(hid_dim * 2, hid_dim * 2, hid_dim, ctx_dim)
        self.final_conv = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, 3, padding=1),
            nn.GroupNorm(min(8, hid_dim), hid_dim),
            nn.SiLU(),
            nn.Conv1d(hid_dim, state_dim, 1),
        )

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, Na, T, D = feat.shape
        # [B,Na,T,D] -> [B*Na, D, T]. A bare ``.view`` here (as in the original
        # v2_2 code) would reinterpret memory and scramble the time/feature axes.
        x = feat.permute(0, 1, 3, 2).reshape(B * Na, D, T)
        ctx_e = ctx.unsqueeze(1).expand(B, Na, -1).reshape(B * Na, -1)

        x = self.init_conv(x)
        x1, s1 = self.down1(x, ctx_e)
        x2, s2 = self.down2(x1, ctx_e)
        x3, s3 = self.down3(x2, ctx_e)
        x = self.middle(x3, ctx_e)
        x = self.up1(x, s3, ctx_e)
        x = self.up2(x, s2, ctx_e)
        x = self.up3(x, s1, ctx_e)
        out = self.final_conv(x)  # [B*Na, state_dim, T]
        return out.view(B, Na, self.state_dim, T).permute(0, 1, 3, 2)
