"""``MultiAgentDiffusionPlanner`` — the MA-DBP model.

The model is a conditional latent-trajectory diffusion model:

    eps_theta( x_t ; c )       (network outputs predicted noise)

where the conditioning vector ``c`` is built by ``ContextEncoder`` from
(env-image, start, goal, n_agents) and added to the diffusion-time
embedding. Inter-agent coupling is supplied by ``AxialPreprocessor`` and
the per-agent denoiser is ``TrajectoryUNet``.

Optional **classifier-free guidance** (CFG) is built in: pass
``cfg_dropout_prob > 0`` at construction to randomly drop conditioning
during training, and call ``sample(..., guidance_scale=w)`` at deploy
time. ``guidance_scale=1.0`` (default) recovers the non-CFG model.

Loss weights (denoising / boundary / temporal / velocity / collision /
goal) are passed to ``forward()``; defaults match the paper. The full
``L_total`` formula is the single line at the end of ``forward()``.

P0 fixes vs. arXiv v1 reference code (see ``docs/changes.md``):

  * Train forward now goes through ``preprocessor`` (the v2_2 reference
    silently bypassed it via ``input_proj``).
  * Single canonical ``collision_loss`` (the reference duplicated it).
  * Single, explicit ``L_total`` formula — no progress-gated dead branches.
  * Curriculum injection (``n_agents``) is the trainer's responsibility,
    never lost between trainer and dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from madbp.models import losses
from madbp.models.context import ContextEncoder
from madbp.models.preprocessor import AxialPreprocessor
from madbp.models.scheduler import BoundaryAwareNoiseScheduler, SinusoidalPosEmb
from madbp.models.unet import TrajectoryUNet


@dataclass
class LossWeights:
    """Auxiliary-loss coefficients for ``L_total``."""

    denoising: float = 0.90
    boundary: float = 0.04
    temporal: float = 0.06
    velocity: float = 0.00
    collision: float = 0.00
    goal: float = 0.00


@dataclass
class LossOutput:
    """Per-term loss values for logging."""

    total: torch.Tensor
    denoising: torch.Tensor
    boundary: torch.Tensor
    temporal: torch.Tensor
    velocity: torch.Tensor
    collision: torch.Tensor
    goal: torch.Tensor


class MultiAgentDiffusionPlanner(nn.Module):
    """Single shared multi-agent diffusion planner with optional CFG."""

    def __init__(
        self,
        max_agents: int = 10,
        horizon: int = 8,
        state_dim: int = 2,
        img_ch: int = 3,
        hid: int = 128,
        diffusion_steps: int = 150,
        schedule_type: str = "linear",
        cfg_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.horizon = horizon
        self.state_dim = state_dim
        self.T_steps = diffusion_steps
        self.cfg_dropout_prob = float(cfg_dropout_prob)

        self.scheduler = BoundaryAwareNoiseScheduler(
            T=diffusion_steps,
            schedule_type=schedule_type,
            max_noise_ratio=3.0,
        )
        self.time_emb = SinusoidalPosEmb(hid)
        self.ctx_enc = ContextEncoder(img_ch, pose_dim=state_dim, hid=hid, max_agents=max_agents)
        self.preprocessor = AxialPreprocessor(
            state_dim=state_dim,
            feat_dim=hid,
            num_heads=4,
            max_agents=max_agents,
            T=horizon,
        )
        self.unet = TrajectoryUNet(
            max_agents=max_agents,
            T=horizon,
            state_dim=state_dim,
            feat_dim=hid,
            hid_dim=hid,
            ctx_dim=hid,
        )

        # Learnable null-context for CFG; only used when cfg_dropout_prob > 0.
        self.null_ctx = nn.Parameter(torch.zeros(hid))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _agent_mask(self, n_agents: torch.Tensor) -> torch.Tensor:
        """``True`` where the agent slot is **padded** (key_padding_mask convention)."""
        return torch.arange(self.max_agents, device=n_agents.device)[None, :] >= n_agents[:, None]

    def _build_context(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Produce ``[B, hid]`` context, with optional per-batch CFG dropout."""
        if force_uncond:
            B = frames.size(0)
            return self.null_ctx.unsqueeze(0).expand(B, -1)

        ctx = self.ctx_enc(frames, starts, goals, n_agents)
        if self.training and self.cfg_dropout_prob > 0.0:
            B = ctx.size(0)
            drop = (torch.rand(B, device=ctx.device) < self.cfg_dropout_prob).float().unsqueeze(-1)
            ctx = (1.0 - drop) * ctx + drop * self.null_ctx.unsqueeze(0)
        return ctx

    # ------------------------------------------------------------------
    # Training forward — predicts noise and returns weighted L_total
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor,
        trajs: torch.Tensor,
        loss_weights: LossWeights | None = None,
    ) -> LossOutput:
        """One denoising step on a batch.

        Parameters
        ----------
        frames:
            ``[B, 3, H, W]`` env images.
        starts, goals:
            ``[B, max_agents, 2]`` padded endpoints.
        n_agents:
            ``[B]`` long tensor of active-agent counts (curriculum-controlled).
        trajs:
            ``[B, max_agents, T, 2]`` clean target trajectories.
        loss_weights:
            Optional override of auxiliary-loss coefficients.
        """
        if loss_weights is None:
            loss_weights = LossWeights()
        B = frames.size(0)
        device = frames.device

        # --- Context (with optional CFG dropout) -------------------------
        ctx = self._build_context(frames, starts, goals, n_agents)

        # --- Forward diffusion -------------------------------------------
        t = torch.randint(0, self.T_steps, (B,), device=device)
        x0 = trajs.permute(0, 1, 3, 2)            # [B, Na, 2, T]
        x_noisy_BNCT, noise_BNCT = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy_BNCT.permute(0, 1, 3, 2)  # [B, Na, T, 2]
        target_noise = noise_BNCT.permute(0, 1, 3, 2)

        # --- Denoiser (axial preprocessor IS in the train path) ----------
        agent_mask = self._agent_mask(n_agents)
        feat = self.preprocessor(x_noisy, agent_mask)
        cond = ctx + self.time_emb(t)
        pred_noise = self.unet(feat, cond)        # [B, Na, T, 2]

        # --- Min-SNR-weighted MSE on epsilon, masked to active agents ----
        snr = self.scheduler.alphas_cum[t] / (1.0 - self.scheduler.alphas_cum[t])
        min_snr_gamma = 5.0
        weight = torch.minimum(snr, snr.new_full((), min_snr_gamma)) / snr  # [B]
        w = weight.view(B, 1, 1, 1)
        active = (~agent_mask)[:, :, None, None].expand_as(pred_noise)
        diff = (pred_noise - target_noise) * w
        denoising = (diff[active] ** 2).mean()

        # --- Reconstruct predicted x0 once for all auxiliary losses ------
        sqrt_a = self.scheduler.sqrt_ac[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.scheduler.sqrt_1mac[t].view(-1, 1, 1, 1)
        pred_x0 = (x_noisy - sqrt_1ma * pred_noise) / (sqrt_a + 1e-8)

        # --- Auxiliary losses (vectorised, masked) -----------------------
        L_b = losses.boundary_loss(pred_x0, trajs, starts, goals, n_agents)
        L_t = losses.temporal_consistency_loss(pred_x0, n_agents)
        L_v = losses.velocity_constraint_loss(pred_x0, n_agents)
        L_c = losses.collision_loss(pred_x0, n_agents)
        L_g = losses.goal_loss(pred_x0, goals, n_agents)

        w_ = loss_weights
        total = (
            w_.denoising * denoising
            + w_.boundary * L_b
            + w_.temporal * L_t
            + w_.velocity * L_v
            + w_.collision * L_c
            + w_.goal * L_g
        )
        return LossOutput(
            total=total,
            denoising=denoising.detach(),
            boundary=L_b.detach(),
            temporal=L_t.detach(),
            velocity=L_v.detach(),
            collision=L_c.detach(),
            goal=L_g.detach(),
        )

    # ------------------------------------------------------------------
    # Sampling (DDIM with hard endpoint constraints + step-size clipping)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor,
        steps: int = 50,
        max_step_size: float = 0.1,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Single-window DDIM sample with start/goal pinning and step clipping.

        Returns ``[B, max_agents, T, 2]`` in the model's normalised coords.
        """
        B = frames.size(0)
        device = frames.device

        ctx_cond = self._build_context(frames, starts, goals, n_agents)
        ctx_null = self.null_ctx.unsqueeze(0).expand(B, -1) if guidance_scale != 1.0 else None

        x = torch.randn(B, self.max_agents, self.horizon, 2, device=device)

        # Hard-pin start (t=0) and goal (t=T-1) for active agents only.
        constraint_mask = torch.zeros(B, self.max_agents, self.horizon, dtype=torch.bool, device=device)
        constraint_mask[:, :, 0] = True
        constraint_mask[:, :, -1] = True
        constraint_values = torch.zeros(B, self.max_agents, self.horizon, 2, device=device)
        active = ~self._agent_mask(n_agents)  # [B, Na] True = active
        for b in range(B):
            n_b = int(n_agents[b].item())
            constraint_values[b, :n_b, 0] = starts[b, :n_b]
            constraint_values[b, :n_b, -1] = goals[b, :n_b]
        # Inactive agents: do not pin
        active_3d = active.unsqueeze(-1).expand_as(constraint_mask)
        constraint_mask = constraint_mask & active_3d

        timesteps = torch.linspace(self.T_steps - 1, 0, steps, device=device).long()
        agent_mask = self._agent_mask(n_agents)

        for i, t in enumerate(timesteps):
            t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
            feat = self.preprocessor(x, agent_mask)
            cond = ctx_cond + self.time_emb(t_b)
            eps = self.unet(feat, cond)

            if guidance_scale != 1.0 and ctx_null is not None:
                cond_u = ctx_null + self.time_emb(t_b)
                eps_u = self.unet(feat, cond_u)
                eps = eps_u + guidance_scale * (eps - eps_u)

            alpha_t = self.scheduler.alphas_cum[t]
            if i < len(timesteps) - 1:
                alpha_n = self.scheduler.alphas_cum[timesteps[i + 1]]
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
                x = torch.sqrt(alpha_n) * pred_x0 + torch.sqrt(1 - alpha_n) * eps
            else:
                x = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            # --- Step-size clipping (skip the very first iteration) -----
            if i > 0 and max_step_size > 0:
                step = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B, Na, T-1, 2]
                mag = step.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = torch.where(mag > max_step_size, max_step_size / mag, torch.ones_like(mag))
                # Don't clip the goal segment — it's pinned anyway.
                step_clipped = step * scale
                x_clipped = torch.cat([x[:, :, :1, :], x[:, :, :-1, :] + step_clipped], dim=2)
                # Preserve the pinned goal at index -1.
                x_clipped[:, :, -1, :] = x[:, :, -1, :]
                x = x_clipped

            # Hard re-apply boundary constraints after each step.
            cm = constraint_mask.unsqueeze(-1).expand_as(x)
            x = torch.where(cm, constraint_values, x)

        return x

    @torch.no_grad()
    def sample_moving_window(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor,
        full_horizon_length: int,
        horizon_size: int | None = None,
        max_step_size: float = 0.1,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Roll out a long trajectory by stitching windowed samples.

        Each window is a single ``self.sample()`` call. The next window's
        start positions are the previous window's end positions, and
        intermediate goals are linearly interpolated toward the final goal.
        """
        H = horizon_size or self.horizon
        if H != self.horizon:
            raise ValueError(
                f"horizon_size={H} must equal model horizon={self.horizon}; "
                "rebuild the model with the desired horizon for now."
            )
        B = frames.size(0)
        device = frames.device

        full = torch.zeros(B, self.max_agents, full_horizon_length, 2, device=device)
        cur_pos = starts.clone()
        n_windows = max(full_horizon_length // H, 1)

        for k in range(n_windows):
            t0, t1 = k * H, k * H + H
            if k == n_windows - 1:
                local_goal = goals
            else:
                p = (k + 1) / n_windows
                local_goal = cur_pos + p * (goals - cur_pos)

            window = self.sample(
                frames, cur_pos, local_goal, n_agents,
                steps=50, max_step_size=max_step_size, guidance_scale=guidance_scale,
            )
            full[:, :, t0:t1, :] = window
            cur_pos = window[:, :, -1, :].clone()

        return full
