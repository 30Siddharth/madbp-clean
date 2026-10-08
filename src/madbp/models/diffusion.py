"""``MultiAgentDiffusionPlanner`` - the MA-DBP model (paper Sec. III).

A single shared conditional diffusion model denoises the trajectories of all
agents at once::

    eps_theta( Axial(x_k, mask) ; c, k )

``Axial`` is the semantic axial-attention pre-processor, ``c`` the context
vector from :class:`~madbp.models.context.ContextEncoder`.

Defaults follow the paper:

* loss weights ``W = [0.85, 0.025, 0.025, 0.1]`` over
  ``[noise, boundary, temporal, collision]`` (Eq. 8), plain noise MSE (Eq. 2);
* DDIM-style sampling from ``N(0, I)`` with the start/goal pinned after every
  step (Algorithm 2).

Everything the original code did on top of that is available but **off by
default**: Min-SNR weighting (``min_snr_gamma``), per-step displacement clipping
(``max_step_size``), classifier-free guidance (``cfg_dropout_prob`` /
``guidance_scale``) and the velocity / goal losses.

Agents are identified by a boolean ``active`` mask ``[B, N]`` (see
:mod:`madbp.models.masks`). Passing ``n_agents`` instead selects the first
``n_agents[b]`` slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from madbp.models import losses
from madbp.models.context import ContextEncoder
from madbp.models.masks import resolve_active
from madbp.models.preprocessor import AxialPreprocessor
from madbp.models.scheduler import BoundaryAwareNoiseScheduler, SinusoidalPosEmb
from madbp.models.unet import TrajectoryUNet


@dataclass
class LossWeights:
    """Coefficients of ``L_total`` (paper Eq. 8: ``[0.85, 0.025, 0.025, 0.1]``)."""

    denoising: float = 0.85
    boundary: float = 0.025
    temporal: float = 0.025
    collision: float = 0.10
    velocity: float = 0.0
    goal: float = 0.0


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
    """Single shared multi-agent diffusion planner."""

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
        min_snr_gamma: float | None = None,
        d_min: float = 0.1,
        d_obs: float = 0.1,
        lambda_jerk: float = 0.1,
        collision_reduction: str = "paper",
        boundary_traj_weight: float = 0.0,
        pose_encoder: str = "per_agent",
        count_encoder: str = "mlp",
        agent_pos: str = "learned",
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.horizon = horizon
        self.state_dim = state_dim
        self.T_steps = diffusion_steps
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.min_snr_gamma = min_snr_gamma
        self.d_min = d_min
        self.d_obs = d_obs
        self.lambda_jerk = lambda_jerk
        self.collision_reduction = collision_reduction
        self.boundary_traj_weight = boundary_traj_weight

        self.scheduler = BoundaryAwareNoiseScheduler(
            T=diffusion_steps, schedule_type=schedule_type, max_noise_ratio=3.0
        )
        self.time_emb = SinusoidalPosEmb(hid)
        self.ctx_enc = ContextEncoder(
            img_ch,
            pose_dim=state_dim,
            hid=hid,
            max_agents=max_agents,
            pose_encoder=pose_encoder,
            count_encoder=count_encoder,
        )
        self.preprocessor = AxialPreprocessor(
            state_dim=state_dim,
            feat_dim=hid,
            num_heads=4,
            max_agents=max_agents,
            T=horizon,
            agent_pos=agent_pos,
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

    def model_config(self) -> dict:
        """Constructor arguments needed to rebuild this model from a checkpoint."""
        return {
            "max_agents": self.max_agents,
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "hid": self.null_ctx.numel(),
            "diffusion_steps": self.T_steps,
            "schedule_type": self.scheduler.schedule_type,
            "cfg_dropout_prob": self.cfg_dropout_prob,
            "min_snr_gamma": self.min_snr_gamma,
            "d_min": self.d_min,
            "d_obs": self.d_obs,
            "lambda_jerk": self.lambda_jerk,
            "collision_reduction": self.collision_reduction,
            "boundary_traj_weight": self.boundary_traj_weight,
            "pose_encoder": self.ctx_enc.pose_encoder,
            "count_encoder": self.ctx_enc.count_encoder,
            "agent_pos": self.preprocessor.agent_pos,
        }

    def _build_context(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        active: torch.Tensor,
        dropout: bool = False,
    ) -> torch.Tensor:
        """``[B, hid]`` context, with optional per-sample CFG dropout in training."""
        ctx = self.ctx_enc(frames, starts, goals, active)
        if dropout and self.training and self.cfg_dropout_prob > 0.0:
            B = ctx.size(0)
            drop = (torch.rand(B, device=ctx.device) < self.cfg_dropout_prob).float().unsqueeze(-1)
            ctx = (1.0 - drop) * ctx + drop * self.null_ctx.unsqueeze(0)
        return ctx

    # ------------------------------------------------------------------
    # Training forward: predicts noise and returns the weighted L_total
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor | None,
        trajs: torch.Tensor,
        loss_weights: LossWeights | None = None,
        active: torch.Tensor | None = None,
        obstacles: torch.Tensor | None = None,
        obstacle_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        """One denoising step on a batch of windows.

        ``frames`` ``[B,3,H,W]``; ``starts``/``goals`` ``[B,N,2]``; ``trajs``
        ``[B,N,T,2]`` clean windows; ``n_agents`` ``[B]`` (or ``None`` when
        ``active`` ``[B,N]`` is given). Optional ``obstacles`` ``[B,M,2]`` and
        ``obstacle_mask`` ``[B,M]`` activate the obstacle term of Eq. 7.
        """
        if loss_weights is None:
            loss_weights = LossWeights()
        B = frames.size(0)
        device = frames.device
        active = resolve_active(n_agents, active, self.max_agents, device)
        agent_mask = ~active  # True = padded

        ctx = self._build_context(frames, starts, goals, active, dropout=True)

        t = torch.randint(0, self.T_steps, (B,), device=device)
        x0 = trajs.permute(0, 1, 3, 2)  # [B,N,2,T]
        x_noisy_c, noise_c = self.scheduler.q_sample(x0, t)
        x_noisy = x_noisy_c.permute(0, 1, 3, 2)  # [B,N,T,2]
        target_noise = noise_c.permute(0, 1, 3, 2)

        feat = self.preprocessor(x_noisy, agent_mask)
        cond = ctx + self.time_emb(t)
        pred_noise = self.unet(feat, cond)  # [B,N,T,2]

        # Eq. 2: noise MSE on active agents (optionally Min-SNR weighted).
        se = (pred_noise - target_noise) ** 2
        if self.min_snr_gamma is not None:
            ac = self.scheduler.alphas_cum[t]
            snr = ac / (1.0 - ac)
            w = torch.minimum(snr, torch.full_like(snr, float(self.min_snr_gamma))) / snr
            se = se * w.view(B, 1, 1, 1)
        act4 = active[:, :, None, None].expand_as(se)
        denoising = se[act4].mean()

        # Predicted clean trajectory, used by every auxiliary loss.
        sqrt_a = self.scheduler.sqrt_ac[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.scheduler.sqrt_1mac[t].view(-1, 1, 1, 1)
        pred_x0 = (x_noisy - sqrt_1ma * pred_noise) / (sqrt_a + 1e-8)

        zero = pred_x0.new_tensor(0.0)
        w_ = loss_weights
        L_b = (
            losses.boundary_loss(
                pred_x0, starts, goals, active, trajs, traj_weight=self.boundary_traj_weight
            )
            if w_.boundary > 0.0
            else zero
        )
        L_t = (
            losses.temporal_consistency_loss(pred_x0, active, self.lambda_jerk)
            if w_.temporal > 0.0
            else zero
        )
        L_c = (
            losses.collision_loss(
                pred_x0,
                active,
                d_min=self.d_min,
                obstacles=obstacles,
                obstacle_mask=obstacle_mask,
                d_obs=self.d_obs,
                reduction=self.collision_reduction,
            )
            if w_.collision > 0.0
            else zero
        )
        L_v = losses.velocity_constraint_loss(pred_x0, active) if w_.velocity > 0.0 else zero
        L_g = losses.goal_loss(pred_x0, goals, active) if w_.goal > 0.0 else zero

        total = (
            w_.denoising * denoising
            + w_.boundary * L_b
            + w_.temporal * L_t
            + w_.collision * L_c
            + w_.velocity * L_v
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
    # Sampling (paper Algorithm 2: DDIM with boundary constraints)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor | None = None,
        *,
        active: torch.Tensor | None = None,
        steps: int = 50,
        max_step_size: float | None = None,
        guidance_scale: float = 1.0,
        init: str = "gaussian",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one window ``[B, N, H, 2]`` (normalised coordinates).

        The start (index 0) and goal (index -1) of every active agent are
        re-imposed after each denoising step.

        ``init="gaussian"`` draws ``X^K ~ N(0, I)`` exactly as Algorithm 2.
        Because the clamped schedule never reaches pure noise
        (``alphas_cum[-1] > 0``), ``init="linear"`` offers a warm start from the
        straight start-goal line noised to the terminal level; it injects a
        straight-line prior, so report it separately if used.
        """
        B = frames.size(0)
        device = frames.device
        H = self.horizon
        active = resolve_active(n_agents, active, self.max_agents, device)
        agent_mask = ~active

        # Call ``model.eval()`` before sampling (BatchNorm / dropout behave differently).
        ctx_cond = self._build_context(frames, starts, goals, active, dropout=False)
        use_cfg = guidance_scale != 1.0
        ctx_null = self.null_ctx.unsqueeze(0).expand(B, -1) if use_cfg else None

        # Start/goal constraints on active slots.
        constraint_mask = torch.zeros(B, self.max_agents, H, dtype=torch.bool, device=device)
        constraint_mask[:, :, 0] = True
        constraint_mask[:, :, -1] = True
        constraint_mask = constraint_mask & active.unsqueeze(-1)
        constraint_values = torch.zeros(B, self.max_agents, H, 2, device=device)
        constraint_values[:, :, 0] = starts
        constraint_values[:, :, -1] = goals
        cm = constraint_mask.unsqueeze(-1)

        noise = torch.randn(B, self.max_agents, H, 2, device=device, generator=generator)
        if init == "gaussian":
            x = noise
        elif init == "linear":
            frac = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
            line = starts.unsqueeze(2) * (1 - frac) + goals.unsqueeze(2) * frac
            a_T = self.scheduler.alphas_cum[-1]
            x = torch.sqrt(a_T) * line + torch.sqrt(1 - a_T) * noise
        else:
            raise ValueError(f"Unknown init {init!r}")

        timesteps = torch.linspace(self.T_steps - 1, 0, steps, device=device).long()
        for i, t in enumerate(timesteps):
            t_b = torch.full((B,), int(t), device=device, dtype=torch.long)
            feat = self.preprocessor(x, agent_mask)
            eps = self.unet(feat, ctx_cond + self.time_emb(t_b))
            if use_cfg and ctx_null is not None:
                eps_u = self.unet(feat, ctx_null + self.time_emb(t_b))
                eps = eps_u + guidance_scale * (eps - eps_u)

            alpha_t = self.scheduler.alphas_cum[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            if i < len(timesteps) - 1:
                alpha_n = self.scheduler.alphas_cum[timesteps[i + 1]]
                x = torch.sqrt(alpha_n) * pred_x0 + torch.sqrt(1 - alpha_n) * eps
            else:
                x = pred_x0

            if max_step_size is not None and max_step_size > 0 and i > 0:
                step = x[:, :, 1:, :] - x[:, :, :-1, :]
                mag = step.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = torch.where(mag > max_step_size, max_step_size / mag, torch.ones_like(mag))
                clipped = torch.cumsum(step * scale, dim=2) + x[:, :, :1, :]
                x = torch.cat([x[:, :, :1, :], clipped], dim=2)

            x = torch.where(cm, constraint_values, x)

        return x
