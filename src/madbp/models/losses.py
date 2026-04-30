"""Auxiliary trajectory losses applied to the predicted clean trajectory ``x0``.

All losses operate on a *batched* ``pred_x0`` tensor of shape
``[B, Na, T, 2]`` and are masked by the per-batch active-agent count.

The losses correspond to the four auxiliary terms in the paper:

    L_total = L_eps  +  w_t * L_temporal  +  w_b * L_boundary
                     +  w_g * L_goal      +  w_c * L_collision
                     +  w_v * L_velocity

Default weights live in ``configs/madbp.yaml`` so they can be overridden per
experiment without touching code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _active_mask(n_agents: torch.Tensor | int, max_agents: int, device: torch.device) -> torch.Tensor:
    """Return a [B, max_agents] bool mask where True = active agent.

    Always materialises a batched tensor so downstream code is purely vectorised.
    """
    if isinstance(n_agents, torch.Tensor):
        n = n_agents.to(device=device, dtype=torch.long)
    else:
        # Scalar broadcast — but we don't know the batch size, so callers using
        # a scalar must instead pass a tensor.
        raise TypeError("n_agents must be a tensor; pass torch.tensor([...]) for scalar broadcasts")
    arange = torch.arange(max_agents, device=device)[None, :]
    return arange < n[:, None]


def boundary_loss(
    pred_x0: torch.Tensor,
    trajs: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    n_agents: torch.Tensor,
) -> torch.Tensor:
    """Penalise drift of (start, goal, full-trajectory) under the predicted x0.

    ``pred_x0`` and ``trajs`` are ``[B, Na, T, 2]``; ``starts``/``goals`` are
    ``[B, Na, 2]``. Loss is averaged over *active* agents only.
    """
    B, Na, T, _ = pred_x0.shape
    mask = _active_mask(n_agents, Na, pred_x0.device)  # [B, Na]
    mask.unsqueeze(-1)  # [B, Na, 1]

    start_err = ((pred_x0[:, :, 0, :] - starts) ** 2).mean(dim=-1)  # [B, Na]
    goal_err = ((pred_x0[:, :, -1, :] - goals) ** 2).mean(dim=-1)  # [B, Na]
    traj_err = ((pred_x0 - trajs) ** 2).mean(dim=(-2, -1))  # [B, Na]

    n_active = mask.float().sum().clamp(min=1.0)
    s = (start_err * mask).sum() / n_active
    g = (goal_err * mask).sum() / n_active
    t = (traj_err * mask).sum() / n_active
    return s + g + 0.1 * t


def goal_loss(pred_x0: torch.Tensor, goals: torch.Tensor, n_agents: torch.Tensor) -> torch.Tensor:
    """MSE between final predicted position and goal, masked over active agents."""
    B, Na, _, _ = pred_x0.shape
    mask = _active_mask(n_agents, Na, pred_x0.device)
    err = ((pred_x0[:, :, -1, :] - goals) ** 2).mean(dim=-1)  # [B, Na]
    n_active = mask.float().sum().clamp(min=1.0)
    return (err * mask).sum() / n_active


def temporal_consistency_loss(pred_x0: torch.Tensor, n_agents: torch.Tensor) -> torch.Tensor:
    """Penalise large velocity changes (acceleration) and jerk."""
    B, Na, T, _ = pred_x0.shape
    if T < 3:
        return pred_x0.new_tensor(0.0)
    mask = _active_mask(n_agents, Na, pred_x0.device).float()  # [B, Na]
    v = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]  # [B, Na, T-1, 2]
    a = v[:, :, 1:, :] - v[:, :, :-1, :]              # [B, Na, T-2, 2]
    accel = a.norm(dim=-1).mean(dim=-1)               # [B, Na]
    loss = (accel * mask).sum() / mask.sum().clamp(min=1.0)
    if a.size(2) > 1:
        j = a[:, :, 1:, :] - a[:, :, :-1, :]          # [B, Na, T-3, 2]
        jerk = j.norm(dim=-1).mean(dim=-1)
        loss = loss + 0.1 * (jerk * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def velocity_constraint_loss(
    pred_x0: torch.Tensor,
    n_agents: torch.Tensor,
    max_velocity: float = 2.0,
    dt: float = 0.1,
) -> torch.Tensor:
    """Hinge penalty for per-step velocities exceeding ``max_velocity``."""
    B, Na, T, _ = pred_x0.shape
    if T < 2:
        return pred_x0.new_tensor(0.0)
    mask = _active_mask(n_agents, Na, pred_x0.device).float()
    v = (pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]) / dt
    vmag = v.norm(dim=-1)  # [B, Na, T-1]
    excess = F.relu(vmag - max_velocity).mean(dim=-1)
    loss = (excess * mask).sum() / mask.sum().clamp(min=1.0)
    if v.size(2) > 1:
        a = (v[:, :, 1:, :] - v[:, :, :-1, :]) / dt
        amag = a.norm(dim=-1).mean(dim=-1)
        loss = loss + 0.1 * (amag * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def collision_loss(
    pred_x0: torch.Tensor,
    n_agents: torch.Tensor,
    min_distance: float = 0.1,
) -> torch.Tensor:
    """Hinge penalty for inter-agent distances below ``min_distance``.

    Vectorised across batch and time. Padded agents are masked out via the
    pairwise active mask.
    """
    B, Na, T, _ = pred_x0.shape
    mask = _active_mask(n_agents, Na, pred_x0.device)  # [B, Na]
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)   # [B, Na, Na]
    eye = torch.eye(Na, dtype=torch.bool, device=pred_x0.device)
    pair_mask = pair_mask & ~eye

    # Pairwise distances at every timestep: [B, T, Na, Na]
    pos = pred_x0.permute(0, 2, 1, 3)  # [B, T, Na, 2]
    diff = pos.unsqueeze(3) - pos.unsqueeze(2)  # [B, T, Na, Na, 2]
    dist = diff.norm(dim=-1)  # [B, T, Na, Na]

    penalty = F.relu(min_distance - dist)  # [B, T, Na, Na]
    pm = pair_mask.unsqueeze(1).expand_as(penalty)  # [B, T, Na, Na]
    n_pairs = pm.float().sum().clamp(min=1.0)
    return (penalty * pm).sum() / n_pairs
