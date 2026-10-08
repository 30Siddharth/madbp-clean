"""Auxiliary trajectory losses applied to the predicted clean trajectory ``x0``.

All losses take a *batched* ``pred_x0`` of shape ``[B, N, T, 2]`` and a boolean
``active`` mask ``[B, N]`` (``True`` = live agent). They are fully vectorised.

Correspondence with the paper (Sec. III-C.3):

====================  =====================================================
paper                 here
====================  =====================================================
Eq. 3 ``L_boundary``  :func:`boundary_loss`    (L2 norms, not squared)
Eq. 4-6 ``L_temporal`` :func:`temporal_consistency_loss`
Eq. 7 ``L_collision``  :func:`collision_loss`  (agent-agent + obstacle term)
====================  =====================================================

Distances (``d_min``, ``d_obs``) are expressed in the *normalised* coordinates
the model works in. Convert world distances with
``NormalizedTrajectoryDataset.norm_distance``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _l2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable L2 norm (finite gradient at zero)."""
    return torch.sqrt((x * x).sum(dim=dim) + _EPS)


def _mean_over_active(per_agent: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Mean of a ``[B, N]`` tensor over active slots."""
    m = active.to(per_agent.dtype)
    return (per_agent * m).sum() / m.sum().clamp(min=1.0)


def boundary_loss(
    pred_x0: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    active: torch.Tensor,
    trajs: torch.Tensor | None = None,
    traj_weight: float = 0.0,
    squared: bool = False,
) -> torch.Tensor:
    """Paper Eq. 3: ``||x̂0(t) - x(t)|| + ||x̂0(t_H) - x*||`` per active agent.

    ``traj_weight > 0`` adds an optional full-trajectory reconstruction term
    (present in the original code, *not* in the paper; off by default).
    """
    start_err = pred_x0[:, :, 0, :] - starts
    goal_err = pred_x0[:, :, -1, :] - goals
    if squared:
        s = (start_err**2).sum(-1)
        g = (goal_err**2).sum(-1)
    else:
        s = _l2(start_err)
        g = _l2(goal_err)
    loss = _mean_over_active(s, active) + _mean_over_active(g, active)
    if traj_weight > 0.0 and trajs is not None:
        t = ((pred_x0 - trajs) ** 2).mean(dim=(-2, -1))
        loss = loss + traj_weight * _mean_over_active(t, active)
    return loss


def goal_loss(pred_x0: torch.Tensor, goals: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Squared error between final predicted position and goal (extra; off by default)."""
    err = ((pred_x0[:, :, -1, :] - goals) ** 2).sum(-1)
    return _mean_over_active(err, active)


def temporal_consistency_loss(
    pred_x0: torch.Tensor,
    active: torch.Tensor,
    lambda_jerk: float = 0.1,
) -> torch.Tensor:
    """Paper Eq. 4-6: velocity-change (``L_direction``) plus ``lambda_jerk`` * jerk.

    Velocities and accelerations are finite differences of the predicted
    positions, as in the paper.
    """
    B, N, T, _ = pred_x0.shape
    if T < 3:
        return pred_x0.new_tensor(0.0)
    v = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]  # [B,N,T-1,2]
    a = v[:, :, 1:, :] - v[:, :, :-1, :]  # [B,N,T-2,2]
    direction = _l2(a).mean(dim=-1)  # [B,N]
    loss = _mean_over_active(direction, active)
    if a.size(2) > 1 and lambda_jerk > 0.0:
        j = a[:, :, 1:, :] - a[:, :, :-1, :]
        jerk = _l2(j).mean(dim=-1)
        loss = loss + lambda_jerk * _mean_over_active(jerk, active)
    return loss


def velocity_constraint_loss(
    pred_x0: torch.Tensor,
    active: torch.Tensor,
    max_velocity: float = 2.0,
    dt: float = 0.1,
) -> torch.Tensor:
    """Hinge penalty on step velocities above ``max_velocity`` (extra; off by default)."""
    B, N, T, _ = pred_x0.shape
    if T < 2:
        return pred_x0.new_tensor(0.0)
    v = (pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]) / dt
    excess = F.relu(_l2(v) - max_velocity).mean(dim=-1)
    loss = _mean_over_active(excess, active)
    if v.size(2) > 1:
        a = (v[:, :, 1:, :] - v[:, :, :-1, :]) / dt
        loss = loss + 0.1 * _mean_over_active(_l2(a).mean(dim=-1), active)
    return loss


def collision_loss(
    pred_x0: torch.Tensor,
    active: torch.Tensor,
    d_min: float = 0.1,
    obstacles: torch.Tensor | None = None,
    obstacle_mask: torch.Tensor | None = None,
    d_obs: float = 0.1,
    reduction: str = "paper",
) -> torch.Tensor:
    """Paper Eq. 7: hinge on inter-agent distance plus a hinge on obstacle distance.

    ``obstacles`` is ``[B, M, 2]`` obstacle centres (normalised coordinates) and
    ``obstacle_mask`` ``[B, M]`` marks real obstacles. With ``obstacles=None``
    only the agent-agent term is computed.

    ``reduction="paper"`` follows Eq. 7 literally: sum over time and (ordered)
    pairs, divided by the batch size. ``reduction="mean"`` instead averages over
    the number of contributing terms, which is invariant to team size and
    horizon.
    """
    B, N, T, _ = pred_x0.shape
    pair = active.unsqueeze(2) & active.unsqueeze(1)  # [B,N,N]
    pair = pair & ~torch.eye(N, dtype=torch.bool, device=pred_x0.device)

    pos = pred_x0.permute(0, 2, 1, 3)  # [B,T,N,2]
    diff = pos.unsqueeze(3) - pos.unsqueeze(2)  # [B,T,N,N,2]
    dist = _l2(diff)  # [B,T,N,N]
    pm = pair.unsqueeze(1).expand_as(dist).to(dist.dtype)
    agent_pen = F.relu(d_min - dist) * pm
    total = agent_pen.sum(dim=(1, 2, 3))  # [B]
    n_terms = pm.sum().clamp(min=1.0)

    obs_total = None
    n_obs_terms = None
    if obstacles is not None and obstacles.size(1) > 0:
        if obstacle_mask is None:
            obstacle_mask = torch.ones(B, obstacles.size(1), dtype=torch.bool, device=pred_x0.device)
        d = _l2(pred_x0.unsqueeze(3) - obstacles[:, None, None, :, :])  # [B,N,T,M]
        om = (
            active[:, :, None, None].to(d.dtype)
            * obstacle_mask[:, None, None, :].to(d.dtype)
        )
        obs_pen = F.relu(d_obs - d) * om
        obs_total = obs_pen.sum(dim=(1, 2, 3))  # [B]
        n_obs_terms = om.sum().clamp(min=1.0)

    if reduction == "paper":
        out = total.mean()
        if obs_total is not None:
            out = out + obs_total.mean()
        return out
    if reduction == "mean":
        out = total.sum() / n_terms
        if obs_total is not None and n_obs_terms is not None:
            out = out + obs_total.sum() / n_obs_terms
        return out
    raise ValueError(f"Unknown reduction {reduction!r}")
