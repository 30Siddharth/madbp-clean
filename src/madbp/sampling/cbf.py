"""Distributed CBF projection — follow-up paper roadmap.

This module is a *stub*. The interfaces are stable; the implementations raise
``NotImplementedError`` so the rest of the package can already import the types
without pulling in heavy QP dependencies.

See ``docs/cbf_backbone.md`` for the full math and design rationale.

Implementation order, when the time comes:
    1. ``DistributedCBFProjector`` with a ``qpth`` backend.
    2. ``DistributedCBFSampler`` wrapping the existing diffusion sampler.
    3. Eval runner in ``src/madbp/evaluation/runners/dist_cbf.py``.

Trainer, dataset, model, and losses are intentionally untouched: the CBF
backbone is a deployment-time add-on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircleObstacle:
    """Static circular obstacle in the planning frame."""

    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class BarrierRegion:
    """Keep-out region defined by a signed-distance function callable.

    The SDF should be positive outside the region and negative inside.
    """

    sdf_id: str  # identifier for a registered SDF; the projector resolves it


# ---------------------------------------------------------------------------
# Projector configuration
# ---------------------------------------------------------------------------


@dataclass
class CBFConfig:
    """Hyperparameters for the distributed CBF projector.

    Attributes
    ----------
    agent_radius:
        Physical radius of every agent (homogeneous teams for now).
    gamma_col, gamma_obs, gamma_bar:
        Discrete-time CBF decay rates, one per barrier family. Each must lie
        in ``(0, 1]``. Larger ``gamma`` = more aggressive recovery toward the
        safe set boundary.
    prediction_inflation:
        Additional radius padding ``delta_pred`` to compensate for disagreement
        between agents' predictions of each other (see ``docs/cbf_backbone.md``
        §4.3). Set ``0`` for the noiseless analysis, calibrate from data for
        deployment.
    qp_backend:
        Which QP backend to use once implemented.
    mode:
        Where in the reverse process to project. ``"per_step"`` is the strong
        guarantee variant; ``"final_only"`` is a cheaper baseline.
    """

    agent_radius: float
    obstacles: list[CircleObstacle] = field(default_factory=list)
    barriers: list[BarrierRegion] = field(default_factory=list)
    gamma_col: float = 0.5
    gamma_obs: float = 0.5
    gamma_bar: float = 0.5
    prediction_inflation: float = 0.0
    qp_backend: Literal["qpth", "cvxpy", "osqp"] = "qpth"
    mode: Literal["per_step", "final_only"] = "per_step"

    def __post_init__(self) -> None:
        for name, val in (
            ("gamma_col", self.gamma_col),
            ("gamma_obs", self.gamma_obs),
            ("gamma_bar", self.gamma_bar),
        ):
            if not 0.0 < val <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]; got {val}")
        if self.agent_radius <= 0:
            raise ValueError("agent_radius must be positive")
        if self.prediction_inflation < 0:
            raise ValueError("prediction_inflation must be non-negative")


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class DistributedCBFProjector:
    """Per-agent CBF projection of a diffusion x0 prediction.

    Each call solves a *local* QP for one agent, treating the other agents'
    trajectories from the same ``x0_hat`` tensor as predicted obstacles. The
    projector is intentionally agent-indexed rather than batched-over-agents
    to make it explicit that no agent sees another's QP.

    Parameters
    ----------
    config:
        See :class:`CBFConfig`.
    """

    def __init__(self, config: CBFConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ API

    def project(
        self,
        x0_hat: Tensor,
        agent_idx: int,
        active_mask: Tensor,
        starts: Tensor,
        goals: Tensor,
    ) -> Tensor:
        """Project ``x0_hat[:, agent_idx]`` onto the local safe set.

        Parameters
        ----------
        x0_hat:
            Predicted clean trajectories, shape ``(B, N, H, d)``. The slabs
            ``x0_hat[:, j]`` for ``j != agent_idx`` are treated as predicted
            neighbor positions. **Not modified.**
        agent_idx:
            Which agent's QP to solve. Returned tensor is for this agent only.
        active_mask:
            Shape ``(B, N)`` bool. Inactive agents are skipped as neighbors.
        starts, goals:
            Shape ``(B, N, d)``. Used as equality constraints at ``tau=0``
            and ``tau=H-1`` for the projected agent.

        Returns
        -------
        Tensor
            Projected trajectory for ``agent_idx``, shape ``(B, H, d)``.

        Notes
        -----
        Constraints assembled per the math in ``docs/cbf_backbone.md`` §3.4:

        * Pairwise inter-agent CBF for every active neighbor.
        * Static obstacle CBF for every obstacle in ``config.obstacles``.
        * Barrier-region CBF for every entry in ``config.barriers``.
        * Endpoint equality.

        All ``h_k`` are linearized around ``x0_hat`` to yield a QP. Solver
        choice driven by ``config.qp_backend``.
        """
        raise NotImplementedError(
            "DistributedCBFProjector.project: see docs/cbf_backbone.md "
            "for the math; implement with qpth in the follow-up paper."
        )

    # -------------------------------------------------------------- helpers

    def _build_inter_agent_constraints(
        self,
        x0_hat: Tensor,
        agent_idx: int,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Linearize pairwise collision CBFs around ``x0_hat``.

        Returns
        -------
        (A, b)
            ``A x_i >= b`` form, shapes ``(B, K, H*d)`` and ``(B, K)`` where
            ``K = sum(active_neighbors) * (H - 1)``.
        """
        raise NotImplementedError

    def _build_obstacle_constraints(self, x0_hat: Tensor, agent_idx: int) -> tuple[Tensor, Tensor]:
        """Linearize static obstacle CBFs."""
        raise NotImplementedError

    def _build_barrier_constraints(self, x0_hat: Tensor, agent_idx: int) -> tuple[Tensor, Tensor]:
        """Linearize keep-out barrier CBFs from registered SDFs."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class DistributedCBFSampler:
    """Wraps a :class:`MultiAgentDiffusionPlanner` with per-step CBF projection.

    The sampler runs the same DDIM schedule as the unconstrained sampler but
    intercepts ``x0_hat`` at every reverse step and projects it through
    :class:`DistributedCBFProjector` before re-noising.

    The model and scheduler are reused unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: object,
        projector: DistributedCBFProjector,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.projector = projector

    @torch.no_grad()
    def sample(
        self,
        context: dict,
        starts: Tensor,
        goals: Tensor,
        n_agents: Tensor,
        cfg_scale: float = 0.0,
    ) -> Tensor:
        """Decentralized CBF-guarded sampling.

        Each agent runs the *same* model on the *same* context (so priors
        agree in distribution), then projects locally. No inter-agent comms
        inside the reverse loop — only initial state broadcast.

        Returns
        -------
        Tensor
            Joint trajectories ``(B, N, H, d)``, with each agent's slab
            taken from its own local sampler.
        """
        raise NotImplementedError(
            "DistributedCBFSampler.sample: implement once "
            "DistributedCBFProjector.project lands; see docs/cbf_backbone.md."
        )


__all__ = [
    "CircleObstacle",
    "BarrierRegion",
    "CBFConfig",
    "DistributedCBFProjector",
    "DistributedCBFSampler",
]
