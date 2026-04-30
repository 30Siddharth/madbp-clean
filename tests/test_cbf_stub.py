"""Smoke tests for the CBF roadmap stubs.

These verify that the stub interfaces import, validate config correctly, and
loudly raise NotImplementedError so nobody mistakes the stub for working code.

When the follow-up paper's implementation lands, replace these with real
correctness tests for projection feasibility, forward invariance, and
predicted-neighbor consistency.
"""

from __future__ import annotations

import pytest
import torch

from madbp.sampling.cbf import (
    BarrierRegion,
    CBFConfig,
    CircleObstacle,
    DistributedCBFProjector,
    DistributedCBFSampler,
)


def test_cbf_config_validates_gamma() -> None:
    with pytest.raises(ValueError, match="gamma_col"):
        CBFConfig(agent_radius=0.1, gamma_col=0.0)
    with pytest.raises(ValueError, match="gamma_obs"):
        CBFConfig(agent_radius=0.1, gamma_obs=1.5)


def test_cbf_config_validates_radius() -> None:
    with pytest.raises(ValueError, match="agent_radius"):
        CBFConfig(agent_radius=0.0)


def test_cbf_config_validates_inflation() -> None:
    with pytest.raises(ValueError, match="prediction_inflation"):
        CBFConfig(agent_radius=0.1, prediction_inflation=-0.5)


def test_cbf_config_accepts_obstacles_and_barriers() -> None:
    cfg = CBFConfig(
        agent_radius=0.1,
        obstacles=[CircleObstacle(center=(0.0, 0.0), radius=0.2)],
        barriers=[BarrierRegion(sdf_id="corridor")],
    )
    assert cfg.obstacles[0].radius == 0.2
    assert cfg.barriers[0].sdf_id == "corridor"


def test_projector_raises_until_implemented() -> None:
    cfg = CBFConfig(agent_radius=0.1)
    proj = DistributedCBFProjector(cfg)
    x0 = torch.zeros(1, 2, 4, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    starts = torch.zeros(1, 2, 2)
    goals = torch.ones(1, 2, 2)
    with pytest.raises(NotImplementedError, match="docs/cbf_backbone.md"):
        proj.project(x0, agent_idx=0, active_mask=mask, starts=starts, goals=goals)


def test_sampler_raises_until_implemented() -> None:
    cfg = CBFConfig(agent_radius=0.1)
    sampler = DistributedCBFSampler(
        model=torch.nn.Identity(), scheduler=object(), projector=DistributedCBFProjector(cfg)
    )
    with pytest.raises(NotImplementedError, match="docs/cbf_backbone.md"):
        sampler.sample(
            context={},
            starts=torch.zeros(1, 2, 2),
            goals=torch.ones(1, 2, 2),
            n_agents=torch.tensor([2]),
        )
