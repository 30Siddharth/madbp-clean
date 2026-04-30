"""Per-method evaluation runners.

Every runner implements ``run_episode(seed) -> EpisodeMetrics`` and is
constructed with the same signature so that the canonical CLI never needs
method-specific branches.

Currently shipped:

  * ``MADPRunner`` — full implementation (DDIM + moving-window).
  * ``MAPPORunner``, ``CLFQPRunner``, ``MMDRunner``, ``ProjectedDiffusionRunner``
    — stubs that raise ``NotImplementedError`` so partial baselines never
    silently produce zeroed metrics.
"""

from __future__ import annotations

from madbp.evaluation.runners.base import RunnerProtocol
from madbp.evaluation.runners.madp import MADPRunner
from madbp.evaluation.runners.stubs import (
    CLFQPRunner,
    MAPPORunner,
    MMDRunner,
    ProjectedDiffusionRunner,
)

_REGISTRY: dict[str, type[RunnerProtocol]] = {
    "madp": MADPRunner,
    "mappo": MAPPORunner,
    "clf_qp": CLFQPRunner,
    "mmd": MMDRunner,
    "projected_diffusion": ProjectedDiffusionRunner,
}


def get_runner(name: str) -> type[RunnerProtocol]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method {name!r}. Choices: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


__all__ = [
    "MADPRunner",
    "MAPPORunner",
    "CLFQPRunner",
    "MMDRunner",
    "ProjectedDiffusionRunner",
    "get_runner",
]
