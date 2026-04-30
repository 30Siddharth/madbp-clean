"""Baseline runners not yet ported to the clean codebase.

These exist so the CLI surface is complete, but they refuse to silently
produce metrics. Implement them in follow-up PRs as the experiment
schedule reaches the corresponding baseline.
"""

from __future__ import annotations

from madbp.evaluation.metrics import EpisodeMetrics


class _NotPortedRunner:
    name = "stub"

    def __init__(self, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(
            f"The {self.name!r} runner has not been ported to the clean codebase yet. "
            "See docs/changes.md for the porting checklist."
        )

    def run_episode(self, seed: int) -> EpisodeMetrics:  # pragma: no cover
        raise NotImplementedError


class MAPPORunner(_NotPortedRunner):
    name = "mappo"


class CLFQPRunner(_NotPortedRunner):
    name = "clf_qp"


class MMDRunner(_NotPortedRunner):
    name = "mmd"


class ProjectedDiffusionRunner(_NotPortedRunner):
    name = "projected_diffusion"
