"""Protocol shared by every evaluation runner."""

from __future__ import annotations

from typing import Protocol

from madbp.evaluation.metrics import EpisodeMetrics


class RunnerProtocol(Protocol):
    def __init__(
        self,
        scenario: str,
        n_agents: int,
        max_steps: int,
        collision_radius: float,
        checkpoint: str | None,
        device: str,
    ) -> None: ...

    def run_episode(self, seed: int) -> EpisodeMetrics: ...
