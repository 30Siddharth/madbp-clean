"""Per-episode evaluation metrics shared across all methods.

Reviewers expect: success rate, collision rate, path length, makespan,
mean acceleration / jerk, planning runtime, and time-to-goal. ``EpisodeMetrics``
is the single struct every method returns so that the per-method runners
in ``runners/`` are interchangeable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class EpisodeMetrics:
    """Per-episode aggregate metrics."""

    method: str
    scenario: str
    n_agents: int
    seed: int
    success: bool
    final_goal_distance: float            # mean over active agents (smaller is better)
    n_collisions: int
    collision_rate: float                 # collisions / step
    path_length_mean: float
    makespan: int                         # steps until last agent reaches goal (or max_steps)
    mean_acceleration: float
    mean_jerk: float
    plan_runtime_s: float                 # wall-clock time spent in planning
    time_to_goal_mean: float              # steps until each agent first within tolerance
    spl: float                            # success weighted by path length (Anderson et al. 2018)

    def as_row(self) -> dict:
        return asdict(self)


def compute_kinematics(positions: np.ndarray, dt: float = 0.1) -> tuple[float, float, float]:
    """Path length, mean acceleration, mean jerk for a [T, n, 2] trajectory."""
    if positions.shape[0] < 2:
        return 0.0, 0.0, 0.0
    v = np.diff(positions, axis=0) / dt
    pl = np.linalg.norm(v * dt, axis=-1).sum(axis=0).mean()
    if v.shape[0] < 2:
        return float(pl), 0.0, 0.0
    a = np.diff(v, axis=0) / dt
    am = np.linalg.norm(a, axis=-1).mean()
    if a.shape[0] < 2:
        return float(pl), float(am), 0.0
    j = np.diff(a, axis=0) / dt
    jm = np.linalg.norm(j, axis=-1).mean()
    return float(pl), float(am), float(jm)


def count_pairwise_collisions(positions: np.ndarray, min_dist: float) -> int:
    """Number of (timestep, pair) collision events across the whole episode."""
    T, N, _ = positions.shape
    if N < 2:
        return 0
    diff = positions[:, :, None, :] - positions[:, None, :, :]  # [T,N,N,2]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist[0], np.inf)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    coll = (dist < min_dist) & mask[None, :, :]
    return int(coll.sum())


def compute_spl(success: bool, optimal_length: float, actual_length: float) -> float:
    """Success weighted by path length (Anderson et al. 2018, eq. 1)."""
    if not success or actual_length <= 0:
        return 0.0
    return float(optimal_length / max(actual_length, optimal_length))
