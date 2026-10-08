"""Scenario description and generators (pure numpy, world units: arena = [-1, 1]^2).

Matches VMAS conventions used for the training data: ``world_semidim = 1`` and
agent radius 0.05, so the paper's collision distance ``d_min = 0.1`` equals two
radii. Obstacles are discs; a "wall" is a row of discs with a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    """Dynamic team change at control step ``step``."""

    step: int
    kind: str  # "add" | "remove"
    slot: int | None = None  # for "remove": slot to drop
    start: tuple[float, float] | None = None  # for "add"
    goal: tuple[float, float] | None = None  # for "add"


@dataclass
class Scenario:
    name: str
    starts: np.ndarray  # [n, 2]
    goals: np.ndarray  # [n, 2]
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # centres
    obstacle_radius: float = 0.08
    bound: float = 1.0
    agent_radius: float = 0.05
    events: list[Event] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        return len(self.starts)


def _sample_free(rng, n, obstacles, obs_r, bound, min_sep, avoid_r, existing=None, tries=2000):
    pts = [] if existing is None else [p for p in existing]
    out = []
    lim = bound - 0.1
    for _ in range(n):
        for _ in range(tries):
            p = rng.uniform(-lim, lim, size=2)
            if len(obstacles) and np.min(np.linalg.norm(obstacles - p, axis=1)) < obs_r + avoid_r:
                continue
            if pts and np.min(np.linalg.norm(np.asarray(pts) - p, axis=1)) < min_sep:
                continue
            pts.append(p)
            out.append(p)
            break
        else:
            raise RuntimeError("could not place point; relax separation")
    return np.asarray(out)


def make_scenario(kind: str, n_agents: int, rng: np.random.Generator) -> Scenario:
    """``kind`` in {"open", "obstacles", "barrier", "corridor"}."""
    r_obs = 0.08
    obstacles = np.zeros((0, 2))
    if kind == "open":
        pass
    elif kind == "obstacles":
        k = int(rng.integers(3, 6))
        obstacles = rng.uniform(-0.6, 0.6, size=(k, 2))
    elif kind == "barrier":
        # horizontal wall at y=0 with a gap of 0.4 at a random x
        gap_x = rng.uniform(-0.5, 0.5)
        xs = np.arange(-1.0, 1.0 + 1e-6, 2 * r_obs)
        xs = xs[np.abs(xs - gap_x) > 0.25]
        obstacles = np.stack([xs, np.zeros_like(xs)], axis=1)
    elif kind == "corridor":
        # two parallel walls (y = +-0.3) with open ends
        xs = np.arange(-0.6, 0.6 + 1e-6, 2 * r_obs)
        obstacles = np.concatenate(
            [np.stack([xs, np.full_like(xs, 0.3)], 1), np.stack([xs, np.full_like(xs, -0.3)], 1)]
        )
    else:
        raise ValueError(f"unknown scenario kind {kind!r}")

    if kind in ("barrier", "corridor"):
        # starts on one side, goals on the other (forces passing the obstacle)
        def side(sign):
            pts = []
            sep = 0.2 if kind == "barrier" else 0.14
            for _ in range(20000):
                if len(pts) == n_agents:
                    break
                p = np.array([rng.uniform(-0.8, 0.8), sign * rng.uniform(0.45, 0.9)])
                if kind == "corridor":
                    p = np.array([sign * rng.uniform(0.68, 0.93), rng.uniform(-0.2, 0.2)])
                if all(np.linalg.norm(p - q) > sep for q in pts):
                    pts.append(p)
            if len(pts) < n_agents:
                raise RuntimeError(f"cannot fit {n_agents} agents in the {kind} start region")
            return np.asarray(pts)

        starts, goals = side(+1), side(-1)
        goals = goals[rng.permutation(n_agents)]
    else:
        starts = _sample_free(rng, n_agents, obstacles, r_obs, 1.0, 0.2, 0.12)
        goals = _sample_free(rng, n_agents, obstacles, r_obs, 1.0, 0.2, 0.12)
        # keep start-goal distances meaningful
        for i in range(n_agents):
            while np.linalg.norm(goals[i] - starts[i]) < 0.6:
                goals[i] = _sample_free(rng, 1, obstacles, r_obs, 1.0, 0.0, 0.12)[0]
    return Scenario(kind, starts, goals, obstacles, r_obs)
