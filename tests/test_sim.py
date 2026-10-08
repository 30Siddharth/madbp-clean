"""Torch-free tests of the closed-loop simulation stack."""

from __future__ import annotations

import numpy as np

from madbp.sim.arena import PointMassArena
from madbp.sim.executor import ExecConfig, run_episode
from madbp.sim.planners import StraightLinePlanner
from madbp.sim.scenario import Event, make_scenario
from madbp.training.curriculum import linear_stage_curriculum


def test_curriculum_staircase():
    vals = [linear_stage_curriculum(e, 400, 4) for e in range(400)]
    assert vals[0] == 1 and vals[-1] == 4 and vals == sorted(vals)


def test_straight_line_open_scene_succeeds():
    sc = make_scenario("open", 3, np.random.default_rng(0))
    r = run_episode(PointMassArena(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3), 0)
    assert r.success and r.n_goals_reached == 3


def test_dynamic_add_remove_bookkeeping():
    sc = make_scenario("open", 3, np.random.default_rng(1))
    sc.events = [Event(30, "add", start=(0.9, 0.9), goal=(-0.9, 0.5)), Event(60, "remove", slot=0)]
    r = run_episode(PointMassArena(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3), 0)
    assert r.events_applied == 2
    assert r.n_goals_total == 3  # 3 initial + 1 added - 1 removed


def test_arena_resolves_overlap_and_counts_collision():
    sc = make_scenario("open", 2, np.random.default_rng(2))
    sc.starts[:] = [[0.0, 0.0], [0.04, 0.0]]
    arena = PointMassArena(sc)
    arena.step(np.zeros((10, 2)))
    assert np.linalg.norm(arena.pos[0] - arena.pos[1]) >= 2 * arena.radius - 1e-6
    assert arena.agent_agent_collisions == 1


def test_barrier_baseline_is_nontrivial():
    """The naive baseline must not trivially solve the barrier scene."""
    wins = 0
    for s in range(5):
        sc = make_scenario("barrier", 3, np.random.default_rng(s))
        wins += run_episode(PointMassArena(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3), s).success
    assert wins < 5
