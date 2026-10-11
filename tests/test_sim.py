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


def test_diffdrive_is_nonholonomic_and_respects_wheel_limit():
    from madbp.sim.arena import DiffDriveArena

    sc = make_scenario("open", 2, np.random.default_rng(0))
    sc.starts[:] = [[-0.5, 0.0], [0.5, 0.5]]
    sc.goals[:] = [[0.5, 0.0], [-0.5, 0.5]]
    a = DiffDriveArena(sc)
    th0 = a.theta[0]
    for _ in range(40):
        a.step(np.tile([[5.0, 0.0]], (a.max_agents, 1)))  # absurd forward command
        vl, vr = a.wheel_speeds()
        assert max(np.abs(vl[a.active]).max(), np.abs(vr[a.active]).max()) <= a.wheel_max + 1e-9
        # velocity always along the heading: no lateral slip
        lat = a.vel[:, 0] * -np.sin(a.theta) + a.vel[:, 1] * np.cos(a.theta)
        assert np.abs(lat[a.active]).max() < 1e-9
    assert abs(a.theta[0] - th0) < 1e-6  # straight command -> no turning


def test_diffdrive_closed_loop_open_scene():
    from madbp.sim.arena import DiffDriveArena

    sc = make_scenario("open", 3, np.random.default_rng(0))
    r = run_episode(DiffDriveArena(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3), 0)
    assert r.success


def test_video_roundtrip(tmp_path):
    from madbp.sim.arena import DiffDriveArena
    from madbp.sim.record import Recorder
    from madbp.sim.video import save_video

    sc = make_scenario("obstacles", 3, np.random.default_rng(1))
    rec = Recorder(every=5)
    run_episode(DiffDriveArena(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3, max_steps=60), 0, recorder=rec)
    out = save_video(rec, sc, str(tmp_path / "ep.gif"), fps=10)
    assert (tmp_path / "ep.gif").stat().st_size > 1000 and out.endswith(".gif")
