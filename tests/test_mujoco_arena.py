"""MuJoCo arena tests; skipped unless ``mujoco`` is installed."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from madbp.sim.executor import ExecConfig, run_episode  # noqa: E402
from madbp.sim.mujoco_arena import MuJoCoArena, MuJoCoDiffDriveArena, build_mjcf  # noqa: E402
from madbp.sim.planners import StraightLinePlanner  # noqa: E402
from madbp.sim.scenario import make_scenario  # noqa: E402


def test_mjcf_compiles():
    import mujoco

    for dd in (False, True):
        mujoco.MjModel.from_xml_string(build_mjcf(4, np.array([[0.0, 0.0]]), 0.08, 0.05, 1.0, 0.005, dd))


@pytest.mark.parametrize("cls", [MuJoCoArena, MuJoCoDiffDriveArena])
def test_episode_runs_and_agents_reach_goals(cls):
    sc = make_scenario("open", 3, np.random.default_rng(0))
    r = run_episode(cls(sc), StraightLinePlanner(8), ExecConfig(subgoal_cap=0.3), 0)
    assert r.n_goals_reached >= 2
    assert r.agent_agent_collisions == 0


def test_contact_prevents_overlap():
    sc = make_scenario("open", 2, np.random.default_rng(2))
    sc.starts[:] = [[-0.2, 0.0], [0.2, 0.0]]
    a = MuJoCoArena(sc)
    for _ in range(60):  # drive them at each other
        cmd = np.zeros((a.max_agents, 2))
        cmd[0], cmd[1] = [3.0, 0.0], [-3.0, 0.0]
        a.step(cmd)
    assert np.linalg.norm(a.pos[0] - a.pos[1]) >= 2 * a.radius - 0.02
