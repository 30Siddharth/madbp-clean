"""MA-DBP evaluation runner on the simulation stack (replaces the VMAS-only one).

Plans with the paper-faithful sampler inside the closed-loop executor
(:mod:`madbp.sim.executor`). The arena is the numpy point-mass arena by default;
``scenario`` selects the layout generator (``open`` / ``obstacles`` / ``barrier`` /
``corridor``). For MuJoCo physics or VMAS-rendered frames use ``scripts/run_sim.py``.
"""

from __future__ import annotations

import numpy as np

from madbp.evaluation.metrics import EpisodeMetrics
from madbp.sim.arena import PointMassArena
from madbp.sim.executor import ExecConfig, run_episode
from madbp.sim.planners import MADBPPlanner
from madbp.sim.scenario import make_scenario


class MADPRunner:
    def __init__(self, scenario, n_agents, max_steps, collision_radius, checkpoint, device,
                 arena: str = "kinematic") -> None:
        if checkpoint is None:
            raise ValueError("MADP runner requires --checkpoint")
        self.kind = scenario
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.planner = MADBPPlanner(checkpoint, device)
        self.arena_kind = arena
        self.cfg = ExecConfig(
            max_steps=max_steps,
            replan_every=self.planner.horizon - 1,
            subgoal_cap=self.planner.window_disp_world or None,
        )

    def run_episode(self, seed: int) -> EpisodeMetrics:
        rng = np.random.default_rng(seed)
        sc = make_scenario(self.kind, self.n_agents, rng)
        if self.arena_kind == "mujoco":
            from madbp.sim.mujoco_arena import MuJoCoArena

            arena = MuJoCoArena(sc, max_agents=self.planner.max_agents)
        else:
            arena = PointMassArena(sc, max_agents=self.planner.max_agents)
        r = run_episode(arena, self.planner, self.cfg, seed)
        T = max(r.makespan_steps, 1)
        n_coll = r.agent_agent_collisions + r.agent_obstacle_collisions
        opt = float(np.linalg.norm(sc.goals - sc.starts, axis=1).mean())
        spl = (opt / max(opt, r.path_length_mean)) if r.success else 0.0
        return EpisodeMetrics(
            method="madp", scenario=self.kind, n_agents=self.n_agents, seed=seed,
            success=r.success,
            final_goal_distance=float(np.linalg.norm(arena.pos - arena.goals, axis=1)[arena.active].mean()),
            n_collisions=n_coll, collision_rate=n_coll / T,
            path_length_mean=r.path_length_mean, makespan=r.makespan_steps,
            mean_acceleration=0.0, mean_jerk=r.mean_jerk,
            plan_runtime_s=r.plan_time_mean_s * r.n_replans,
            time_to_goal_mean=float(r.makespan_steps), spl=spl,
        )
