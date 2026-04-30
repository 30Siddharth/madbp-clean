"""Reference MA-DBP evaluation runner.

Wraps the moving-window sampler in the VMAS Navigation environment with a
PD position-velocity controller (``F = Kp * e_p + Kv * e_v``) tracking the
sampled trajectory points. The environment is constructed lazily so the
package can be imported without VMAS installed (CI uses a tiny CPU smoke
test that never instantiates this runner).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from madbp.evaluation.metrics import (
    EpisodeMetrics,
    compute_kinematics,
    compute_spl,
    count_pairwise_collisions,
)
from madbp.models.diffusion import MultiAgentDiffusionPlanner


class MADPRunner:
    """Plans with MA-DBP, executes via PD on a VMAS Navigation env."""

    def __init__(
        self,
        scenario: str,
        n_agents: int,
        max_steps: int,
        collision_radius: float,
        checkpoint: str | None,
        device: str,
        # PD controller gains
        kp: float = 30.0,
        kv: float = 2.5,
        max_force: float = 1.0,
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.collision_radius = collision_radius
        self.device = torch.device(device)
        self.kp = kp
        self.kv = kv
        self.max_force = max_force

        if checkpoint is None:
            raise ValueError("MADP runner requires --checkpoint")
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("cfg", {})
        self.model = MultiAgentDiffusionPlanner(
            max_agents=cfg.get("max_agents", 10),
            horizon=cfg.get("horizon", 8),
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=cfg.get("diffusion_steps", 150),
            schedule_type=cfg.get("schedule", "linear"),
            cfg_dropout_prob=cfg.get("cfg_dropout_prob", 0.0),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self._env: Any | None = None  # constructed lazily

    def _make_env(self, seed: int):
        from vmas import make_env  # imported lazily so CI can run without VMAS
        return make_env(
            scenario=self.scenario,
            num_envs=1,
            device=self.device,
            continuous_actions=True,
            max_steps=self.max_steps,
            n_agents=self.n_agents,
            seed=seed,
        )

    def run_episode(self, seed: int) -> EpisodeMetrics:
        env = self._make_env(seed)
        obs = env.reset()
        # Initial state from agent objects
        starts = torch.stack([a.state.pos for a in env.agents], dim=1)  # [1, n, 2]
        goals = torch.stack([a.goal.state.pos for a in env.agents], dim=1)  # [1, n, 2]
        frame = env.render(mode="rgb_array")
        frame_t = torch.tensor(frame, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Pad to max_agents
        max_agents = self.model.max_agents
        starts_p = torch.zeros(1, max_agents, 2, device=self.device)
        goals_p = torch.zeros(1, max_agents, 2, device=self.device)
        starts_p[0, : self.n_agents] = starts[0]
        goals_p[0, : self.n_agents] = goals[0]
        n_agents_t = torch.tensor([self.n_agents], device=self.device)

        t0 = time.perf_counter()
        traj = self.model.sample_moving_window(
            frame_t, starts_p, goals_p, n_agents_t,
            full_horizon_length=self.max_steps,
            horizon_size=self.model.horizon,
            max_step_size=0.05,
        )
        plan_time = time.perf_counter() - t0

        positions: list[np.ndarray] = []
        time_to_goal = np.full(self.n_agents, self.max_steps, dtype=np.int32)
        reached = np.zeros(self.n_agents, dtype=bool)
        success = False

        for step in range(self.max_steps):
            cur = torch.stack([a.state.pos for a in env.agents], dim=1)  # [1, n, 2]
            vel = torch.stack([a.state.vel for a in env.agents], dim=1)
            target = traj[0, : self.n_agents, min(step, traj.size(2) - 1), :]
            pos_err = target - cur[0]
            vel_err = -vel[0]
            forces = (self.kp * pos_err + self.kv * vel_err).clamp(-self.max_force, self.max_force)
            actions = [forces[i].unsqueeze(0) for i in range(self.n_agents)]
            obs, rew, done, info = env.step(actions)

            cur_np = cur[0].cpu().numpy()
            positions.append(cur_np)

            dist = np.linalg.norm(cur_np - goals[0].cpu().numpy(), axis=-1)
            for i in range(self.n_agents):
                if not reached[i] and dist[i] < self.collision_radius:
                    reached[i] = True
                    time_to_goal[i] = step
            if reached.all():
                success = True
                break

        positions_arr = np.stack(positions, axis=0)  # [T, n, 2]
        path_len, accel, jerk = compute_kinematics(positions_arr)
        n_coll = count_pairwise_collisions(positions_arr, min_dist=self.collision_radius)
        final_dist = float(np.linalg.norm(positions_arr[-1] - goals[0].cpu().numpy(), axis=-1).mean())
        T = positions_arr.shape[0]
        # Optimal length = straight-line start->goal
        starts_np = starts[0].cpu().numpy()
        goals_np = goals[0].cpu().numpy()
        optimal = float(np.linalg.norm(goals_np - starts_np, axis=-1).mean())
        spl = compute_spl(success, optimal, path_len)

        return EpisodeMetrics(
            method="madp",
            scenario=self.scenario,
            n_agents=self.n_agents,
            seed=seed,
            success=bool(success),
            final_goal_distance=final_dist,
            n_collisions=int(n_coll),
            collision_rate=float(n_coll) / max(T, 1),
            path_length_mean=float(path_len),
            makespan=int(T),
            mean_acceleration=float(accel),
            mean_jerk=float(jerk),
            plan_runtime_s=float(plan_time),
            time_to_goal_mean=float(time_to_goal.mean()),
            spl=float(spl),
        )
