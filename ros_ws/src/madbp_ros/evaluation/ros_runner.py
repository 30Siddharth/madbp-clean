"""RunnerProtocol implementation that drives evaluation via live ROS 2 topics.

This runner bridges the existing ``madbp-eval`` CLI into a live ROS session:
  * Sends a PlanMultiAgentPath action goal to the MA-DBP planner node.
  * Monitors /robot_{i}/odom topics to collect ground-truth positions.
  * Computes the same EpisodeMetrics as MADPRunner so results are
    directly comparable to the VMAS-based numbers from Experiment 1.

Usage (requires a running ROS 2 session with lichtblik_bridge or
mujoco_bridge and madbp_planner_node):

    madbp-eval --method ros \\
               --scenario navigation_v2 \\
               --n-agents 4 \\
               --seed 0 \\
               --n-episodes 20 \\
               --out results/ros_navigation_v2_n4.csv

The runner blocks until the action completes then executes the episode
by polling odometry.
"""

from __future__ import annotations

import time
from threading import Event

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav_msgs.msg import Odometry

from madbp.evaluation.metrics import (
    EpisodeMetrics,
    compute_kinematics,
    compute_spl,
    count_pairwise_collisions,
)

try:
    from madbp_interfaces.action import PlanMultiAgentPath  # type: ignore
except ImportError:
    PlanMultiAgentPath = None  # type: ignore


class ROSRunner:
    """Drives an episode via live ROS 2 topics. Implements RunnerProtocol."""

    def __init__(
        self,
        scenario: str,
        n_agents: int,
        max_steps: int,
        collision_radius: float,
        checkpoint: str | None,
        device: str,
        # ROS-specific
        odom_hz: float = 20.0,
        action_timeout_s: float = 30.0,
    ) -> None:
        self.scenario = scenario
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.collision_radius = collision_radius
        self.odom_hz = odom_hz
        self.action_timeout_s = action_timeout_s

        if not rclpy.ok():
            rclpy.init()

        self._node = Node("madbp_ros_runner")

        # Odometry subscribers — one per agent
        self._latest_pos: list[np.ndarray] = [np.zeros(2) for _ in range(n_agents)]
        for i in range(n_agents):
            self._node.create_subscription(
                Odometry,
                f"/robot_{i}/odom",
                lambda msg, idx=i: self._odom_cb(msg, idx),
                10,
            )

        if PlanMultiAgentPath is not None:
            self._action_client = ActionClient(
                self._node, PlanMultiAgentPath, "plan_multi_agent_path"
            )
        else:
            self._action_client = None

    def _odom_cb(self, msg: Odometry, idx: int) -> None:
        p = msg.pose.pose.position
        self._latest_pos[idx] = np.array([p.x, p.y], dtype=np.float32)

    def run_episode(self, seed: int) -> EpisodeMetrics:
        """Send a plan request and collect odometry for one episode."""
        import rclpy.spin_once

        rng = np.random.default_rng(seed)
        # Random start/goal in normalised [-1, 1] space
        starts_np = rng.uniform(-0.9, 0.9, (self.n_agents, 2)).astype(np.float32)
        goals_np = rng.uniform(-0.9, 0.9, (self.n_agents, 2)).astype(np.float32)

        if self._action_client is not None:
            goal_msg = PlanMultiAgentPath.Goal()
            goal_msg.n_agents = self.n_agents
            from geometry_msgs.msg import Point as GPoint
            goal_msg.starts = [GPoint(x=float(s[0]), y=float(s[1]), z=0.0) for s in starts_np]
            goal_msg.goals = [GPoint(x=float(g[0]), y=float(g[1]), z=0.0) for g in goals_np]

            done_event = Event()
            future = self._action_client.send_goal_async(goal_msg)
            future.add_done_callback(lambda _: done_event.set())

            deadline = time.time() + self.action_timeout_s
            while not done_event.is_set() and time.time() < deadline:
                rclpy.spin_once(self._node, timeout_sec=0.1)

        # Collect odometry
        positions: list[np.ndarray] = []
        dt = 1.0 / self.odom_hz
        reached = np.zeros(self.n_agents, dtype=bool)
        time_to_goal = np.full(self.n_agents, self.max_steps, dtype=np.int32)
        success = False

        for step in range(self.max_steps):
            rclpy.spin_once(self._node, timeout_sec=dt)
            pos = np.stack([self._latest_pos[i].copy() for i in range(self.n_agents)])
            positions.append(pos)
            dist = np.linalg.norm(pos - goals_np, axis=-1)
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
        final_dist = float(np.linalg.norm(positions_arr[-1] - goals_np, axis=-1).mean())
        T = positions_arr.shape[0]
        optimal = float(np.linalg.norm(goals_np - starts_np, axis=-1).mean())
        spl = compute_spl(success, optimal, path_len)

        return EpisodeMetrics(
            method="ros",
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
            plan_runtime_s=0.0,  # measured server-side
            time_to_goal_mean=float(time_to_goal.mean()),
            spl=float(spl),
        )
