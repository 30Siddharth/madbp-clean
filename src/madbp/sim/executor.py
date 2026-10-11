"""Receding-horizon closed-loop executor with dynamic agent add/remove.

Loop (every ``replan_every`` control steps, and immediately after a team change)::

    sub-goal_i = pos_i + clip(goal_i - pos_i, cap)        # cap ~ training window displacement
    window     = planner.plan(frame, pos, sub-goal, active)   # [N,H,2]
    track window waypoints 1..K with a PD controller on the arena

The arena owns physics; the planner never sees velocities (the paper plans positions only).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from madbp.sim.render import render_scenario


@dataclass
class ExecConfig:
    max_steps: int = 600
    replan_every: int = 7  # waypoints executed per plan (<= H-1)
    steps_per_waypoint: int = 4
    goal_tol: float = 0.05
    subgoal_cap: float | None = None  # world units; None -> no cap (use goal directly)
    kp: float = 12.0
    kv: float = 5.0
    lin_gain: float = 6.0  # unicycle tracker: speed per metre of waypoint distance
    ang_gain: float = 6.0  # unicycle tracker: rad/s per rad of heading error
    frame_size: int = 64
    stall_steps: int = 0  # 0 disables early stop


@dataclass
class EpisodeResult:
    planner: str
    kind: str
    n_agents0: int
    seed: int
    success: bool
    n_goals_total: int
    n_goals_reached: int
    agent_agent_collisions: int
    agent_obstacle_collisions: int
    makespan_steps: int
    path_length_mean: float
    mean_jerk: float
    tracking_error_mean: float
    plan_time_mean_s: float
    n_replans: int
    min_pair_distance: float
    events_applied: int
    extras: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extras"))
        return d


def _frame(arena, size):
    idx = np.flatnonzero(arena.active)
    return render_scenario(arena.pos[idx], arena.goals[idx], arena.obstacles,
                           arena.obs_r, arena.radius, arena.bound, size)


def run_episode(arena, planner, cfg: ExecConfig, seed: int = 0, frame_fn=None, recorder=None) -> EpisodeResult:
    sc = arena.sc
    events = sorted(sc.events, key=lambda e: e.step)
    ev_i, events_applied = 0, 0
    N = arena.max_agents
    reached = np.zeros(N, dtype=bool)
    ever_goal = arena.active.copy()  # slots whose goal counts
    path_len = np.zeros(N)
    traj = [arena.pos.copy()]
    track_err, plan_times = [], []
    waypoints, wp_k, wp_steps = None, 0, 0
    need_plan = True
    frame, refresh_frame = None, False
    step = 0
    n_replans = 0
    min_pair = np.inf
    start_agents = int(arena.active.sum())

    while step < cfg.max_steps:
        # --- dynamic team changes ---
        while ev_i < len(events) and events[ev_i].step <= step:
            e = events[ev_i]
            if e.kind == "add":
                s = arena.add_agent(np.asarray(e.start, float), np.asarray(e.goal, float))
                if s is not None:
                    ever_goal[s] = True
                    reached[s] = False
                    path_len[s] = 0.0
            elif e.kind == "remove" and e.slot is not None:
                arena.remove_agent(e.slot)
                ever_goal[e.slot] = False  # a removed agent no longer owes a goal
            events_applied += 1
            ev_i += 1
            need_plan = True
            refresh_frame = True

        if need_plan or wp_k >= cfg.replan_every:
            pos, goals, act = arena.pos.copy(), arena.goals.copy(), arena.active.copy()
            sub = goals.copy()
            if cfg.subgoal_cap is not None:
                d = goals - pos
                n = np.linalg.norm(d, axis=1, keepdims=True)
                sub = pos + d * np.minimum(1.0, cfg.subgoal_cap / np.maximum(n, 1e-9))
            if frame is None or refresh_frame:
                # training used frame 0 of each episode only, so the scenario image is
                # fixed at episode start and re-rendered only when the team changes
                frame = frame_fn(arena) if frame_fn else _frame(arena, cfg.frame_size)
                refresh_frame = False
            t0 = time.perf_counter()
            waypoints = planner.plan(frame, pos, sub, act)
            plan_times.append(getattr(planner, "last_plan_time", None) or (time.perf_counter() - t0))
            n_replans += 1
            wp_k, wp_steps, need_plan = 1, 0, False

        # --- PD tracking of the current waypoint ---
        k = min(wp_k, waypoints.shape[1] - 1)
        target = waypoints[:, k, :]
        err = target - arena.pos
        cmd = arena.track(target, *_gains(arena, cfg))
        prev = arena.pos.copy()
        arena.step(cmd)
        if recorder is not None:
            recorder.capture(arena, waypoints, wp_k)
        path_len += np.where(arena.active, np.linalg.norm(arena.pos - prev, axis=1), 0.0)
        track_err.append(float(np.linalg.norm(err[arena.active], axis=1).mean()) if arena.active.any() else 0.0)
        traj.append(arena.pos.copy())
        min_pair = min(min_pair, arena.min_pair_distance())
        wp_steps += 1
        if wp_steps >= cfg.steps_per_waypoint:
            wp_steps, wp_k = 0, wp_k + 1
        step += 1

        d_goal = np.linalg.norm(arena.pos - arena.goals, axis=1)
        newly = arena.active & ~reached & (d_goal < cfg.goal_tol)
        reached |= newly
        pending = ever_goal & arena.active & ~reached
        if not pending.any() and ev_i >= len(events):
            break

    n_total = int(ever_goal.sum())
    n_reached = int((reached & ever_goal).sum())
    traj_a = np.stack(traj)  # [T+1, N, 2]
    jerk = _mean_jerk(traj_a, arena.dt, ever_goal)
    return EpisodeResult(
        planner=getattr(planner, "name", type(planner).__name__),
        kind=sc.name, n_agents0=start_agents, seed=seed,
        success=bool(n_total > 0 and n_reached == n_total and ev_i >= len(events)),
        n_goals_total=n_total, n_goals_reached=n_reached,
        agent_agent_collisions=arena.agent_agent_collisions,
        agent_obstacle_collisions=arena.agent_obstacle_collisions,
        makespan_steps=step,
        path_length_mean=float(path_len[ever_goal].mean()) if ever_goal.any() else 0.0,
        mean_jerk=jerk,
        tracking_error_mean=float(np.mean(track_err)) if track_err else 0.0,
        plan_time_mean_s=float(np.mean(plan_times)) if plan_times else 0.0,
        n_replans=n_replans,
        min_pair_distance=float(min_pair),
        events_applied=events_applied,
    )


def _gains(arena, cfg):
    # point mass: PD gains (kp, kv); unicycle: (linear-speed gain, heading gain)
    return (cfg.kp, cfg.kv) if arena.control == "accel" else (cfg.lin_gain, cfg.ang_gain)


def _mean_jerk(traj, dt, mask):
    if traj.shape[0] < 4 or not mask.any():
        return 0.0
    p = traj[:, mask]
    j = np.diff(p, n=3, axis=0) / dt**3
    return float(np.linalg.norm(j, axis=-1).mean())
