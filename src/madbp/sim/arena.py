"""Point-mass arena with a fixed number of agent slots (numpy, no extra deps).

``PointMassArena`` is a double-integrator simulator with speed/acceleration
limits, disc obstacles, wall bounds and non-penetration resolution. It is the
test double for, and the base class of, :class:`madbp.sim.mujoco_arena.MuJoCoArena`
(which swaps only the physics integrator). Slots make dynamic teams trivial:
``add_agent`` takes a free slot, ``remove_agent`` frees one, and ``active``
is exactly the mask the planner consumes.
"""

from __future__ import annotations

import numpy as np

from madbp.sim.scenario import Scenario


class PointMassArena:
    def __init__(self, scenario: Scenario, max_agents: int = 10, dt: float = 0.05,
                 v_max: float = 0.6, a_max: float = 3.0) -> None:
        self.sc = scenario
        self.max_agents = max_agents
        self.dt = dt
        self.v_max = v_max
        self.a_max = a_max
        self.radius = scenario.agent_radius
        self.bound = scenario.bound
        self.obstacles = np.asarray(scenario.obstacles, dtype=float).reshape(-1, 2)
        self.obs_r = scenario.obstacle_radius

        n = scenario.n_agents
        assert n <= max_agents
        self.pos = np.zeros((max_agents, 2))
        self.vel = np.zeros((max_agents, 2))
        self.goals = np.zeros((max_agents, 2))
        self.active = np.zeros(max_agents, dtype=bool)
        self.pos[:n] = scenario.starts
        self.goals[:n] = scenario.goals
        self.active[:n] = True
        self.t = 0
        self._in_contact_aa: set[tuple[int, int]] = set()
        self._in_contact_ao: set[tuple[int, int]] = set()
        self.agent_agent_collisions = 0  # onset events
        self.agent_obstacle_collisions = 0

    # ---- control interface (used by the executor) ----------------------
    control = "accel"

    def track(self, target: np.ndarray, kp: float, kv: float) -> np.ndarray:
        """PD law on position/velocity -> commanded acceleration ``[N,2]``."""
        return kp * (target - self.pos) - kv * self.vel

    def heading(self) -> np.ndarray:
        """Display heading ``[N]`` (direction of motion for a holonomic point mass)."""
        return np.arctan2(self.vel[:, 1], self.vel[:, 0])

    # ---- team changes --------------------------------------------------
    def add_agent(self, start, goal) -> int | None:
        free = np.flatnonzero(~self.active)
        if len(free) == 0:
            return None
        s = int(free[0])
        self.pos[s], self.vel[s], self.goals[s] = start, 0.0, goal
        self.active[s] = True
        self._on_slot_change(s)
        return s

    def remove_agent(self, slot: int) -> None:
        self.active[slot] = False
        self.vel[slot] = 0.0
        self._on_slot_change(slot)

    def _on_slot_change(self, slot: int) -> None:  # hook for MuJoCo
        pass

    # ---- physics ---------------------------------------------------------
    def step(self, acc: np.ndarray) -> None:
        acc = np.where(self.active[:, None], acc, 0.0)
        n = np.linalg.norm(acc, axis=1, keepdims=True)
        acc = acc * np.minimum(1.0, self.a_max / np.maximum(n, 1e-9))
        self._integrate(acc)
        self._account_contacts()
        self.t += 1

    def _integrate(self, acc: np.ndarray) -> None:
        self.vel = self.vel + acc * self.dt
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = self.vel * np.minimum(1.0, self.v_max / np.maximum(sp, 1e-9))
        self.pos = self.pos + self.vel * self.dt
        self._resolve()

    def _resolve(self) -> None:
        lim = self.bound - self.radius
        over = np.abs(self.pos) > lim
        self.pos = np.clip(self.pos, -lim, lim)
        self.vel = np.where(over, 0.0, self.vel)
        idx = np.flatnonzero(self.active)
        if len(idx) == 0:
            return
        if len(self.obstacles):
            m = self.radius + self.obs_r
            for _ in range(2):
                d = self.pos[idx][:, None, :] - self.obstacles[None]  # [n,K,2]
                dist = np.linalg.norm(d, axis=-1)
                hit = dist < m
                if not hit.any():
                    break
                for ii, kk in zip(*np.nonzero(hit)):
                    i = idx[ii]
                    nrm = d[ii, kk] / max(dist[ii, kk], 1e-9)
                    self.pos[i] = self.obstacles[kk] + nrm * m
                    self.vel[i] -= min(0.0, self.vel[i] @ nrm) * nrm
        if len(idx) > 1:
            for _ in range(2):
                p = self.pos[idx]
                d = p[:, None] - p[None]
                dist = np.linalg.norm(d, axis=-1)
                np.fill_diagonal(dist, np.inf)
                a_, b_ = np.nonzero(np.triu(dist < 2 * self.radius, 1))
                if len(a_) == 0:
                    break
                for x, y in zip(a_, b_):
                    i, j = idx[x], idx[y]
                    dv = self.pos[i] - self.pos[j]
                    dd = np.linalg.norm(dv)
                    nrm = dv / dd if dd > 1e-9 else np.array([1.0, 0.0])
                    push = 0.5 * (2 * self.radius - dd) * nrm
                    self.pos[i] += push
                    self.pos[j] -= push

    # ---- collision accounting (onset events, tolerance for resolver) ----
    def _account_contacts(self) -> None:
        tol = 1e-3
        idx = np.flatnonzero(self.active)
        now = set()
        if len(idx) > 1:
            p = self.pos[idx]
            dist = np.linalg.norm(p[:, None] - p[None], axis=-1)
            for x, y in zip(*np.nonzero(np.triu(dist <= 2 * self.radius + tol, 1))):
                now.add((int(idx[x]), int(idx[y])))
        self.agent_agent_collisions += len(now - self._in_contact_aa)
        self._in_contact_aa = now
        now_o = set()
        if len(idx) and len(self.obstacles):
            d = np.linalg.norm(self.pos[idx][:, None] - self.obstacles[None], axis=-1)
            for x, k in zip(*np.nonzero(d <= self.radius + self.obs_r + tol)):
                now_o.add((int(idx[x]), int(k)))
        self.agent_obstacle_collisions += len(now_o - self._in_contact_ao)
        self._in_contact_ao = now_o

    # ---- observation -----------------------------------------------------
    def min_pair_distance(self) -> float:
        idx = np.flatnonzero(self.active)
        if len(idx) < 2:
            return np.inf
        p = self.pos[idx]
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        return float(d.min())


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class DiffDriveArena(PointMassArena):
    """Differential-drive (unicycle) robots with a circular footprint.

    State per slot: ``(x, y, theta)``. Commands are ``(v, omega)``; they pass through
    first-order acceleration limits and a *wheel-speed* limit
    ``max(|v ± omega*b/2|) <= wheel_max`` (``b`` = wheel track). Motion is non-holonomic:
    no lateral slip. Contacts are resolved by the same non-penetration step as the
    point-mass arena (and by MuJoCo in :class:`MuJoCoDiffDriveArena`).
    """

    control = "unicycle"

    def __init__(self, scenario: Scenario, max_agents: int = 10, dt: float = 0.05,
                 wheel_max: float = 0.6, track_width: float = 0.08,
                 a_v: float = 1.5, a_w: float = 12.0) -> None:
        super().__init__(scenario, max_agents, dt, v_max=wheel_max)
        self.wheel_max = wheel_max
        self.b = track_width
        self.a_v, self.a_w = a_v, a_w
        self.theta = np.zeros(max_agents)
        self.v = np.zeros(max_agents)
        self.w = np.zeros(max_agents)
        # initial heading: toward the goal (robots are placed facing their task)
        d = self.goals - self.pos
        self.theta = np.arctan2(d[:, 1], d[:, 0])
        self.theta[~self.active] = 0.0

    def add_agent(self, start, goal):
        s = super().add_agent(start, goal)
        if s is not None:
            d = np.asarray(goal) - np.asarray(start)
            self.theta[s], self.v[s], self.w[s] = np.arctan2(d[1], d[0]), 0.0, 0.0
            self._on_slot_change(s)
        return s

    def remove_agent(self, slot: int) -> None:
        super().remove_agent(slot)
        self.v[slot] = self.w[slot] = 0.0

    def heading(self) -> np.ndarray:
        return self.theta.copy()

    def limits(self, v: np.ndarray, w: np.ndarray):
        """Clip ``(v, w)`` so both wheel speeds stay within ``wheel_max``."""
        need = np.abs(v) + np.abs(w) * self.b / 2
        s = np.minimum(1.0, self.wheel_max / np.maximum(need, 1e-9))
        return v * s, w * s

    def wheel_speeds(self):
        return self.v - self.w * self.b / 2, self.v + self.w * self.b / 2

    def track(self, target: np.ndarray, kp: float = 6.0, kv: float = 6.0) -> np.ndarray:
        """Unicycle waypoint tracker -> ``[N,2]`` of ``(v_cmd, omega_cmd)``.

        Heading error ``alpha`` steers the robot toward the waypoint; forward speed is
        scaled by ``cos(alpha)`` (zero when the waypoint is behind, i.e. rotate in place).
        ``kp`` is the linear-speed gain, ``kv`` the heading gain.
        """
        d = target - self.pos
        dist = np.linalg.norm(d, axis=1)
        alpha = _wrap(np.arctan2(d[:, 1], d[:, 0]) - self.theta)
        v_cmd = np.minimum(self.wheel_max, kp * dist) * np.maximum(np.cos(alpha), 0.0)
        gain = np.minimum(1.0, dist / 0.02)  # avoid jitter when on the waypoint
        w_cmd = np.clip(kv * alpha * gain, -4.0, 4.0)
        return np.stack([v_cmd, w_cmd], axis=1)

    def step(self, cmd: np.ndarray) -> None:
        act = self.active
        v_c = np.where(act, cmd[:, 0], 0.0)
        w_c = np.where(act, cmd[:, 1], 0.0)
        v_c, w_c = self.limits(v_c, w_c)
        self.v = self.v + np.clip(v_c - self.v, -self.a_v * self.dt, self.a_v * self.dt)
        self.w = self.w + np.clip(w_c - self.w, -self.a_w * self.dt, self.a_w * self.dt)
        self.v, self.w = self.limits(self.v, self.w)
        self._integrate(None)
        self._account_contacts()
        self.t += 1

    def _integrate(self, _cmd) -> None:
        self.theta = _wrap(self.theta + self.w * self.dt)
        self.vel = np.stack([self.v * np.cos(self.theta), self.v * np.sin(self.theta)], axis=1)
        self.pos = self.pos + self.vel * self.dt
        self._resolve()
