"""MuJoCo arena: same interface as :class:`PointMassArena`, physics from MuJoCo.

Each slot is a sphere with two slide joints (x, y); commanded accelerations are
applied as forces (``xfrc_applied = m * a``). Obstacles are static cylinders,
the arena boundary four static boxes. Contacts are resolved by MuJoCo, so agent
agent / agent-obstacle collisions are physical, not just counted. Inactive slots
are parked far outside the arena with contacts disabled, which makes dynamic
add/remove a pose reset plus a ``geom_contype`` toggle.

Not exercised in the authoring environment (no ``mujoco`` wheel was available);
``tests/test_mujoco_arena.py`` skips unless MuJoCo is installed.
"""

from __future__ import annotations

import numpy as np

from madbp.sim.arena import PointMassArena
from madbp.sim.scenario import Scenario

_PARK = 50.0


def build_mjcf(max_agents: int, obstacles: np.ndarray, obs_r: float, radius: float,
               bound: float, timestep: float) -> str:
    parts = [
        f'<mujoco model="madbp_arena"><option timestep="{timestep}" gravity="0 0 0" '
        'integrator="implicitfast"/>',
        '<default><geom friction="0 0 0" solref="0.005 1" solimp="0.95 0.99 0.001"/></default>',
        "<worldbody>",
    ]
    h = 0.5
    t = 0.05
    for name, pos, size in [
        ("wall_n", (0, bound + t, 0), (bound + 2 * t, t, h)),
        ("wall_s", (0, -bound - t, 0), (bound + 2 * t, t, h)),
        ("wall_e", (bound + t, 0, 0), (t, bound + 2 * t, h)),
        ("wall_w", (-bound - t, 0, 0), (t, bound + 2 * t, h)),
    ]:
        parts.append(f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
                     f'size="{size[0]} {size[1]} {size[2]}"/>')
    for k, c in enumerate(np.asarray(obstacles).reshape(-1, 2)):
        parts.append(f'<geom name="obs{k}" type="cylinder" pos="{c[0]} {c[1]} 0" '
                     f'size="{obs_r} {h}"/>')
    for i in range(max_agents):
        parts.append(
            f'<body name="a{i}" pos="{_PARK + 3 * i} {_PARK} 0">'
            f'<joint name="x{i}" type="slide" axis="1 0 0"/>'
            f'<joint name="y{i}" type="slide" axis="0 1 0"/>'
            f'<geom name="g{i}" type="sphere" size="{radius}" mass="1" contype="0" conaffinity="0"/>'
            "</body>"
        )
    parts.append("</worldbody></mujoco>")
    return "".join(parts)


class MuJoCoArena(PointMassArena):
    def __init__(self, scenario: Scenario, max_agents: int = 10, dt: float = 0.05,
                 v_max: float = 0.6, a_max: float = 3.0, substeps: int = 10) -> None:
        import mujoco

        self._mj = mujoco
        super().__init__(scenario, max_agents, dt, v_max, a_max)
        self.substeps = substeps
        xml = build_mjcf(max_agents, self.obstacles, self.obs_r, self.radius, self.bound, dt / substeps)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._gid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(max_agents)]
        self._bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"a{i}") for i in range(max_agents)]
        self._qadr = [int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"x{i}")])
                      for i in range(max_agents)]
        self._vadr = [int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"x{i}")])
                      for i in range(max_agents)]
        for i in range(max_agents):
            self._on_slot_change(i)
        mujoco.mj_forward(self.model, self.data)

    # joint qpos is relative to the body's parked frame -> subtract the park offset
    def _park_xy(self, i: int) -> np.ndarray:
        return np.array([_PARK + 3 * i, _PARK])

    def _on_slot_change(self, slot: int) -> None:
        on = bool(self.active[slot])
        g = self._gid[slot]
        self.model.geom_contype[g] = 1 if on else 0
        self.model.geom_conaffinity[g] = 1 if on else 0
        q, v = self._qadr[slot], self._vadr[slot]
        if on:
            rel = self.pos[slot] - self._park_xy(slot)
            self.data.qpos[q:q + 2] = rel
            self.data.qvel[v:v + 2] = self.vel[slot]
        else:
            self.data.qpos[q:q + 2] = 0.0
            self.data.qvel[v:v + 2] = 0.0

    def _integrate(self, acc: np.ndarray) -> None:
        mj = self._mj
        for i in range(self.max_agents):
            q, v = self._qadr[i], self._vadr[i]
            if self.active[i]:
                self.data.qpos[q:q + 2] = self.pos[i] - self._park_xy(i)
                self.data.qvel[v:v + 2] = self.vel[i]
                self.data.xfrc_applied[self._bid[i], :2] = acc[i] * 1.0  # mass = 1
            else:
                self.data.xfrc_applied[self._bid[i], :2] = 0.0
        for _ in range(self.substeps):
            mj.mj_step(self.model, self.data)
            for i in np.flatnonzero(self.active):  # speed limit inside the sub-loop
                v = self._vadr[i]
                sp = np.linalg.norm(self.data.qvel[v:v + 2])
                if sp > self.v_max:
                    self.data.qvel[v:v + 2] *= self.v_max / sp
        for i in np.flatnonzero(self.active):
            q, v = self._qadr[i], self._vadr[i]
            self.pos[i] = self.data.qpos[q:q + 2] + self._park_xy(i)
            self.vel[i] = self.data.qvel[v:v + 2]
