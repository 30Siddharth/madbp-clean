"""MuJoCo arenas: same interface as the numpy arenas, physics/contacts from MuJoCo.

* :class:`MuJoCoArena` - holonomic point-mass spheres (x/y slide joints), force actuated.
* :class:`MuJoCoDiffDriveArena` - differential-drive robots: a planar (x, y, yaw) body
  with a visual chassis, two wheels and a heading marker. The non-holonomic constraint is
  enforced by commanding the body velocity ``(v cos th, v sin th, omega)`` every sub-step
  after first-order acceleration and wheel-speed limits (no lateral slip, exactly like an
  ideal wheel-ground contact). MuJoCo resolves robot-robot / robot-obstacle / wall contacts.

Collision geometry is an invisible sphere of the footprint radius (robust sphere-cylinder
contacts); the visible chassis/wheels are contact-free.

Inactive slots are parked outside the arena with contacts disabled, so dynamic add/remove is
a pose reset plus a ``geom_contype`` toggle.

Not exercised in the authoring environment (no ``mujoco`` wheel available);
``tests/test_mujoco_arena.py`` skips unless MuJoCo is installed.
"""

from __future__ import annotations

import numpy as np

from madbp.sim.arena import DiffDriveArena, PointMassArena
from madbp.sim.scenario import Scenario

_PARK = 50.0
_COLORS = ["0.12 0.47 0.71", "1.0 0.5 0.05", "0.17 0.63 0.17", "0.84 0.15 0.16", "0.58 0.4 0.74",
           "0.55 0.34 0.29", "0.89 0.47 0.76", "0.5 0.5 0.5", "0.74 0.74 0.13", "0.09 0.75 0.81"]


def build_mjcf(max_agents: int, obstacles: np.ndarray, obs_r: float, radius: float,
               bound: float, timestep: float, diffdrive: bool = False, track_width: float = 0.08) -> str:
    p = [
        f'<mujoco model="madbp_arena"><option timestep="{timestep}" gravity="0 0 0" integrator="implicitfast"/>',
        '<visual><global offwidth="960" offheight="960"/><headlight ambient="0.5 0.5 0.5"/></visual>',
        '<default><geom friction="0 0 0" solref="0.005 1" solimp="0.95 0.99 0.001"/></default>',
        "<worldbody>",
        f'<light pos="0 0 3" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>',
        f'<geom name="floor" type="plane" size="{bound} {bound} 0.1" rgba="0.96 0.96 0.96 1" contype="0" conaffinity="0"/>',
        f'<camera name="top" pos="0 0 {2.6 * bound + 0.6}" xyaxes="1 0 0 0 1 0" fovy="45"/>',
    ]
    h, t = 0.5, 0.05
    for name, pos, size in [
        ("wall_n", (0, bound + t), (bound + 2 * t, t)),
        ("wall_s", (0, -bound - t), (bound + 2 * t, t)),
        ("wall_e", (bound + t, 0), (t, bound + 2 * t)),
        ("wall_w", (-bound - t, 0), (t, bound + 2 * t)),
    ]:
        p.append(f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} 0" size="{size[0]} {size[1]} {h}" '
                 'rgba="0.2 0.2 0.2 1"/>')
    for k, c in enumerate(np.asarray(obstacles).reshape(-1, 2)):
        p.append(f'<geom name="obs{k}" type="cylinder" pos="{c[0]} {c[1]} 0" size="{obs_r} {h}" rgba="0.35 0.35 0.35 1"/>')
    hb = track_width / 2
    for i in range(max_agents):
        col = _COLORS[i % len(_COLORS)]
        p.append(f'<body name="a{i}" pos="{_PARK + 3 * i} {_PARK} {radius}">'
                 f'<joint name="x{i}" type="slide" axis="1 0 0"/><joint name="y{i}" type="slide" axis="0 1 0"/>')
        if diffdrive:
            p.append(f'<joint name="yaw{i}" type="hinge" axis="0 0 1"/>')
        p.append(f'<geom name="g{i}" type="sphere" size="{radius}" mass="1" rgba="0 0 0 0" contype="0" conaffinity="0"/>')
        if diffdrive:
            p.append(f'<geom type="cylinder" size="{radius * 0.95} 0.02" pos="0 0 -{radius * 0.4}" rgba="{col} 1" contype="0" conaffinity="0" mass="0"/>')
            for sgn in (1, -1):  # wheels
                p.append(f'<geom type="cylinder" size="{radius * 0.4} 0.008" pos="0 {sgn * hb * 1.15} -{radius * 0.45}" '
                         'euler="90 0 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" mass="0"/>')
            p.append(f'<geom type="box" size="{radius * 0.35} {radius * 0.12} 0.012" pos="{radius * 0.6} 0 -{radius * 0.15}" '
                     'rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>')
        else:
            p.append(f'<geom type="cylinder" size="{radius * 0.95} 0.02" pos="0 0 -{radius * 0.4}" rgba="{col} 1" contype="0" conaffinity="0" mass="0"/>')
        p.append("</body>")
    p.append("</worldbody></mujoco>")
    return "".join(p)


class _MuJoCoMixin:
    """Shared MuJoCo plumbing; concrete classes call :meth:`_setup_mujoco` last in ``__init__``."""

    def _setup_mujoco(self, substeps: int, diffdrive: bool, track_width: float = 0.08) -> None:
        import mujoco

        self._mj = mujoco
        self.substeps = substeps
        self._dd = diffdrive
        xml = build_mjcf(self.max_agents, self.obstacles, self.obs_r, self.radius, self.bound,
                         self.dt / substeps, diffdrive, track_width)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        nm = lambda kind, n: mujoco.mj_name2id(self.model, kind, n)  # noqa: E731
        J, G, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
        self._gid = [nm(G, f"g{i}") for i in range(self.max_agents)]
        self._bid = [nm(B, f"a{i}") for i in range(self.max_agents)]
        self._qadr = [int(self.model.jnt_qposadr[nm(J, f"x{i}")]) for i in range(self.max_agents)]
        self._vadr = [int(self.model.jnt_dofadr[nm(J, f"x{i}")]) for i in range(self.max_agents)]
        for i in range(self.max_agents):
            self._on_slot_change(i)
        mujoco.mj_forward(self.model, self.data)

    def _park_xy(self, i: int) -> np.ndarray:
        return np.array([_PARK + 3 * i, _PARK])

    def _on_slot_change(self, slot: int) -> None:
        if not hasattr(self, "model"):
            return
        on = bool(self.active[slot])
        g = self._gid[slot]
        self.model.geom_contype[g] = 1 if on else 0
        self.model.geom_conaffinity[g] = 1 if on else 0
        q, v = self._qadr[slot], self._vadr[slot]
        if on:
            self.data.qpos[q:q + 2] = self.pos[slot] - self._park_xy(slot)
            self.data.qvel[v:v + 2] = self.vel[slot]
            if self._dd:
                self.data.qpos[q + 2] = self.theta[slot]
                self.data.qvel[v + 2] = 0.0
        else:
            self.data.qpos[q:q + (3 if self._dd else 2)] = 0.0
            self.data.qvel[v:v + (3 if self._dd else 2)] = 0.0
        self._mj.mj_forward(self.model, self.data)

    def _read_back(self) -> None:
        for i in np.flatnonzero(self.active):
            q, v = self._qadr[i], self._vadr[i]
            self.pos[i] = self.data.qpos[q:q + 2] + self._park_xy(i)
            self.vel[i] = self.data.qvel[v:v + 2]
            if self._dd:
                self.theta[i] = (self.data.qpos[q + 2] + np.pi) % (2 * np.pi) - np.pi

    def make_renderer(self, width: int = 640, height: int = 640):
        """Native MuJoCo offscreen renderer (needs OpenGL; fine on macOS, EGL/OSMesa on Linux)."""
        return self._mj.Renderer(self.model, height=height, width=width)


class MuJoCoArena(_MuJoCoMixin, PointMassArena):
    """Holonomic point-mass spheres, force-actuated."""

    def __init__(self, scenario: Scenario, max_agents: int = 10, dt: float = 0.05,
                 v_max: float = 0.6, a_max: float = 3.0, substeps: int = 10) -> None:
        PointMassArena.__init__(self, scenario, max_agents, dt, v_max, a_max)
        self._setup_mujoco(substeps, diffdrive=False)

    def _integrate(self, acc: np.ndarray) -> None:
        for i in range(self.max_agents):
            q, v = self._qadr[i], self._vadr[i]
            if self.active[i]:
                self.data.qpos[q:q + 2] = self.pos[i] - self._park_xy(i)
                self.data.qvel[v:v + 2] = self.vel[i]
                self.data.xfrc_applied[self._bid[i], :2] = acc[i]  # mass = 1
            else:
                self.data.xfrc_applied[self._bid[i], :2] = 0.0
        for _ in range(self.substeps):
            self._mj.mj_step(self.model, self.data)
            for i in np.flatnonzero(self.active):
                v = self._vadr[i]
                sp = np.linalg.norm(self.data.qvel[v:v + 2])
                if sp > self.v_max:
                    self.data.qvel[v:v + 2] *= self.v_max / sp
        self._read_back()


class MuJoCoDiffDriveArena(_MuJoCoMixin, DiffDriveArena):
    """Differential-drive robots with MuJoCo contacts (see module docstring)."""

    def __init__(self, scenario: Scenario, max_agents: int = 10, dt: float = 0.05,
                 wheel_max: float = 0.6, track_width: float = 0.08, a_v: float = 1.5,
                 a_w: float = 12.0, substeps: int = 10) -> None:
        DiffDriveArena.__init__(self, scenario, max_agents, dt, wheel_max, track_width, a_v, a_w)
        self._setup_mujoco(substeps, diffdrive=True, track_width=track_width)

    def _integrate(self, _cmd) -> None:
        for _ in range(self.substeps):
            for i in np.flatnonzero(self.active):
                q, v = self._qadr[i], self._vadr[i]
                th = self.data.qpos[q + 2]
                self.data.qvel[v:v + 3] = [self.v[i] * np.cos(th), self.v[i] * np.sin(th), self.w[i]]
            self._mj.mj_step(self.model, self.data)
        self._read_back()
