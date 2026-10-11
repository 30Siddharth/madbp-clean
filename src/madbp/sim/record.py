"""Episode recorder: per-step snapshots used for videos and post-hoc analysis."""

from __future__ import annotations

import numpy as np


class Recorder:
    """Collects arena snapshots every ``every`` control steps.

    ``mj_renderer`` (optional ``mujoco.Renderer``) additionally grabs native MuJoCo
    pixels from MuJoCo arenas.
    """

    def __init__(self, every: int = 2, mj_renderer=None, mj_camera: str | int = "top") -> None:
        self.every = max(1, every)
        self.frames: list[dict] = []
        self.pixels: list[np.ndarray] = []
        self.mj_renderer = mj_renderer
        self.mj_camera = mj_camera
        self._k = 0

    def capture(self, arena, window: np.ndarray | None = None, wp_k: int = 0) -> None:
        if self._k % self.every == 0:
            self.frames.append(
                dict(
                    t=arena.t,
                    pos=arena.pos.copy(),
                    theta=arena.heading(),
                    active=arena.active.copy(),
                    goals=arena.goals.copy(),
                    window=None if window is None else window.copy(),
                    wp_k=wp_k,
                    aa=arena.agent_agent_collisions,
                    ao=arena.agent_obstacle_collisions,
                )
            )
            if self.mj_renderer is not None and hasattr(arena, "data"):
                self.mj_renderer.update_scene(arena.data, camera=self.mj_camera)
                self.pixels.append(self.mj_renderer.render().copy())
        self._k += 1
