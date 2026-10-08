"""Approximate scenario rasteriser (numpy only).

Produces the ``O_t`` image the context encoder expects *if* no VMAS render is
available. It is NOT pixel-identical to the VMAS renders the diffusion model was
trained on; for the headline results use :mod:`madbp.sim.vmas_layout`, which
renders with VMAS itself, so there is no image domain gap.
"""

from __future__ import annotations

import numpy as np


def render_scenario(starts, goals, obstacles, obstacle_radius, agent_radius=0.05,
                    bound=1.0, size=64) -> np.ndarray:
    img = np.ones((size, size, 3), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    wx = (xs + 0.5) / size * 2 * bound - bound
    wy = bound - (ys + 0.5) / size * 2 * bound

    def disc(c, r, color):
        m = (wx - c[0]) ** 2 + (wy - c[1]) ** 2 <= r**2
        img[m] = color

    for c in obstacles:
        disc(c, obstacle_radius, (0.1, 0.1, 0.1))
    for g in goals:
        disc(g, agent_radius * 0.8, (0.2, 0.8, 0.2))
    for s in starts:
        disc(s, agent_radius, (0.2, 0.3, 0.9))
    return img
