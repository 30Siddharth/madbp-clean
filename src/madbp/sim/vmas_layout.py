"""Layouts and frames taken from VMAS itself (no image domain gap).

The diffusion model's context encoder was trained on VMAS renders, so the
headline closed-loop results should use the same renderer. This module draws
reset layouts from a VMAS scenario and returns a :class:`Scenario` plus a
``frame_fn`` that re-renders the *current* state via VMAS.

Dataset-specific preprocessing (resize, colour inversion, channel order) must
match how the HDF5 frames were produced; pass ``frame_size``/``invert`` that
reproduce your dataset (check one dataset frame next to :func:`render_frame`).

Requires ``vmas`` (and a rendering backend); untested in the authoring
environment. ``scripts/check_vmas_layout.py`` validates the assumptions below.
"""

from __future__ import annotations

import numpy as np

from madbp.sim.scenario import Scenario


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def render_frame(env, frame_size: int = 64, invert: bool = False) -> np.ndarray:
    img = env.render(mode="rgb_array")
    img = np.asarray(img)
    if img.shape[0] != frame_size:
        from PIL import Image

        img = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((frame_size, frame_size)))
    if invert:
        img = 255 - img
    return img


def vmas_scenario(name: str, n_agents: int, seed: int, frame_size: int = 64,
                  invert: bool = False, device: str = "cpu", **kwargs):
    """Return ``(Scenario, frame, env)`` for one reset of a VMAS scenario."""
    from vmas import make_env

    env = make_env(scenario=name, num_envs=1, device=device, continuous_actions=True,
                   n_agents=n_agents, seed=seed, **kwargs)
    env.reset()
    starts = np.stack([_to_np(a.state.pos)[0] for a in env.agents])
    goals = np.stack([_to_np(a.goal.state.pos)[0] for a in env.agents])
    goal_ids = {id(a.goal) for a in env.agents}
    obst = [
        _to_np(l.state.pos)[0] for l in env.world.landmarks
        if id(l) not in goal_ids and getattr(l, "collide", False)
    ]
    obstacles = np.stack(obst) if obst else np.zeros((0, 2))
    r_obs = 0.08
    if env.world.landmarks:
        for l in env.world.landmarks:
            if id(l) not in goal_ids and getattr(l, "collide", False):
                r_obs = float(getattr(l.shape, "radius", r_obs))
                break
    sc = Scenario(name, starts, goals, obstacles, r_obs,
                  bound=float(getattr(env.scenario, "world_semidim", 1.0) or 1.0),
                  agent_radius=float(env.agents[0].shape.radius))
    return sc, render_frame(env, frame_size, invert), env
