"""Planners usable by the closed-loop executor.

Interface (all arrays in WORLD coordinates, slot-indexed, ``N = max_agents``)::

    plan(frame, pos[N,2], goals[N,2], active[N]) -> window[N,H,2]

``goals`` here are the *sub-goals* chosen by the executor. The window's first
point is the current position and its last point the sub-goal (pinned), as in
Algorithm 2.
"""

from __future__ import annotations

import numpy as np


class StraightLinePlanner:
    """Baseline: straight line to the sub-goal (no inter-agent / obstacle reasoning)."""

    name = "straight"

    def __init__(self, horizon: int = 8) -> None:
        self.horizon = horizon

    def plan(self, frame, pos, goals, active):
        f = np.linspace(0.0, 1.0, self.horizon)[None, :, None]
        return pos[:, None, :] * (1 - f) + goals[:, None, :] * f


class MADBPPlanner:
    """Wraps a trained :class:`MultiAgentDiffusionPlanner` (torch imported lazily)."""

    name = "madbp"

    def __init__(self, checkpoint: str, device: str = "cpu", steps: int = 50,
                 init: str = "gaussian", max_step_size: float | None = None,
                 guidance_scale: float = 1.0, seed: int = 0) -> None:
        import torch

        from madbp.models.diffusion import MultiAgentDiffusionPlanner

        self.torch = torch
        self.device = torch.device(device)
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        mc = ck.get("model_config")
        if mc is None:
            raise ValueError(
                "checkpoint has no 'model_config'; it was not produced by the paper-aligned "
                "trainer and cannot be loaded (old v2_2 weights are incompatible)."
            )
        self.model = MultiAgentDiffusionPlanner(**mc).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.xy_mean = ck["xy_mean"].float().to(self.device)
        self.xy_std = ck["xy_std"].float().to(self.device)
        self.horizon = self.model.horizon
        self.max_agents = self.model.max_agents
        self.steps, self.init = steps, init
        self.max_step_size, self.guidance_scale = max_step_size, guidance_scale
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        # mean window displacement (world units) seen in training; used to cap sub-goals
        self.window_disp_world = float(ck.get("window_displacement", 0.0)) * float(
            (3 * self.xy_std).mean()
        )
        self.last_plan_time = 0.0

    def _norm(self, xy):
        return ((xy - self.xy_mean) / (3 * self.xy_std)).clamp(-1, 1)

    def _denorm(self, xy):
        return xy * (3 * self.xy_std) + self.xy_mean

    def plan(self, frame, pos, goals, active):
        import time

        torch = self.torch
        from madbp.data.frames import frame_to_tensor

        t0 = time.perf_counter()
        d = self.device
        f = frame_to_tensor(frame, d)
        s = self._norm(torch.as_tensor(pos, dtype=torch.float32, device=d))[None]
        g = self._norm(torch.as_tensor(goals, dtype=torch.float32, device=d))[None]
        a = torch.as_tensor(active, dtype=torch.bool, device=d)[None]
        out = self.model.sample(
            f, s, g, active=a, steps=self.steps, init=self.init,
            max_step_size=self.max_step_size, guidance_scale=self.guidance_scale,
            generator=self.gen,
        )
        w = self._denorm(out[0]).cpu().numpy()
        self.last_plan_time = time.perf_counter() - t0
        return w
