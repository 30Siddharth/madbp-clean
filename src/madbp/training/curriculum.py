"""Curriculum schedules for active-agent count.

The schedule maps ``epoch -> n_active`` and is consumed by the trainer to
construct the per-epoch ``collate_fn``. This is the *only* place the
curriculum lives — there is no second copy in the dataset.
"""

from __future__ import annotations

import math


def sigmoid_curriculum(epoch: int, total_epochs: int, max_agents: int) -> int:
    """Smooth sigmoid ramp from 1 to ``max_agents`` over the run.

    At ``epoch=0`` returns ~1; at ``epoch=total_epochs`` returns
    ``max_agents``.
    """
    if total_epochs <= 0:
        return max_agents
    progress = epoch / total_epochs
    s = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
    return max(1, int(round(max_agents * s)))


def constant_curriculum(epoch: int, total_epochs: int, max_agents: int) -> int:
    """No curriculum — always train at full agent count."""
    return max_agents
