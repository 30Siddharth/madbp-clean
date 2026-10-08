"""Curriculum schedules for the number of active agents.

Maps ``epoch -> n_active``; the trainer calls it once per epoch and builds the
epoch's ``collate_fn`` from the result, so the curriculum lives in exactly one
place.

The paper (Sec. III-D, "Training Procedure") says the agent count "increases
linearly every ``1000 * n_train_max`` epochs", which cannot be literal for the
2000-epoch runs in the configs. :func:`linear_stage_curriculum` implements the
intent: one agent is added every ``stage_epochs`` epochs until ``n_train_max`` is
reached (default ``stage_epochs = total_epochs // n_train_max``). State the value
you actually use in the paper.
"""

from __future__ import annotations

import math


def linear_stage_curriculum(
    epoch: int,
    total_epochs: int,
    n_train_max: int,
    stage_epochs: int | None = None,
) -> int:
    """Staircase ramp ``1 -> n_train_max``; one more agent every ``stage_epochs``."""
    if n_train_max <= 1:
        return max(1, n_train_max)
    if stage_epochs is None or stage_epochs <= 0:
        stage_epochs = max(total_epochs // n_train_max, 1)
    return int(min(n_train_max, 1 + epoch // stage_epochs))


def sigmoid_curriculum(epoch: int, total_epochs: int, max_agents: int) -> int:
    """Smooth sigmoid ramp from 1 to ``max_agents`` (legacy schedule of the original code)."""
    if total_epochs <= 0:
        return max_agents
    progress = epoch / total_epochs
    s = 1.0 / (1.0 + math.exp(-10.0 * (progress - 0.5)))
    return max(1, int(round(max_agents * s)))


def constant_curriculum(epoch: int, total_epochs: int, max_agents: int) -> int:
    """No curriculum - always the full training team."""
    return max_agents
