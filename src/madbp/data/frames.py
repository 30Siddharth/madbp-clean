"""Scenario-image preprocessing shared by training, evaluation and deployment.

The context encoder sees the scenario image ``O_t``. Training and inference must
feed it the *same* pixel scale, otherwise BatchNorm statistics learned on one
scale are applied to the other. Every code path therefore goes through
:func:`to_unit_range`.
"""

from __future__ import annotations

import numpy as np
import torch


def to_unit_range(frame: torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor in ``[0, 1]``.

    Values above 1.5 are interpreted as 8-bit intensities (``0..255``); anything
    already in ``[0, 1]`` is left untouched.
    """
    frame = frame.to(torch.float32)
    if frame.numel() > 0 and float(frame.max()) > 1.5:
        frame = frame / 255.0
    return frame


def frame_to_tensor(frame: np.ndarray | torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
    """``HxWx3`` (or ``3xHxW``) image -> ``[1, 3, H, W]`` float tensor in ``[0, 1]``."""
    t = torch.as_tensor(np.ascontiguousarray(frame) if isinstance(frame, np.ndarray) else frame)
    if t.dim() == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    t = to_unit_range(t).unsqueeze(0)
    return t.to(device)
