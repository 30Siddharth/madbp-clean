"""Device selection with automatic MPS fallback for Apple Silicon.

Usage::

    from madbp.utils import get_device
    device = get_device("cuda")   # returns MPS on Apple Silicon if CUDA unavailable
"""

from __future__ import annotations

import warnings

import torch


def get_device(requested: str = "cuda") -> torch.device:
    """Return the best available device, falling back gracefully.

    Resolution order:
      - ``'cpu'``  → always CPU, no fallback.
      - ``'mps'``  → MPS if available, raises ``RuntimeError`` otherwise.
      - ``'cuda'`` → CUDA if available, else MPS (Apple Silicon), else CPU.
      - anything else is passed through to ``torch.device()`` directly.

    Args:
        requested: Device string, e.g. ``"cuda"``, ``"mps"``, ``"cpu"``.

    Returns:
        A :class:`torch.device` instance.
    """
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "MPS requested but not available on this system. "
            "Ensure you are running macOS 12.3+ with an Apple Silicon chip "
            "and PyTorch >= 2.0."
        )

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            warnings.warn(
                "CUDA not available; falling back to MPS (Apple Silicon). "
                "Pass --device mps explicitly to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("mps")
        return torch.device("cpu")

    # Pass-through for anything else (e.g. "cuda:1", "cpu")
    return torch.device(requested)
