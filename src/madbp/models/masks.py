"""Agent-mask helpers.

Everywhere in MA-DBP an *active mask* is a boolean tensor ``[B, N]`` where
``True`` marks a slot that holds a live agent. Active agents need not occupy a
prefix of the slots: during training agents are placed in random slots so every
slot embedding is exercised, and during dynamic deployment agents can leave or
join in any slot.

``prefix_mask`` converts the legacy "first ``n`` slots are active" convention.
"""

from __future__ import annotations

import torch


def prefix_mask(n_agents: torch.Tensor, max_agents: int) -> torch.Tensor:
    """``[B, max_agents]`` bool mask with the first ``n_agents[b]`` slots active."""
    n = n_agents.to(dtype=torch.long)
    arange = torch.arange(max_agents, device=n.device)[None, :]
    return arange < n[:, None]


def resolve_active(
    n_agents: torch.Tensor | None,
    active: torch.Tensor | None,
    max_agents: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a ``[B, max_agents]`` bool active mask from either input form."""
    if active is not None:
        return active.to(device=device, dtype=torch.bool)
    if n_agents is None:
        raise ValueError("Provide either `n_agents` or `active`.")
    return prefix_mask(n_agents.to(device), max_agents)
