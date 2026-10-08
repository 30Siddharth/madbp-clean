"""Semantic axial pre-processor (paper Sec. III-C.2, Eq. 1 and the text after it).

Couples agents over two axes *before* the per-agent U-Net:

  * **Agent axis** - multi-head self-attention over agents at each timestep.
    Padded / inactive agents are masked out through ``key_padding_mask``.
  * **Time axis** - multi-head self-attention over timesteps, per agent.

followed by a residual per-token MLP. Input embeddings are
``X_pos = Linear(X) + PE_agent + PE_time`` (Eq. 1).

Because attention is permutation-equivariant in the agent axis, the same weights
apply for any number of active agents. The *learned* agent-slot embedding is the
one component that is not: a slot that never held an agent in training keeps its
random initial value. Train with slot shuffling (``shuffle_slots=True`` in the
trainer, on by default) so every slot embedding is trained, or set
``agent_pos="none"`` for a fully permutation-equivariant block.
"""

from __future__ import annotations

import torch
from torch import nn


class AxialPreprocessor(nn.Module):
    """Inter-agent + temporal axial attention block."""

    def __init__(
        self,
        state_dim: int = 2,
        feat_dim: int = 128,
        num_heads: int = 4,
        max_agents: int = 10,
        T: int = 20,
        dropout: float = 0.1,
        agent_pos: str = "learned",
    ) -> None:
        super().__init__()
        if agent_pos not in {"learned", "none"}:
            raise ValueError(f"agent_pos must be 'learned' or 'none', got {agent_pos!r}")
        self.max_agents = max_agents
        self.T = T
        self.agent_pos = agent_pos

        self.input_proj = nn.Linear(state_dim, feat_dim)
        if agent_pos == "learned":
            self.agent_pos_emb = nn.Parameter(torch.randn(max_agents, feat_dim) * 0.02)
        else:
            self.register_parameter("agent_pos_emb", None)
        self.time_pos_emb = nn.Parameter(torch.randn(T, feat_dim) * 0.02)

        self.agent_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(self, x: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        """Run the axial block.

        Parameters
        ----------
        x:
            ``[B, N, T, state_dim]`` (typically the noisy trajectory ``X^k``).
        agent_mask:
            Boolean ``[B, N]`` where ``True`` marks a **padded** (inactive) slot.
        """
        B, N, T, _ = x.shape
        if N > self.max_agents:
            raise ValueError(f"got {N} agent slots but the block was built for {self.max_agents}")
        if T > self.T:
            raise ValueError(f"got horizon {T} but the block was built for {self.T}")
        x = self.input_proj(x)

        if self.agent_pos_emb is not None:
            x = x + self.agent_pos_emb[:N].unsqueeze(0).unsqueeze(2)
        x = x + self.time_pos_emb[:T].unsqueeze(0).unsqueeze(1)

        # Agent axis: attend across agents at every timestep.
        key_mask = agent_mask.clone()
        empty = key_mask.all(dim=1)  # samples with no active agent would give NaN
        if empty.any():
            key_mask[empty, 0] = False
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)
        mask_flat = key_mask.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
        x_agent, _ = self.agent_attn(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        x_agent = x_agent.reshape(B, T, N, -1).permute(0, 2, 1, 3)
        x = self.norm1(x + x_agent)

        # Time axis: attend across timesteps within each agent.
        x_flat = x.reshape(B * N, T, -1)
        x_time, _ = self.time_attn(x_flat, x_flat, x_flat)
        x_time = x_time.reshape(B, N, T, -1)
        x = self.norm2(x + x_time)

        return x + self.mlp(x)
