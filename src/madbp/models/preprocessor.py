"""Semantic axial pre-processor.

Couples agents over two axes *before* the per-agent U-Net:

  * **Agent axis** — a multi-head self-attention over agents, applied
    independently at each timestep. Padded agents are masked out via
    ``key_padding_mask``.
  * **Time axis** — a multi-head self-attention over timesteps, applied
    independently per agent. Captures temporal context inside the
    moving-window horizon.

The two attentions are followed by a per-token MLP (residual). This is the
single module the paper credits for ``train-small / deploy-large`` behaviour:
because it is permutation-equivariant in the agent axis and length-flexible
in the time axis, the same weights apply when ``Na`` (active agents) or
``T`` (horizon) change between training and deployment.
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
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.T = T

        self.input_proj = nn.Linear(state_dim, feat_dim)
        self.agent_pos_emb = nn.Parameter(torch.randn(max_agents, feat_dim))
        self.time_pos_emb = nn.Parameter(torch.randn(T, feat_dim))

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
            Trajectory tensor ``[B, Na, T, state_dim]`` (typically the
            *noisy* trajectory at diffusion step ``t``).
        agent_mask:
            Boolean ``[B, Na]`` tensor where ``True`` marks **padded**
            (inactive) agents.
        """
        B, Na, T, _ = x.shape
        x = self.input_proj(x)

        # Slice positional embeddings so deploy-time Na/T may be ≤ training.
        x = x + self.agent_pos_emb[:Na].unsqueeze(0).unsqueeze(2)
        x = x + self.time_pos_emb[:T].unsqueeze(0).unsqueeze(1)

        # Agent-axis: attend across agents at each timestep.
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, Na, -1)
        mask_flat = agent_mask.unsqueeze(1).expand(B, T, Na).reshape(B * T, Na)
        x_agent, _ = self.agent_attn(x_flat, x_flat, x_flat, key_padding_mask=mask_flat)
        x_agent = x_agent.view(B, T, Na, -1).permute(0, 2, 1, 3)
        x = self.norm1(x + x_agent)

        # Time-axis: attend across timesteps within each agent.
        x_flat = x.reshape(B * Na, T, -1)
        x_time, _ = self.time_attn(x_flat, x_flat, x_flat)
        x_time = x_time.view(B, Na, T, -1)
        x = self.norm2(x + x_time)

        x = x + self.mlp(x)
        return x
