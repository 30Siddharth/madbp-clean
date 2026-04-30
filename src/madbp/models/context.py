"""Context encoder.

Fuses three conditioning signals into a single context vector ``c in R^hid``
that conditions the trajectory U-Net at every layer (via FiLM):

    1. Environment state — an RGB top-down image of the scene.
    2. Active-agent endpoints — the (start, goal) pair of every agent.
    3. Active agent count — an embedding of ``n_agents``.

These three tokens are fused with a single multi-head self-attention layer,
mean-pooled, and layer-normed. The same context is broadcast across every
agent and timestep when the U-Net runs (see ``unet.TrajectoryUNet``).
"""

from __future__ import annotations

import torch
from torch import nn


class ContextEncoder(nn.Module):
    """Multi-modal context encoder producing ``[B, hid]`` tokens."""

    def __init__(
        self,
        img_ch: int = 3,
        pose_dim: int = 2,
        hid: int = 128,
        max_agents: int = 10,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.pose_dim = pose_dim

        self.img_enc = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hid),
        )

        # Start + goal for every slot in the padded [max_agents, 2] tensors.
        self.pose_enc = nn.Sequential(
            nn.Linear(2 * max_agents * pose_dim, hid * 2),
            nn.ReLU(),
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
        )

        self.count_emb = nn.Embedding(max_agents + 1, hid)
        self.fusion = nn.MultiheadAttention(hid, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hid)

    def forward(
        self,
        img: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        n_agents: torch.Tensor | int,
    ) -> torch.Tensor:
        B = img.size(0)
        f_img = self.img_enc(img)

        poses = torch.cat([start, goal], dim=-1).flatten(1)
        f_pose = self.pose_enc(poses)

        if isinstance(n_agents, torch.Tensor):
            # Use the per-batch maximum so a single embedding represents the
            # padded batch. (All batch elements share the same max_agents
            # padding; the per-element n is communicated via the agent_mask
            # downstream.)
            count_idx = n_agents.max().clamp(max=self.count_emb.num_embeddings - 1)
            n_emb = self.count_emb(count_idx).unsqueeze(0).expand(B, -1)
        else:
            count_idx = torch.tensor(min(n_agents, self.count_emb.num_embeddings - 1), device=f_img.device)
            n_emb = self.count_emb(count_idx).unsqueeze(0).expand(B, -1)

        tokens = torch.stack([f_img, f_pose, n_emb], dim=1)
        fused, _ = self.fusion(tokens, tokens, tokens)
        return self.norm(fused.mean(1))
