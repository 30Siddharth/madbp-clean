"""Context encoder (paper Sec. III-C.2, "Network Architecture and Context Encoding").

Fuses three conditioning signals into a single vector ``c in R^hid``:

    1. a CNN embedding of the scenario image ``O_t``,
    2. an MLP embedding of the start/goal poses,
    3. a learned embedding of the number of active agents.

They are fused with multi-head attention, mean-pooled and layer-normed, then
injected into the U-Net with FiLM.

Two design points matter for the *train-small, deploy-large* claim:

* ``pose_encoder="per_agent"`` (default) applies one MLP to every agent's
  ``[start, goal]`` pair and masked-mean pools over active agents. This is
  permutation-invariant and independent of how many slots are filled. The legacy
  ``"flat"`` option concatenates all ``max_agents`` slots into one vector, so
  slots that were never populated during training are random inputs at
  deployment.
* ``count_encoder="mlp"`` (default) embeds the *number* of active agents with a
  small MLP on ``n / max_agents``. The legacy ``"table"`` option is an
  ``nn.Embedding`` lookup whose rows for counts above the training size never
  receive gradient.
"""

from __future__ import annotations

import torch
from torch import nn


class ContextEncoder(nn.Module):
    """Multi-modal context encoder producing ``[B, hid]`` vectors."""

    def __init__(
        self,
        img_ch: int = 3,
        pose_dim: int = 2,
        hid: int = 128,
        max_agents: int = 10,
        pose_encoder: str = "per_agent",
        count_encoder: str = "mlp",
    ) -> None:
        super().__init__()
        if pose_encoder not in {"per_agent", "flat"}:
            raise ValueError(f"pose_encoder must be 'per_agent' or 'flat', got {pose_encoder!r}")
        if count_encoder not in {"mlp", "table"}:
            raise ValueError(f"count_encoder must be 'mlp' or 'table', got {count_encoder!r}")
        self.max_agents = max_agents
        self.pose_dim = pose_dim
        self.pose_encoder = pose_encoder
        self.count_encoder = count_encoder

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

        in_pose = 2 * pose_dim if pose_encoder == "per_agent" else 2 * max_agents * pose_dim
        self.pose_enc = nn.Sequential(
            nn.Linear(in_pose, hid * 2),
            nn.ReLU(),
            nn.Linear(hid * 2, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
        )

        if count_encoder == "table":
            self.count_emb = nn.Embedding(max_agents + 1, hid)
        else:
            self.count_mlp = nn.Sequential(nn.Linear(1, hid), nn.SiLU(), nn.Linear(hid, hid))

        self.fusion = nn.MultiheadAttention(hid, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hid)

    def _pose_feature(
        self, start: torch.Tensor, goal: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        m = active.unsqueeze(-1).to(start.dtype)  # [B,N,1]
        pairs = torch.cat([start, goal], dim=-1) * m  # [B,N,2*pose_dim], padded slots zeroed
        if self.pose_encoder == "per_agent":
            per = self.pose_enc(pairs)  # [B,N,hid]
            return (per * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.pose_enc(pairs.flatten(1))

    def _count_feature(self, active: torch.Tensor) -> torch.Tensor:
        n = active.sum(dim=1)  # [B] long
        if self.count_encoder == "table":
            return self.count_emb(n.clamp(max=self.max_agents))
        x = (n.to(torch.float32) / float(self.max_agents)).unsqueeze(-1)
        return self.count_mlp(x)

    def forward(
        self,
        img: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """``img`` ``[B,C,H,W]``; ``start``/``goal`` ``[B,N,2]``; ``active`` ``[B,N]`` bool."""
        f_img = self.img_enc(img)
        f_pose = self._pose_feature(start, goal, active)
        f_count = self._count_feature(active)
        tokens = torch.stack([f_img, f_pose, f_count], dim=1)  # [B,3,hid]
        fused, _ = self.fusion(tokens, tokens, tokens)
        return self.norm(fused.mean(1))
