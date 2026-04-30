"""HDF5-backed multi-agent trajectory dataset.

Each episode in the HDF5 file lives under three top-level groups:

    /frames/<ep>          -> [F, H, W, 3]  uint8 or float (we use frame 0)
    /trajectories/<ep>    -> [T_full, Na, >=2]  float
    /start_poses/<ep>     -> [Na, >=2]  float
    /goal_poses/<ep>      -> [Na, >=2]  float

Only the first two state dimensions (XY) are used. Coordinates are
normalised by ``(x - mean) / (3 * std)`` and clipped to ``[-1, 1]``.

The dataset returns *full* trajectories of length ``T_full``; the trainer
slices them into moving-window horizons so that the same dataset can serve
training and the moving-window rollout.

Curriculum
----------
The trainer may at any time clamp ``n_agents <= n_active`` by passing
``active_agents=`` to ``collate_fn``. The collate function then pads the
``[B, max_agents, ...]`` tensors with zeros for inactive slots and writes
``n_agents[b] = min(Ni, n_active)`` so every downstream module masks
correctly. This is the **curriculum-injection fix** (P0).
"""

from __future__ import annotations

from collections.abc import Callable

import h5py
import torch
from torch.utils.data import Dataset


class NormalizedTrajectoryDataset(Dataset):
    """HDF5 trajectory dataset with on-the-fly XY normalisation."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        max_agents: int = 10,
        stats_episodes: int = 100,
    ) -> None:
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, "r")
        eps = sorted(self.h5["frames"].keys())
        cut = int(len(eps) * train_ratio)
        self.episodes = eps[:cut] if split == "train" else eps[cut:]
        if not self.episodes:
            raise RuntimeError(f"No episodes for split={split!r} in {h5_path}")
        self.max_agents = max_agents
        self._compute_xy_stats(stats_episodes)

    def _compute_xy_stats(self, stats_episodes: int) -> None:
        all_xy = []
        for ep in self.episodes[: min(stats_episodes, len(self.episodes))]:
            traj = torch.tensor(self.h5["trajectories"][ep][:])
            all_xy.append(traj[..., :2].reshape(-1, 2))
        all_xy = torch.cat(all_xy, dim=0)
        self.xy_mean = all_xy.mean(dim=0)
        self.xy_std = all_xy.std(dim=0).clamp(min=0.1)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        frame = torch.tensor(self.h5["frames"][ep][0], dtype=torch.float32)
        traj = torch.tensor(self.h5["trajectories"][ep][:], dtype=torch.float32)
        start = torch.tensor(self.h5["start_poses"][ep][:], dtype=torch.float32)
        goal = torch.tensor(self.h5["goal_poses"][ep][:], dtype=torch.float32)

        n_agents = traj.shape[1]
        traj_xy = (traj[..., :2] - self.xy_mean) / (3 * self.xy_std)
        start_xy = (start[..., :2] - self.xy_mean) / (3 * self.xy_std)
        goal_xy = (goal[..., :2] - self.xy_mean) / (3 * self.xy_std)
        traj_xy = traj_xy.clamp(-1, 1)
        start_xy = start_xy.clamp(-1, 1)
        goal_xy = goal_xy.clamp(-1, 1)
        return {
            "frame": frame,
            "traj": traj_xy,    # [T_full, Na, 2]
            "start": start_xy,  # [Na, 2]
            "goal": goal_xy,    # [Na, 2]
            "n_agents": n_agents,
        }


def collate_fn(
    batch: list[dict],
    max_agents: int = 10,
    active_agents: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to ``max_agents`` and (optionally) clamp active count for curriculum.

    Returns
    -------
    frames, starts, goals, n_agents, trajs
        With shapes ``[B, 3, H, W]``, ``[B, max_agents, 2]``,
        ``[B, max_agents, 2]``, ``[B]`` long, ``[B, max_agents, T_full, 2]``.
    """
    B = len(batch)
    T_full = batch[0]["traj"].shape[0]

    # Channels-first frames; the dataset stores HxWx3 by VMAS convention.
    frames = torch.stack([b["frame"] for b in batch])
    if frames.dim() == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2).contiguous()

    starts = torch.zeros(B, max_agents, 2)
    goals = torch.zeros(B, max_agents, 2)
    trajs = torch.zeros(B, max_agents, T_full, 2)
    n_agents = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        Ni = int(b["n_agents"])
        if active_agents is not None:
            Ni = min(Ni, int(active_agents))
        n_agents[i] = Ni
        starts[i, :Ni] = b["start"][:Ni, :2]
        goals[i, :Ni] = b["goal"][:Ni, :2]
        trajs[i, :Ni] = b["traj"][:, :Ni, :2].permute(1, 0, 2)
    return frames, starts, goals, n_agents, trajs


def make_collate(max_agents: int, active_agents: int | None) -> Callable:
    """Return a picklable closure of ``collate_fn`` for DataLoader."""

    def _f(batch):
        return collate_fn(batch, max_agents=max_agents, active_agents=active_agents)

    return _f
