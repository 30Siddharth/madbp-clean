"""HDF5-backed multi-agent trajectory dataset.

Each episode in the HDF5 file lives under these top-level groups:

    /frames/<ep>          -> [F, H, W, 3]  uint8 or float (we use frame 0)
    /trajectories/<ep>    -> [T_full, Na, >=2]  float
    /start_poses/<ep>     -> [Na, >=2]  float
    /goal_poses/<ep>      -> [Na, >=2]  float
    /obstacles/<ep>       -> [M, 2]  float        (optional; world coordinates)

Only the first two state dimensions (XY) are used. Coordinates are normalised
by ``(x - mean) / (3 * std)`` and clipped to ``[-1, 1]``. Frames are mapped to
``[0, 1]`` with :func:`madbp.data.frames.to_unit_range`.

Curriculum and slot shuffling
-----------------------------
``collate_fn(active_agents=k)`` keeps at most ``k`` agents per episode. With
``shuffle_slots=True`` the kept agents are scattered over *random* slots of the
``max_agents`` padded tensor instead of occupying the first ones, so every
learned agent-slot embedding in the model is trained even when the training team
is small. The returned ``active`` mask marks the occupied slots.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import h5py
import torch
from torch.utils.data import Dataset

from madbp.data.frames import to_unit_range


class Batch(NamedTuple):
    """One padded training batch."""

    frames: torch.Tensor  # [B, 3, H, W]
    starts: torch.Tensor  # [B, N, 2]
    goals: torch.Tensor  # [B, N, 2]
    n_agents: torch.Tensor  # [B] long
    trajs: torch.Tensor  # [B, N, T_full, 2]
    active: torch.Tensor  # [B, N] bool, True = occupied slot
    obstacles: torch.Tensor  # [B, M, 2]  (M may be 0)
    obstacle_mask: torch.Tensor  # [B, M] bool


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
        self._h5 = None  # opened lazily so the dataset can be pickled to DataLoader workers (macOS spawn)
        eps = sorted(self.h5["frames"].keys())
        cut = int(len(eps) * train_ratio)
        self.episodes = eps[:cut] if split == "train" else eps[cut:]
        if not self.episodes:
            raise RuntimeError(f"No episodes for split={split!r} in {h5_path}")
        self.max_agents = max_agents
        self.has_obstacles = "obstacles" in self.h5
        self._compute_xy_stats(stats_episodes)

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_h5"] = None
        return d

    # -- normalisation -------------------------------------------------
    def _compute_xy_stats(self, stats_episodes: int) -> None:
        all_xy = []
        for ep in self.episodes[: min(stats_episodes, len(self.episodes))]:
            traj = torch.tensor(self.h5["trajectories"][ep][:])
            all_xy.append(traj[..., :2].reshape(-1, 2))
        all_xy = torch.cat(all_xy, dim=0)
        self.xy_mean = all_xy.mean(dim=0)
        self.xy_std = all_xy.std(dim=0).clamp(min=0.1)

    @property
    def norm_scale(self) -> float:
        """Isotropic scale ``3 * mean(std)`` used to convert distances to normalised units."""
        return float(3.0 * self.xy_std.mean())

    def norm_distance(self, d_world: float) -> float:
        """Convert a world-space distance into normalised coordinates."""
        return float(d_world) / self.norm_scale

    def normalize(self, xy: torch.Tensor) -> torch.Tensor:
        return ((xy - self.xy_mean.to(xy)) / (3 * self.xy_std.to(xy))).clamp(-1, 1)

    def denormalize(self, xy: torch.Tensor) -> torch.Tensor:
        return xy * (3 * self.xy_std.to(xy)) + self.xy_mean.to(xy)

    # -- dataset protocol ----------------------------------------------
    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        frame = to_unit_range(torch.tensor(self.h5["frames"][ep][0]))
        traj = torch.tensor(self.h5["trajectories"][ep][:], dtype=torch.float32)
        start = torch.tensor(self.h5["start_poses"][ep][:], dtype=torch.float32)
        goal = torch.tensor(self.h5["goal_poses"][ep][:], dtype=torch.float32)

        n_agents = traj.shape[1]
        item = {
            "frame": frame,
            "traj": self.normalize(traj[..., :2]),  # [T_full, Na, 2]
            "start": self.normalize(start[..., :2]),  # [Na, 2]
            "goal": self.normalize(goal[..., :2]),  # [Na, 2]
            "n_agents": n_agents,
            "obstacles": None,
        }
        if self.has_obstacles and ep in self.h5["obstacles"]:
            obs = torch.tensor(self.h5["obstacles"][ep][:], dtype=torch.float32)
            item["obstacles"] = self.normalize(obs[..., :2])
        return item


def collate_fn(
    batch: list[dict],
    max_agents: int = 10,
    active_agents: int | None = None,
    shuffle_slots: bool = False,
    generator: torch.Generator | None = None,
) -> Batch:
    """Pad to ``max_agents`` and (optionally) clamp the active count for the curriculum."""
    B = len(batch)
    T_full = batch[0]["traj"].shape[0]

    frames = torch.stack([b["frame"] for b in batch])
    if frames.dim() == 4 and frames.shape[-1] == 3:  # HWC -> CHW
        frames = frames.permute(0, 3, 1, 2).contiguous()

    starts = torch.zeros(B, max_agents, 2)
    goals = torch.zeros(B, max_agents, 2)
    trajs = torch.zeros(B, max_agents, T_full, 2)
    n_agents = torch.zeros(B, dtype=torch.long)
    active = torch.zeros(B, max_agents, dtype=torch.bool)

    for i, b in enumerate(batch):
        Ni = min(int(b["n_agents"]), max_agents)
        if active_agents is not None:
            Ni = min(Ni, int(active_agents))
        n_agents[i] = Ni
        if shuffle_slots:
            slots = torch.randperm(max_agents, generator=generator)[:Ni]
        else:
            slots = torch.arange(Ni)
        starts[i, slots] = b["start"][:Ni, :2]
        goals[i, slots] = b["goal"][:Ni, :2]
        trajs[i, slots] = b["traj"][:, :Ni, :2].permute(1, 0, 2)
        active[i, slots] = True

    m_max = max((0 if b.get("obstacles") is None else b["obstacles"].shape[0]) for b in batch)
    obstacles = torch.zeros(B, m_max, 2)
    obstacle_mask = torch.zeros(B, m_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        o = b.get("obstacles")
        if o is not None and o.shape[0] > 0:
            obstacles[i, : o.shape[0]] = o[:, :2]
            obstacle_mask[i, : o.shape[0]] = True

    return Batch(frames, starts, goals, n_agents, trajs, active, obstacles, obstacle_mask)


class CollateFn:
    """Picklable collate callable (a closure would break DataLoader workers under macOS/Windows spawn)."""

    def __init__(self, max_agents: int, active_agents: int | None, shuffle_slots: bool = False) -> None:
        self.max_agents = max_agents
        self.active_agents = active_agents
        self.shuffle_slots = shuffle_slots

    def __call__(self, batch):
        return collate_fn(batch, self.max_agents, self.active_agents, self.shuffle_slots)


def make_collate(max_agents: int, active_agents: int | None, shuffle_slots: bool = False) -> Callable:
    return CollateFn(max_agents, active_agents, shuffle_slots)
