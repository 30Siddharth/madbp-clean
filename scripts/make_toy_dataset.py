"""Write a small synthetic HDF5 dataset in the training format (for smoke tests only).

Straight-line start->goal "expert" trajectories with random layouts. Lets you verify
the full train/eval pipeline on a laptop without the real MAPPO dataset. Do NOT use it
for results.

    python scripts/make_toy_dataset.py --out data/toy.h5 --episodes 64 --agents 4 --steps 40
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/toy.h5")
p.add_argument("--episodes", type=int, default=64)
p.add_argument("--agents", type=int, default=4)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--size", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with h5py.File(a.out, "w") as f:
    for g in ("frames", "trajectories", "start_poses", "goal_poses", "obstacles"):
        f.create_group(g)
    for e in range(a.episodes):
        name = f"ep_{e:05d}"
        s = rng.uniform(-0.9, 0.9, (a.agents, 2))
        g = rng.uniform(-0.9, 0.9, (a.agents, 2))
        fr = np.linspace(0, 1, a.steps)[:, None, None]
        traj = s[None] * (1 - fr) + g[None] * fr  # [T, Na, 2]
        obs = rng.uniform(-0.6, 0.6, (3, 2))
        img = (rng.random((1, a.size, a.size, 3)) * 255).astype(np.uint8)
        f["frames"][name] = img
        f["trajectories"][name] = traj.astype(np.float32)
        f["start_poses"][name] = s.astype(np.float32)
        f["goal_poses"][name] = g.astype(np.float32)
        f["obstacles"][name] = obs.astype(np.float32)
print("wrote", a.out)
