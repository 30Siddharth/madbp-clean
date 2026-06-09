"""Moving-window training loop for MA-DBP.

Pipeline (matches the paper):

  for epoch in range(epochs):
      n_active = curriculum(epoch)                     # P0: actually injected
      loader   = build_loader(active_agents=n_active)
      for batch in loader:
          for window in moving_windows(batch.traj):    # length-H windows
              loss = model(frames, cur_pos, sub_goal, n_agents, window, weights)
              loss.total.backward(); opt.step()
              cur_pos = window[:, :, -1, :]            # shift window
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from madbp.data import NormalizedTrajectoryDataset
from madbp.data.dataset import make_collate
from madbp.models.diffusion import LossWeights, MultiAgentDiffusionPlanner
from madbp.training.curriculum import sigmoid_curriculum
from madbp.utils import get_device


@dataclass
class TrainConfig:
    h5_path: str
    out_dir: str = "checkpoints"
    run_name: str = "madbp"
    max_agents: int = 10
    horizon: int = 8
    diffusion_steps: int = 150
    schedule: str = "linear"
    cfg_dropout_prob: float = 0.0
    epochs: int = 2000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    log_every: int = 50
    save_every: int = 200
    # "cuda" is the canonical default; get_device() in Trainer.__init__ handles
    # the MPS / CPU fallback automatically so the config file stays portable.
    device: str = "cuda"
    use_curriculum: bool = True
    loss_weights: LossWeights = LossWeights()

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        lw = cfg.pop("loss_weights", {}) or {}
        cfg["loss_weights"] = LossWeights(**lw)
        return cls(**cfg)


class Trainer:
    """Moving-window trainer with explicit curriculum injection."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.device)   # MPS fallback on Apple Silicon
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(os.path.join("runs", cfg.run_name))

        # Dataset is built once; the loader is rebuilt each epoch with the
        # curriculum-controlled collate function (P0 fix).
        self.train_ds = NormalizedTrajectoryDataset(
            cfg.h5_path, split="train", max_agents=cfg.max_agents
        )
        self.test_ds = NormalizedTrajectoryDataset(
            cfg.h5_path, split="test", max_agents=cfg.max_agents
        )

        self.model = MultiAgentDiffusionPlanner(
            max_agents=cfg.max_agents,
            horizon=cfg.horizon,
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=cfg.diffusion_steps,
            schedule_type=cfg.schedule,
            cfg_dropout_prob=cfg.cfg_dropout_prob,
        ).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs, eta_min=1e-6
        )
        self.global_step = 0

    # --------------------------------------------------------------
    # One moving-window pass over a batch
    # --------------------------------------------------------------
    def _train_batch(
        self,
        frames: torch.Tensor,
        starts: torch.Tensor,
        goals: torch.Tensor,
        n_agents: torch.Tensor,
        full_trajs: torch.Tensor,
    ) -> dict[str, float]:
        H = self.cfg.horizon
        B, _, T_full, _ = full_trajs.shape
        n_windows = max(T_full // H, 1)
        cur_pos = starts.clone()
        agg = {"total": 0.0, "denoising": 0.0, "boundary": 0.0,
               "temporal": 0.0, "velocity": 0.0, "collision": 0.0, "goal": 0.0}

        for k in range(n_windows):
            t0, t1 = k * H, k * H + H
            window = full_trajs[:, :, t0:t1, :].clone()
            window[:, :, 0, :] = cur_pos
            if k == n_windows - 1:
                sub_goal = goals
            else:
                p = (k + 1) / n_windows
                sub_goal = cur_pos + p * (goals - cur_pos)

            self.opt.zero_grad(set_to_none=True)
            out = self.model(
                frames, cur_pos, sub_goal, n_agents, window,
                loss_weights=self.cfg.loss_weights,
            )
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

            for k_ in agg:
                agg[k_] += float(getattr(out, k_).item() if torch.is_tensor(getattr(out, k_)) else getattr(out, k_))
            cur_pos = full_trajs[:, :, t1 - 1, :].detach().clone()

        return {k: v / n_windows for k, v in agg.items()}

    # --------------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        for epoch in tqdm(range(cfg.epochs), desc="train"):
            self.model.train()
            n_active = (
                sigmoid_curriculum(epoch, cfg.epochs, cfg.max_agents)
                if cfg.use_curriculum else cfg.max_agents
            )
            loader = DataLoader(
                self.train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),  # pin_memory unsupported on MPS
                collate_fn=make_collate(cfg.max_agents, n_active),
            )

            running = {"total": 0.0, "denoising": 0.0, "boundary": 0.0,
                       "temporal": 0.0, "velocity": 0.0, "collision": 0.0, "goal": 0.0}
            n_batches = 0
            for frames, starts, goals, n_agents, trajs in loader:
                frames = frames.to(self.device, non_blocking=True)
                starts = starts.to(self.device, non_blocking=True)
                goals = goals.to(self.device, non_blocking=True)
                n_agents = n_agents.to(self.device, non_blocking=True)
                trajs = trajs.to(self.device, non_blocking=True)

                stats = self._train_batch(frames, starts, goals, n_agents, trajs)
                for k in running:
                    running[k] += stats[k]
                n_batches += 1
                self.global_step += 1

            self.sched.step()
            for k in running:
                self.writer.add_scalar(f"loss/{k}", running[k] / max(n_batches, 1), epoch)
            self.writer.add_scalar("curriculum/n_active", n_active, epoch)
            self.writer.add_scalar("optim/lr", self.opt.param_groups[0]["lr"], epoch)

            if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
                ckpt = Path(cfg.out_dir) / f"{cfg.run_name}_epoch_{epoch:05d}.pth"
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "cfg": cfg.__dict__}, ckpt)

        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MA-DBP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
