"""Moving-window training loop for MA-DBP (paper Sec. III-D).

    for epoch in range(epochs):
        n_active = curriculum(epoch)                  # agent count actually injected
        loader   = build_loader(active_agents=n_active, shuffle_slots=True)
        for batch in loader:
            for window in moving_windows(batch.trajs):  # length-H windows
                loss = model(frames, cur_pos, sub_goal, ..., window, weights)
                loss.total.backward(); opt.step()
                cur_pos = expert position at the end of the window

Sub-goals: the paper does not define the sub-goal of a non-final window.
``subgoal_mode="expert"`` (default) uses the expert (MAPPO) position at the end
of the window, i.e. a goal the demonstrator really reached; the last window uses
the true goal. ``"linear"`` reproduces the original straight-line interpolation.
At execution time the closed-loop executor uses a straight-line sub-goal whose
distance is capped by the mean window displacement stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from madbp.data.dataset import NormalizedTrajectoryDataset, make_collate
from madbp.models.diffusion import LossWeights, MultiAgentDiffusionPlanner
from madbp.training.curriculum import (
    constant_curriculum,
    linear_stage_curriculum,
    sigmoid_curriculum,
)

LOSS_KEYS = ("total", "denoising", "boundary", "temporal", "velocity", "collision", "goal")


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
    min_snr_gamma: float | None = None  # off by default (not in the paper)
    epochs: int = 2000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    save_every: int = 200
    device: str = "auto"  # "auto" -> cuda, else mps (Apple silicon), else cpu
    # curriculum: "linear_stage" | "sigmoid" | "none"
    curriculum: str = "linear_stage"
    use_curriculum: bool = True  # kept for backwards compatibility; False == "none"
    n_train_max: int = 4
    stage_epochs: int | None = None
    shuffle_slots: bool = True
    # safety distances in WORLD units (converted to normalised units at runtime)
    d_min_world: float = 0.1
    d_obs_world: float = 0.1
    subgoal_mode: str = "expert"  # "expert" | "linear"
    loss_weights: LossWeights = field(default_factory=LossWeights)

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        lw = cfg.pop("loss_weights", {}) or {}
        cfg["loss_weights"] = LossWeights(**lw)
        known = set(cls.__dataclass_fields__)
        unknown = set(cfg) - known
        if unknown:
            print(f"[TrainConfig] ignoring unknown keys: {sorted(unknown)}")
            cfg = {k: v for k, v in cfg.items() if k in known}
        return cls(**cfg)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    """Moving-window trainer with explicit curriculum injection."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(resolve_device(cfg.device))
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(os.path.join("runs", cfg.run_name))
        except Exception:  # tensorboard optional
            self.writer = None

        self.train_ds = NormalizedTrajectoryDataset(cfg.h5_path, split="train", max_agents=cfg.max_agents)
        self.test_ds = NormalizedTrajectoryDataset(cfg.h5_path, split="test", max_agents=cfg.max_agents)

        self.model = MultiAgentDiffusionPlanner(
            max_agents=cfg.max_agents,
            horizon=cfg.horizon,
            state_dim=2,
            img_ch=3,
            hid=128,
            diffusion_steps=cfg.diffusion_steps,
            schedule_type=cfg.schedule,
            cfg_dropout_prob=cfg.cfg_dropout_prob,
            min_snr_gamma=cfg.min_snr_gamma,
            d_min=self.train_ds.norm_distance(cfg.d_min_world),
            d_obs=self.train_ds.norm_distance(cfg.d_obs_world),
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs, eta_min=1e-6)
        self.global_step = 0
        # running mean of per-window displacement (normalised units), for the executor
        self._disp_sum = 0.0
        self._disp_n = 0

    # --------------------------------------------------------------
    def n_active(self, epoch: int) -> int:
        c = self.cfg
        mode = c.curriculum if c.use_curriculum else "none"
        if mode == "linear_stage":
            return linear_stage_curriculum(epoch, c.epochs, c.n_train_max, c.stage_epochs)
        if mode == "sigmoid":
            return sigmoid_curriculum(epoch, c.epochs, c.n_train_max)
        return constant_curriculum(epoch, c.epochs, c.n_train_max)

    def _train_batch(self, b) -> dict[str, float]:
        H = self.cfg.horizon
        trajs = b.trajs
        _, _, T_full, _ = trajs.shape
        n_windows = max(T_full // H, 1)
        cur_pos = b.starts.clone()
        agg = {k: 0.0 for k in LOSS_KEYS}
        act = b.active

        for k in range(n_windows):
            t0, t1 = k * H, k * H + H
            window = trajs[:, :, t0:t1, :].clone()
            window[:, :, 0, :] = cur_pos
            last = k == n_windows - 1
            if last:
                sub_goal = b.goals
            elif self.cfg.subgoal_mode == "linear":
                p = (k + 1) / n_windows
                sub_goal = cur_pos + p * (b.goals - cur_pos)
            else:
                sub_goal = trajs[:, :, t1 - 1, :]
            # the window's own end point is pinned by the boundary loss: keep it
            # consistent with the sub-goal given to the model.
            window[:, :, -1, :] = sub_goal if last or self.cfg.subgoal_mode == "expert" else window[:, :, -1, :]

            self.opt.zero_grad(set_to_none=True)
            out = self.model(
                b.frames,
                cur_pos,
                sub_goal,
                None,
                window,
                loss_weights=self.cfg.loss_weights,
                active=act,
                obstacles=b.obstacles if b.obstacles.shape[1] > 0 else None,
                obstacle_mask=b.obstacle_mask if b.obstacles.shape[1] > 0 else None,
            )
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()

            for key in LOSS_KEYS:
                v = getattr(out, key)
                agg[key] += float(v.item() if torch.is_tensor(v) else v)
            with torch.no_grad():
                d = (window[:, :, -1] - window[:, :, 0]).norm(dim=-1)[act]
                if d.numel():
                    self._disp_sum += float(d.mean())
                    self._disp_n += 1
            cur_pos = trajs[:, :, t1 - 1, :].detach().clone()

        return {k: v / n_windows for k, v in agg.items()}

    def window_displacement(self) -> float:
        return self._disp_sum / max(self._disp_n, 1)

    def save(self, epoch: int) -> Path:
        cfg = self.cfg
        ckpt = Path(cfg.out_dir) / f"{cfg.run_name}_epoch_{epoch:05d}.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": self.model.model_config(),
                "epoch": epoch,
                "cfg": {k: (asdict(v) if k == "loss_weights" else v) for k, v in cfg.__dict__.items()},
                "xy_mean": self.train_ds.xy_mean,
                "xy_std": self.train_ds.xy_std,
                "window_displacement": self.window_displacement(),
            },
            ckpt,
        )
        return ckpt

    def fit(self) -> None:
        cfg = self.cfg
        for epoch in tqdm(range(cfg.epochs), desc="train"):
            self.model.train()
            n_active = self.n_active(epoch)
            loader = DataLoader(
                self.train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=make_collate(cfg.max_agents, n_active, cfg.shuffle_slots),
            )
            running = {k: 0.0 for k in LOSS_KEYS}
            n_batches = 0
            for batch in loader:
                batch = type(batch)(*(x.to(self.device, non_blocking=True) for x in batch))
                stats = self._train_batch(batch)
                for k in running:
                    running[k] += stats[k]
                n_batches += 1
                self.global_step += 1

            self.sched.step()
            if self.writer is not None:
                for k in running:
                    self.writer.add_scalar(f"loss/{k}", running[k] / max(n_batches, 1), epoch)
                self.writer.add_scalar("curriculum/n_active", n_active, epoch)
                self.writer.add_scalar("optim/lr", self.opt.param_groups[0]["lr"], epoch)

            if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
                self.save(epoch)

        if self.writer is not None:
            self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MA-DBP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    Trainer(TrainConfig.from_yaml(args.config)).fit()


if __name__ == "__main__":
    main()
