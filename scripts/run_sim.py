"""Closed-loop feasibility experiments (kinematic or MuJoCo arena).

Examples
--------
# torch-free sanity run with the straight-line baseline
python scripts/run_sim.py --planner straight --arena kinematic --kinds open barrier --n-agents 4 --seeds 20

# MA-DBP trained on 4 agents, deployed on 4..10, MuJoCo physics, dynamic team
python scripts/run_sim.py --planner madbp --checkpoint checkpoints/madbp_paper_epoch_01999.pth \
    --arena mujoco --kinds open obstacles barrier --n-agents 4 6 8 10 --seeds 50 --dynamic \
    --out results/madbp_mujoco.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from madbp.sim.arena import DiffDriveArena, PointMassArena
from madbp.sim.record import Recorder
from madbp.sim.video import save_pixels_video, save_video
from madbp.sim.executor import ExecConfig, run_episode
from madbp.sim.planners import MADBPPlanner, StraightLinePlanner
from madbp.sim.scenario import Event, make_scenario


def make_arena(kind: str, sc, max_agents: int, robot: str = "point"):
    if kind == "mujoco":
        from madbp.sim.mujoco_arena import MuJoCoArena, MuJoCoDiffDriveArena

        cls = MuJoCoDiffDriveArena if robot == "diffdrive" else MuJoCoArena
        return cls(sc, max_agents=max_agents)
    cls = DiffDriveArena if robot == "diffdrive" else PointMassArena
    return cls(sc, max_agents=max_agents)


def add_dynamic_events(sc, rng, max_agents):
    n = sc.n_agents
    ev = []
    if n < max_agents:
        s = rng.uniform(-0.9, 0.9, 2)
        g = rng.uniform(-0.9, 0.9, 2)
        ev.append(Event(60, "add", start=tuple(s), goal=tuple(g)))
    if n > 2:
        ev.append(Event(120, "remove", slot=int(rng.integers(0, n))))
    sc.events = ev


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--planner", choices=["madbp", "straight"], default="straight")
    p.add_argument("--checkpoint")
    p.add_argument("--arena", choices=["kinematic", "mujoco"], default="kinematic")
    p.add_argument("--kinds", nargs="+", default=["open", "obstacles", "barrier"])
    p.add_argument("--n-agents", nargs="+", type=int, default=[4])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--max-agents", type=int, default=10)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--init", default="gaussian", choices=["gaussian", "linear"])
    p.add_argument("--no-pinning-cap", action="store_true", help="use the final goal as sub-goal (no window cap)")
    p.add_argument("--robot", choices=["point", "diffdrive"], default="point",
                   help="point: holonomic point mass (matches the VMAS training dynamics); "
                        "diffdrive: non-holonomic differential-drive robot + unicycle tracker")
    p.add_argument("--video-dir", default=None, help="write episode videos here")
    p.add_argument("--n-videos", type=int, default=3, help="videos per (kind, n_agents) cell")
    p.add_argument("--video-format", choices=["gif", "mp4"], default="gif")
    p.add_argument("--video-every", type=int, default=2, help="record every k-th control step")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--video-backend", choices=["matplotlib", "mujoco"], default="matplotlib",
                   help="matplotlib: top-down plot (no OpenGL); mujoco: native renderer (needs --arena mujoco + GL)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/sim.csv")
    a = p.parse_args()

    if a.planner == "madbp":
        planner = MADBPPlanner(a.checkpoint, a.device, steps=a.steps, init=a.init)
        cap = None if a.no_pinning_cap else planner.window_disp_world or None
        horizon = planner.horizon
    else:
        planner, cap, horizon = StraightLinePlanner(8), 0.3, 8
    cfg = ExecConfig(replan_every=horizon - 1, subgoal_cap=cap)

    rows = []
    for kind in a.kinds:
        for n in a.n_agents:
            for seed in range(a.seeds):
                rng = np.random.default_rng(10_000 * seed + n)
                try:
                    sc = make_scenario(kind, n, rng)
                except RuntimeError as e:
                    print("skip", kind, n, seed, e)
                    continue
                if a.dynamic:
                    add_dynamic_events(sc, rng, a.max_agents)
                arena = make_arena(a.arena, sc, a.max_agents, a.robot)
                rec = None
                if a.video_dir and seed < a.n_videos:
                    mjr = None
                    if a.video_backend == "mujoco":
                        if a.arena != "mujoco":
                            raise SystemExit("--video-backend mujoco requires --arena mujoco")
                        mjr = arena.make_renderer()
                    rec = Recorder(every=a.video_every, mj_renderer=mjr)
                r = run_episode(arena, planner, cfg, seed, recorder=rec)
                if rec is not None:
                    Path(a.video_dir).mkdir(parents=True, exist_ok=True)
                    stem = f"{a.planner}_{a.robot}_{a.arena}_{kind}_n{n}_s{seed}"
                    vp = str(Path(a.video_dir) / f"{stem}.{a.video_format}")
                    ttl = f"{a.planner}/{a.robot}/{kind}"
                    if a.video_backend == "mujoco":
                        save_pixels_video(rec.pixels, vp, a.video_fps)
                    else:
                        save_video(rec, sc, vp, a.video_fps, title=ttl)
                    print("video:", vp)
                row = r.as_row()
                row.update(arena=a.arena, robot=a.robot, dynamic=a.dynamic, init=a.init)
                rows.append(row)
            sub = [r for r in rows if r["kind"] == kind and r["n_agents0"] == n]
            if sub:
                sr = np.mean([r["success"] for r in sub])
                print(f"{kind:10s} n={n:2d} success={sr:.2f} "
                      f"coll={np.mean([r['agent_agent_collisions'] for r in sub]):.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
