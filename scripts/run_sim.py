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

from madbp.sim.arena import PointMassArena
from madbp.sim.executor import ExecConfig, run_episode
from madbp.sim.planners import MADBPPlanner, StraightLinePlanner
from madbp.sim.scenario import Event, make_scenario


def make_arena(kind: str, sc, max_agents: int):
    if kind == "mujoco":
        from madbp.sim.mujoco_arena import MuJoCoArena

        return MuJoCoArena(sc, max_agents=max_agents)
    return PointMassArena(sc, max_agents=max_agents)


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
                r = run_episode(make_arena(a.arena, sc, a.max_agents), planner, cfg, seed)
                row = r.as_row()
                row.update(arena=a.arena, dynamic=a.dynamic, init=a.init)
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
