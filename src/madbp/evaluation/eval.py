"""Canonical CLI evaluator.

Usage::

    madbp-eval --method madp --scenario navigation_v2 --n-agents 4 \\
               --seed 0 --n-episodes 50 --checkpoint path/to.pth \\
               --out results/madp_navigation_v2_n4.csv

Each call writes one CSV with one row per episode. Aggregation across
methods/seeds happens in downstream notebooks (``scripts/aggregate.py``).

Per-method runners live in ``madbp.evaluation.runners`` and all return an
``EpisodeMetrics``. The MADP runner is implemented; MAPPO / CLF-QP / MMD /
projected-diffusion runners are stubbed and clearly raise ``NotImplementedError``
so partial baselines never silently pass.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from madbp.evaluation.metrics import EpisodeMetrics
from madbp.evaluation.runners import get_runner
from madbp.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a planning method")
    parser.add_argument("--method", required=True,
                        choices=["madp", "mappo", "clf_qp", "mmd", "projected_diffusion"])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--collision-radius", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    runner = get_runner(args.method)(
        scenario=args.scenario,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        collision_radius=args.collision_radius,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[EpisodeMetrics] = []
    for ep in range(args.n_episodes):
        m = runner.run_episode(seed=args.seed * 10_000 + ep)
        rows.append(m)
        print(
            f"[ep {ep:03d}] success={m.success} coll={m.n_collisions} "
            f"makespan={m.makespan} dist={m.final_goal_distance:.3f}"
        )

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].as_row().keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.as_row())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
