"""Canonical CLI evaluator for MA-DBP and all baseline methods.

Usage::

    madbp-eval --method madp --scenario navigation_v2 --n-agents 4 \\
               --seed 0 --n-episodes 50 --checkpoint path/to.pth \\
               --out results/madp_navigation_v2_n4.csv

    # On Apple Silicon (no CUDA) -- MPS is used automatically:
    madbp-eval --method madp --scenario navigation_v2 --n-agents 4 \\
               --seed 0 --n-episodes 50 --checkpoint path/to.pth \\
               --device mps --out results/madp_navigation_v2_n4.csv

Each call writes one CSV with one row per episode. Aggregation across
methods/seeds is done in ``scripts/aggregate.py``.

Per-episode seeds are derived from a ``numpy.random.SeedSequence`` so any
single episode can be reproduced independently by a reviewer:

    import numpy as np
    ss = np.random.SeedSequence(GLOBAL_SEED)
    ep_seed = int(ss.spawn(N_EPISODES)[EP_IDX].generate_state(1)[0])
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from madbp.evaluation.metrics import EpisodeMetrics
from madbp.evaluation.runners import get_runner
from madbp.utils import get_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a planning method")
    parser.add_argument(
        "--method", required=True,
        choices=["madp", "mappo", "clf_qp", "mmd", "projected_diffusion"],
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50).",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--collision-radius", type=float, default=0.1,
        help="Inter-agent distance threshold for collision counting.",
    )
    parser.add_argument(
        "--goal-radius", type=float, default=0.15,
        help="Distance threshold for declaring an agent has reached its goal. "
             "Should be >= --collision-radius to avoid contradictions.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device: cuda | mps | cpu. "
             "Automatically falls back MPS then CPU when cuda is unavailable.",
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = get_device(args.device)
    seed_everything(args.seed)

    # Reproducible per-episode seeds derived from a SeedSequence so each
    # episode can be independently reproduced without re-running all prior ones.
    ss = np.random.SeedSequence(args.seed)
    ep_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(args.n_episodes)]

    runner = get_runner(args.method)(
        scenario=args.scenario,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        collision_radius=args.collision_radius,
        goal_radius=args.goal_radius,
        checkpoint=args.checkpoint,
        device=str(device),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[EpisodeMetrics] = []
    for ep, ep_seed in enumerate(ep_seeds):
        m = runner.run_episode(seed=ep_seed)
        rows.append(m)
        print(
            f"[ep {ep:03d}] success={int(m.success)} "
            f"coll={m.n_collisions:3d} "
            f"makespan={m.makespan:4d} "
            f"dist={m.final_goal_distance:.3f} "
            f"SPL={m.spl:.3f}"
        )

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].as_row().keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r.as_row())

    # Aggregate summary (printed and visible in CI logs)
    sr  = float(np.mean([r.success       for r in rows]))
    cr  = float(np.mean([r.collision_rate for r in rows]))
    spl = float(np.mean([r.spl           for r in rows]))
    ttg_vals = [r.time_to_goal_mean for r in rows if not np.isnan(r.time_to_goal_mean)]
    ttg = float(np.mean(ttg_vals)) if ttg_vals else float("nan")
    print(f"\nResults ({len(rows)} episodes, device={device}):")
    print(f"  success_rate   = {sr:.3f}")
    print(f"  collision_rate = {cr:.4f}")
    print(f"  SPL            = {spl:.3f}")
    print(f"  time_to_goal   = {ttg:.1f} steps (successful episodes only)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
