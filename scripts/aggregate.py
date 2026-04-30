"""Aggregate per-episode CSVs from `madbp-eval` into a summary table.

Usage::

    python scripts/aggregate.py results/exp1 -o results/exp1_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing per-episode CSVs")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    rows: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for path in Path(args.results_dir).glob("*.csv"):
        with path.open() as f:
            for r in csv.DictReader(f):
                key = (r["method"], r["scenario"], int(r["n_agents"]))
                rows[key].append(r)

    fields = [
        "method", "scenario", "n_agents", "n_episodes",
        "success_rate", "success_rate_ci95",
        "collision_rate_mean", "path_length_mean",
        "makespan_mean", "plan_runtime_s_mean", "spl_mean",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (method, scenario, n), eps in sorted(rows.items()):
            succ = [int(r["success"] == "True") for r in eps]
            sr = sum(succ) / len(succ)
            sd = statistics.stdev(succ) if len(succ) > 1 else 0.0
            ci = 1.96 * sd / max(len(succ) ** 0.5, 1)
            w.writerow({
                "method": method,
                "scenario": scenario,
                "n_agents": n,
                "n_episodes": len(eps),
                "success_rate": round(sr, 4),
                "success_rate_ci95": round(ci, 4),
                "collision_rate_mean": round(statistics.mean(float(r["collision_rate"]) for r in eps), 4),
                "path_length_mean": round(statistics.mean(float(r["path_length_mean"]) for r in eps), 4),
                "makespan_mean": round(statistics.mean(float(r["makespan"]) for r in eps), 2),
                "plan_runtime_s_mean": round(statistics.mean(float(r["plan_runtime_s"]) for r in eps), 4),
                "spl_mean": round(statistics.mean(float(r["spl"]) for r in eps), 4),
            })
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
