# Changes vs. arXiv v1 reference code

This file documents every behavioural difference between this clean
implementation and the original `30Siddharth/madbp` v2_2 scripts. Use it to
audit whether a result reproduces.

## P0 — silent correctness fixes

| # | Issue in v2_2 | Fix here |
|---|---------------|----------|
| 1 | Train forward bypassed the axial pre-processor (`MADP_diffusion_v2_2.py:242` used `self.input_proj(x_noisy)`), but sampling at line 332 used `self.preprocessor(...)`. Two different architectures were trained vs. evaluated. | `forward()` now calls `self.preprocessor(x_noisy, agent_mask)` exactly like `sample()`. Verified by `tests/test_smoke.py::test_preprocessor_is_in_train_path`. |
| 2 | `compute_collision_loss` was defined twice (lines 149 and 193); only the second won. | Single canonical implementation in `src/madbp/models/losses.py::collision_loss`, fully vectorised over batch and time. |
| 3 | `total_loss` was a single hardcoded line ignoring three `if/elif` branches above it. | One explicit `LossWeights` dataclass with named coefficients; the formula is the last line of `forward()` and matches `docs/math.md` §6. |
| 4 | `enhanced_curriculum_schedule(epoch, ...)` was computed in the trainer but never fed into the dataloader, so the dataset always returned full-team episodes regardless of curriculum stage. | Curriculum is the *only* signal that controls the active-agent count, and it flows through one channel: `Trainer.fit()` calls `sigmoid_curriculum(epoch)`, builds a `make_collate(active_agents=...)`, and rebuilds the loader for that epoch. Verified by `tests/test_smoke.py::test_collate_curriculum_clamp`. |
| 5 | Boundary / temporal / velocity / collision losses each contained Python `for b in range(B)` loops, which were both slow and made masking error-prone. | All five losses are now fully vectorised in `src/madbp/models/losses.py` and share a single `_active_mask` helper. |

## Structural

| Was | Now |
|---|---|
| ~25 sibling `MADP_*_v2_*.py` files at repo root | One package `src/madbp/` with `models/`, `data/`, `training/`, `evaluation/`, `sampling/`, `utils/` |
| `MADP_diffusion_v2_2.py` + `MADP_diffusion_CFG_v2_2.py` (separate classes, code duplicated) | One `MultiAgentDiffusionPlanner` with `cfg_dropout_prob`. CFG-off (`= 0.0`) is bit-identical to non-CFG training. |
| Hardcoded paths and hyperparameters in scripts | YAML configs in `configs/` consumed by the `madbp-train` console script |
| Eight `execution_speed_comparison_results_*` directories with inconsistent step budgets | One canonical `madbp-eval` CLI with a single `EpisodeMetrics` struct |
| No `requirements.txt`, no `pyproject.toml` | `pyproject.toml` with pinned `torch`, `vmas` extras, `dev` extras |
| No CI, no tests | GitHub Actions `ci.yml` runs ruff + 6 CPU smoke tests |
| No README | `README.md` + `docs/{pipeline,math,changes}.md` |

## Numerics that are intentionally unchanged

These were sometimes flagged as "non-standard" but are kept because the
paper's results depend on them:

* Normalisation by `(x - mean) / (3 * std)` and clipping to `[-1, 1]` —
  unchanged from v2_2 dataset.
* Min-SNR loss weighting with `γ = 5` — unchanged.
* `BoundaryAwareNoiseScheduler` clamp on `alphas_cum` — unchanged. The clamp
  is what enables the start/goal pinning to remain feasible at every
  diffusion step.
* DDIM with hard endpoint pinning — unchanged.

## Things deliberately *not* ported (yet)

* `Dynamic_agents_simulator.py` (the pub/sub agent-add/remove demo) — needs
  porting once Experiment 2 (dynamic agent count) is on the schedule.
* `MAPPO`, `CLF-QP`, `MMD`, `projected-diffusion MAPF` evaluation runners —
  shipped as `NotImplementedError` stubs in
  `src/madbp/evaluation/runners/stubs.py` so the CLI surface is complete
  but no baseline silently passes.
* `Create_test_data.py` (HDF5 dataset generator) — should be ported with
  pinned VMAS version. Tracked in the repo TODO.
