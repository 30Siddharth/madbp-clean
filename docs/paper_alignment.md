# Paper ↔ code alignment (branch `paper-aligned`)

Paper: *Train-Small Deploy-Large* (arXiv 2604.06598). Default behaviour now follows the
paper; extras from the original code remain as **off-by-default** flags.

## What changed in the code
| Area | Before | Now |
|---|---|---|
| Training path | noisy input went through `input_proj`, sampling used the untrained axial pre-processor | pre-processor is in the training path (test: `test_preprocessor_in_train_path`) |
| Loss weights | 0.90/0.04/0.06/**0.0** (collision off) | `[0.85, 0.025, 0.025, 0.10]` (Eq. 8); collision per Eq. 7 incl. obstacle hinge when obstacle coords are in the HDF5 (`/obstacles/<ep>`) |
| U-Net input | `.view` reinterpreted memory (time/feature axes scrambled) | `permute(0,1,3,2).reshape` |
| Agent handling | prefix `n_agents` | explicit `active` mask; slot shuffling in training; padded slots provably inert (test) |
| Context | flat pose MLP, count table indexed by batch max | shared per-agent pose MLP + masked mean pool; count via MLP of `n/N_max` per sample |
| Sampling | mutated `eval()`, step clipping always on | `N(0,I)` start, start/goal pinning per Alg. 2; step clipping, CFG, Min-SNR are flags |
| Frames | 0–255 at inference vs different scale in training | single `to_unit_range` everywhere |
| Curriculum | sigmoid, never injected | `linear_stage` (+1 agent per `stage_epochs`), injected into the loader |
| Trainer | `TrainConfig` mutable default → ValueError on Python ≥ 3.11 | `default_factory` |

## Things the paper should say explicitly
1. **Curriculum**: "every 1000·n_train_max epochs" cannot be literal for 2000-epoch runs. State `stage_epochs`.
2. **Sub-goals** of non-final windows are unspecified. Training uses the expert position at the window end; execution uses a straight-line sub-goal capped at the mean training window displacement (stored in the checkpoint).
3. **Noise schedule**: the clamped schedule has ᾱ_T = 0.7, so Algorithm 2's `N(0,I)` start does not match the training terminal distribution. Report `init="linear"` (warm start) separately if you use it.
4. **U-Net** is a strided-conv U-Net with FiLM residual blocks and bottleneck self-attention.
5. Distances (`d_min`, `d_obs`) are in world units (0.1 = 2 agent radii) and converted to normalised coordinates.
6. Because endpoints are pinned, "reaches goal" is partly by construction. Report collisions, obstacle clearance, smoothness, and the straight-line + tracker baseline and a no-pinning/cap ablation alongside success.

## Simulation (feasibility) stack — `madbp.sim`
* `scenario.py` layouts (open / obstacles / barrier / corridor) and `Event(add|remove)`.
* `arena.py` numpy double-integrator arena (test double); `mujoco_arena.py` MuJoCo physics, same interface.
* `executor.py` receding-horizon closed loop (PD tracking), dynamic teams, metrics.
* `planners.py` `MADBPPlanner` (checkpoint) and the straight-line baseline.
* `vmas_layout.py` VMAS layouts/frames (no image domain gap).
* `scripts/run_sim.py` experiment driver → CSV.

**Not verifiable where this was written** (no torch / mujoco / vmas available): everything torch,
the MuJoCo arena, VMAS layout extraction. The torch-free simulation core and executor are tested.

## Old checkpoints
v2_2 diffusion checkpoints do not load into the paper-faithful model and were trained without the
pre-processor in the loop; retrain. Datasets and MAPPO checkpoints are reusable.
