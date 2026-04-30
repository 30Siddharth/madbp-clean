# Pipeline

This document mirrors Sec. III of the paper and the diagram below describes the
data flow used in both training and deployment.

## High-level flow

```
                ┌────────────────────────────┐
   env image ──▶│                            │
   start  ────▶ │     ContextEncoder         │──▶  c ∈ R^{hid}
   goal   ────▶ │  (CNN + pose MLP +         │
   n_agents ──▶ │   count emb + attn)        │
                └────────────────────────────┘

           x_t ∈ R^{B×Na×T×2}                       t ∈ {0,…,T_diff-1}
                  │                                       │
                  ▼                                       ▼
       ┌──────────────────────┐                   ┌──────────────┐
       │   AxialPreprocessor  │                   │ SinusoidalPE │
       │  (agent-axis attn +  │                   └──────┬───────┘
       │   time-axis attn +   │                          │
       │   token-wise MLP)    │                          ▼
       └──────────┬───────────┘                  cond = c + emb(t)
                  ▼
       feat ∈ R^{B×Na×T×hid}        ─────────────▶
                                                   ┌────────────────────┐
                                                   │   TrajectoryUNet   │
                                                   │   (per-agent 1-D   │
                                                   │   U-Net w/ FiLM    │
                                                   │   from cond)       │
                                                   └─────────┬──────────┘
                                                             ▼
                                                   ε̂ ∈ R^{B×Na×T×2}
```

The same module graph is used for training and sampling — the **only**
difference is which loss runs after `ε̂` is produced.

## Training step

```
x_0  ──q_sample(t)──▶  x_t ─────▶ ε̂_θ(x_t, c, t)
                                       │
                                       ▼
                       L_total = w_ε L_ε + w_b L_bound + w_t L_temp
                                + w_v L_vel + w_c L_coll + w_g L_goal
```

- ``q_sample`` is the boundary-aware forward diffusion (Sec. III-B).
- ``L_ε`` is **Min-SNR-weighted** MSE on epsilon, masked to active agents.
- All other losses are computed on the *one-shot* recovered ``x̂_0``:
  ``x̂_0 = (x_t - sqrt(1 - α_t) · ε̂) / sqrt(α_t)``.

## Curriculum injection

The trainer rebuilds the dataloader at the start of every epoch with a
``collate_fn`` parameterised by ``n_active = curriculum(epoch)``. The collate
function clamps ``n_agents[b] ← min(n_agents[b], n_active)``. Every
downstream module (axial mask, U-Net loss masking, boundary/collision losses)
already respects this per-batch ``n_agents``, so the curriculum signal flows
cleanly through one channel.

This is the **P0 fix** vs. the arXiv v1 reference, which computed
``n_active`` in the trainer but never injected it into the dataset.

## Moving-window rollout

At deploy time we sample one window of length ``H`` at a time:

```
window_0:  s_0 = start                      g_0 = start + (1/K)(goal - start)
window_1:  s_1 = window_0[-1]               g_1 = s_1 + (1/(K-1))(goal - s_1)
…
window_{K-1}: s_{K-1} = window_{K-2}[-1]    g_{K-1} = goal       (final goal)
```

Each window call uses ``MultiAgentDiffusionPlanner.sample()``, which runs
DDIM with hard endpoint pinning at ``t=0`` and ``t=H-1`` and clips
inter-step displacements to ``max_step_size``. Inactive (padded) agents are
*never* pinned.

## Optional classifier-free guidance

When ``cfg_dropout_prob > 0`` is set at construction:

- Training: with probability ``p`` the conditional context ``c`` is replaced
  by a learnable null token ``c_∅``.
- Sampling: ``ε̂ ← ε̂_∅ + w · (ε̂_c - ε̂_∅)`` where ``w = guidance_scale``.

This is folded into the same model class — there is no separate CFG variant.
