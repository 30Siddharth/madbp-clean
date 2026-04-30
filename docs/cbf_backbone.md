# Distributed CBF Backbone — Follow-up Paper Roadmap

> **Status:** Design document for the next paper, not implemented in this repo.
> **Working title:** *Distributed Diffusion Planning with Predicted-Neighbor CBF Filtering*
> **Target:** Journal (T-RO / RA-L journal track / IJRR)
> **Companion paper:** [arXiv:2604.06598](https://arxiv.org/abs/2604.06598) — MA-DBP (this repo).

---

## 1. Pitch in one paragraph

MA-DBP gives every agent a **shared trajectory prior** trained on small teams. At deployment, each agent runs the same diffusion model **locally**, conditions on its own goal and the observed states of its neighbors, and rolls out the joint plan. We exploit this to build a **fully decentralized safety filter**: each agent uses *its own* diffusion samples as a learned predictor of what its neighbors will do, and projects every reverse-diffusion step onto a local control-barrier-function (CBF) safe set. The reverse process itself stays in the safe set, no central coordinator is needed, and the CBF inherits the train-small/deploy-large generalization of the prior. Comms requirement at deploy time is the same as standard MAS: low-latency neighbor states.

---

## 2. What's new vs. the literature

| Prior work | What they do | What's missing |
|---|---|---|
| Tan & Dimarogonas '22 (KTH) [LCSS] | Distributed CBF-QP with adaptive multipliers | No learned prior; nominal controller assumed |
| SafeDiffuser (Xiao '23), SMD (ICML '25) | CBF / projection inside diffusion reverse process | **Centralized** — single QP over the joint state |
| PACS (arXiv:2511.06385) | Path-consistent reachability filter on DPs | Single-robot, not multi-agent |
| Zhu et al. '21 RA-L | Learned neighbor predictor + decentralized MPC | No CBF, no diffusion |
| Liu et al. '24 (arXiv:2407.00741) | Diffusion + decentralized CBF for offline MARL | CBF is a separate post-hoc module, not in the reverse process |
| MADP (arXiv:2509.17244) | Decentralized diffusion policies for coverage | No safety filter, no formal guarantees |

**Our gap:** decentralized CBF projection **inside the reverse diffusion process**, where the diffusion prior itself is the neighbor predictor. No prior work combines these three.

---

## 3. Math

Notation follows `docs/math.md`. Per-agent state is $x_i^{(\tau)} \in \mathbb{R}^d$ at horizon step $\tau \in \{0, \dots, H-1\}$. Joint state $x^{(\tau)} = (x_1^{(\tau)}, \dots, x_N^{(\tau)})$.

### 3.1 Per-agent CBFs

For agent $i$, define local barrier functions over **predicted** neighbor positions $\hat{x}_j$ ($j \in \mathcal{N}_i$):

**Inter-agent collision:**
\[
h^{\text{col}}_{ij}(x_i^{(\tau)}, \hat{x}_j^{(\tau)}) = \|x_i^{(\tau)} - \hat{x}_j^{(\tau)}\|^2 - (2r)^2, \quad j \in \mathcal{N}_i
\]

**Static obstacle $o$:**
\[
h^{\text{obs}}_{io}(x_i^{(\tau)}) = \|x_i^{(\tau)} - p_o\|^2 - (r + r_o)^2
\]

**Barrier region $B$ (signed-distance to keep-out):**
\[
h^{\text{bar}}_{iB}(x_i^{(\tau)}) = \mathrm{sdf}_B(x_i^{(\tau)}) - r
\]

### 3.2 Discrete-time CBF condition along the planning horizon

For each barrier $h_k$ active for agent $i$, between consecutive horizon steps $\tau, \tau{+}1$:
\[
h_k(x_i^{(\tau+1)}, \hat{x}_{\mathcal{N}_i}^{(\tau+1)}) \;\ge\; (1 - \gamma_k)\, h_k(x_i^{(\tau)}, \hat{x}_{\mathcal{N}_i}^{(\tau)}),
\quad \gamma_k \in (0, 1].
\]
This is what makes it a **backbone** — safety is enforced along the entire planned trajectory, not just at one timestep.

### 3.3 Predicted-neighbor surrogate (the key trick)

Because every agent runs the **same** diffusion model with the **same** scenario context, each agent's local sample $\hat{x}_0$ already contains a prediction of every other agent's trajectory. Agent $i$ uses:

\[
\hat{x}_j^{(\tau)} \;=\; \big[\hat{x}_0^{(i)}\big]_{j,\tau}, \quad j \in \mathcal{N}_i, \;\; \tau = 0, \dots, H-1
\]

That is, agent $i$'s own sampled $\hat{x}_0$ is the predictor it uses for neighbor $j$. **No negotiation, no consensus, no shared multipliers.** Comms is just "broadcast my current state at deploy time."

> **Why this is sound:** If all agents use the same model and the same shared context (scenario/obstacle layout), their priors over joint trajectories are *identical distributions*. Agent $i$'s sample is a draw from the same posterior as agent $j$'s. The disagreement is only the realized sample noise — which we'll bound in Section 6.1.

### 3.4 Projected reverse step (per agent, per diffusion step)

In the standard MA-DBP sampler:
\[
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\, \hat\epsilon_\theta(x_t, t, c)}{\sqrt{\bar\alpha_t}},
\quad
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\, \hat\epsilon
\]

We replace it with a per-agent local QP:

\[
\tilde{x}_{0,i} \;=\; \arg\min_{x_i \in \mathbb{R}^{H \times d}} \tfrac{1}{2}\big\|x_i - [\hat{x}_0]_i\big\|^2
\]
**subject to**
\[
\nabla_{x_i^{(\tau)}} h_k \cdot (x_i^{(\tau+1)} - x_i^{(\tau)}) \;\ge\; -\gamma_k\, h_k(\hat{x}_0^{(\tau)}) \quad \forall k, \tau
\]
\[
x_i^{(0)} = s_i, \quad x_i^{(H-1)} = g_i.
\]

The constraints are **linearizations of $h_k$ around $\hat{x}_0$** — yielding a QP with $H{\cdot}d$ variables and $|\mathcal{N}_i| H + |\mathcal{O}| H + |B| H$ constraints. Each agent solves its own QP; **no communication** during the projection itself.

Then re-noise with the model's own $\hat\epsilon$:
\[
x_{t-1,i} = \sqrt{\bar\alpha_{t-1}}\, \tilde{x}_{0,i} + \sqrt{1-\bar\alpha_{t-1}}\, \hat\epsilon_i.
\]

Endpoint pinning is already in the QP (equality constraint), so the existing `torch.where` is no longer needed when CBF is on.

### 3.5 Algorithm (per agent $i$)

```
Input: model θ, scheduler {ᾱ_t}, neighbor states, start s_i, goal g_i, scenario c
x ~ N(0, I)                                                       # local sample
for t = T, T-1, ..., 1:
    ε̂ ← ε_θ(x, t, c)                                              # local forward
    x̂_0 ← (x − √(1−ᾱ_t) ε̂) / √ᾱ_t                                # x0 prediction
    Ĥ_neighbors ← extract neighbor slabs from x̂_0                # predicted neighbors
    Build local QP: linearize h_k around (x̂_0_i, Ĥ_neighbors)
    x̃_0_i ← QP.solve()                                           # local projection
    x ← √(ᾱ_{t-1}) x̃_0_i + √(1−ᾱ_{t-1}) ε̂                        # re-noise
return x_0_i
```

No agent-to-agent comms inside the loop. Agents only broadcast initial neighbor state once (and optionally re-broadcast for re-planning).

---

## 4. Theoretical contributions

### 4.1 Forward invariance under linearization (proposition)

> **Proposition (informal).** If at every reverse step the QP in §3.4 is feasible and $h_k$ is twice-differentiable with bounded Hessian on the operating set, then the projected sampler produces $x_0$ that satisfies $h_k(x_0^{(\tau)}) \ge 0$ for all $\tau$, up to a residual $O(\|\hat\epsilon\| \cdot \sqrt{1-\bar\alpha_t})$ that vanishes as $t \to 0$.

The vanishing residual is the key: the linearization error scales with the noise still present at step $t$, and at $t=0$ that noise is zero, so the final $x_0$ satisfies the original (nonlinear) CBF constraints exactly.

### 4.2 Consistency of predicted neighbors (proposition)

> **Proposition (informal).** Let $\hat{x}_0^{(i)}$, $\hat{x}_0^{(j)}$ be samples from agents $i$ and $j$ using model $\theta$, the same context $c$, and independent noise. Then $\mathbb{E}[\hat{x}_0^{(i)}]_{j} = \mathbb{E}[\hat{x}_0^{(j)}]_{j}$ and the disagreement on neighbor $j$'s position has variance bounded by the model's predictive variance.

This justifies the predicted-neighbor surrogate: in expectation, agent $i$'s prediction of agent $j$ matches agent $j$'s prediction of itself.

### 4.3 Inflation against prediction disagreement

To handle the actual realized disagreement, inflate barriers:
\[
\tilde h^{\text{col}}_{ij} = \|x_i - \hat{x}_j\|^2 - (2r + \delta_{\text{pred}})^2
\]
where $\delta_{\text{pred}}$ is a (data-driven) bound on the model's predictive std. We'll calibrate $\delta_{\text{pred}}$ from held-out data.

---

## 5. Implementation plan

```
src/madbp/sampling/
  cbf.py          # DistributedCBFProjector (interface + qpth ref impl)
  cbf_backend.py  # cvxpylayers / OSQP / custom GPU dual-ascent variants
  distributed_sampler.py  # DistributedCBFSampler wrapping the existing model
```

Stub interface:

```python
class DistributedCBFProjector:
    def __init__(
        self,
        agent_radius: float,
        obstacles: list[Obstacle],
        barriers: list[BarrierRegion],
        gamma_col: float = 0.5,
        gamma_obs: float = 0.5,
        gamma_bar: float = 0.5,
        prediction_inflation: float = 0.0,
    ): ...

    def project(
        self,
        x0_hat: Tensor,          # (B, N, H, d) — full joint sample
        agent_idx: int,          # which agent's QP to solve
        active_mask: Tensor,     # (B, N) — which neighbors exist
    ) -> Tensor:                 # (B, H, d) — projected trajectory for agent_idx
        ...

class DistributedCBFSampler:
    def __init__(self, model, scheduler, projector, mode="per_step"): ...

    @torch.no_grad()  # except for QP backward
    def sample(self, context, starts, goals, n_agents): ...
```

The existing `MultiAgentDiffusionPlanner` is **unchanged**. The projector is purely a sampler add-on. Trainer and dataset are unchanged. This is the clean factoring the repo is set up for.

### Backends, ranked by engineering cost

1. **`qpth`** (Amos & Kolter): differentiable batched QP, easiest first impl.
2. **`cvxpylayers`**: more flexible constraint expression, slower.
3. **OSQP** via `osqpth`: faster but requires careful warm-starting.
4. **Custom GPU dual-ascent**: best long-term, only if profiling demands it.

---

## 6. Experiment plan

### 6.1 Synthetic ablations (sim, VMAS)

| Experiment | What we vary | What we measure | Goal |
|---|---|---|---|
| **A. Centralized vs distributed CBF** | one joint QP vs $N$ local QPs | success rate, runtime, comms bytes | Show distributed matches centralized at $\le 5\%$ success drop, but $O(N)$ scalability |
| **B. Filter ablation** | none / soft loss / final-only / per-step | success, collision count, OOD rate | Justify per-step projection |
| **C. Prediction inflation sweep** | $\delta_{\text{pred}} \in [0, 2r]$ | success vs collisions tradeoff | Calibrate the conservatism knob |
| **D. Train-small / deploy-large** | $n_\text{train}{=}4$, $n_\text{deploy} \in \{4,...,16\}$ | success, collision rate | The headline plot — does the CBF preserve the generalization? |
| **E. Comms drop / latency** | drop rate $\in [0, 0.5]$, delay $\in [0, 200ms]$ | success | Robustness story (deferred from primary scope but cheap to add) |

### 6.2 Hardware / realism (stretch goal)

Port to ROS2 + a small Crazyflie or TurtleBot swarm. Two scenarios: open-field rendezvous, narrow-corridor exchange. **Realistic for journal version, optional for first submission.**

### 6.3 Table III draft (vs. MA-DBP main paper Table I)

| Method | Empty (n=4→7) | Obstacles (n=4→7) | Barrier (n=4→7) | Comms |
|---|---|---|---|---|
| MA-DBP (this repo, soft loss) | 0.94 | 0.74 | 0.78 | none |
| MA-DBP + central CBF (final-x0) | ? | ? | ? | full joint state |
| MA-DBP + central CBF (per-step) | ? | ? | ? | full joint state |
| **MA-DBP + dist. CBF (ours)** | **?** | **?** | **?** | neighbor only |
| MAPPO (paper baseline) | 0.95 | 0.94 | 0.92 | full joint state |
| CLF-QP (paper baseline) | 0.95 | 0.61 | 0.36 | full joint state |

The pitch is the **last two columns**: distributed CBF closes the obstacle/barrier gap to MAPPO with strictly less communication.

---

## 7. Risks & honest pitfalls

| Risk | Mitigation |
|---|---|
| Per-step QP is too slow for real-time | Pre-compile QP with `qpth`, batch across diffusion steps, or use guidance variant as fallback |
| Predicted-neighbor surrogate disagrees too much | Inflation bound (§4.3) + ablation on $\delta_{\text{pred}}$ |
| QP infeasibility in dense scenarios | Slack variables with hierarchical priority (cf. Decentralized RCBFs '25) |
| Theory needs extra assumptions | Keep claims modest: "forward invariance under linearization + bounded model variance" |
| Reviewer asks "why not just train a safer prior?" | The whole point: prior is generic, filter is parametric in obstacles/radii — change deployment, no retrain |
| Centralized CBF outperforms distributed | Acceptable — the contribution is the *scalability + comms* story, not absolute performance |

---

## 8. Suggested timeline (after current CoRL submission lands)

| Week | Milestone |
|---|---|
| 1-2 | Implement `DistributedCBFProjector` with `qpth`; integration test with MADBp sampler |
| 3-4 | Experiments A, B, C in VMAS |
| 5-6 | Experiments D, E + theory write-up (Section 4) |
| 7-8 | Hardware sketch (or skip for journal v1) |
| 9-10 | Write paper, target T-RO regular paper |

---

## 9. Files this roadmap would touch (when implemented)

```
src/madbp/sampling/cbf.py                  # NEW
src/madbp/sampling/cbf_backend.py          # NEW
src/madbp/sampling/distributed_sampler.py  # NEW
src/madbp/evaluation/runners/dist_cbf.py   # NEW (eval runner)
configs/madbp_dist_cbf.yaml                # NEW
tests/test_cbf_projection.py               # NEW
docs/math.md                                # extend with §6 CBF
README.md                                   # add follow-up paper link
```

Trainer, model, dataset, losses are **unchanged**. CBF is purely a deployment-time module.

---

## 10. References to track

1. Tan, X. & Dimarogonas, D. V. (2022). *Distributed implementation of CBF-induced QPs for MAS.* L-CSS.
2. Xiao, W. et al. (2023). *SafeDiffuser: Safe planning with diffusion probabilistic models.* arXiv:2306.00148.
3. Liang, J. et al. (2025). *Simultaneous Multi-Robot Motion Planning with Projected Diffusion.* ICML.
4. Schiavi, A. et al. (2025). *PACS: Path-Consistent Safety Filtering for Diffusion Policies.* arXiv:2511.06385.
5. Zhu, H. et al. (2021). *Learning Interaction-Aware Trajectory Predictions for Decentralized Multi-Robot Motion Planning.* RA-L.
6. Liu, J. et al. (2024). *Diffusion Models for Offline MARL with Safety Constraints.* arXiv:2407.00741.
7. Singh, S. et al. (2025). *Train-Small Deploy-Large: MA-DBP.* arXiv:2604.06598.

---

*Draft v0.1 — owner: Siddharth Singh. Open for revision before any code lands on `main`.*
