# Mathematical reference

This document is the math contract for the MA-DBP code. Every equation here
is implemented somewhere in `src/madbp` and the relevant file is cited inline.

## Notation

| Symbol | Meaning |
|---|---|
| \(B\) | Batch size |
| \(N\) | `max_agents` (architecture-fixed) |
| \(N_a^{(b)}\) | Active agent count in batch element \(b\); \(N_a^{(b)} \le N\) |
| \(H\) | Horizon length of one moving window (`horizon`) |
| \(T_d\) | Number of diffusion steps (`diffusion_steps`) |
| \(\mathbf{x}_0 \in \mathbb{R}^{B\times N\times H\times 2}\) | Clean trajectory tensor |
| \(\mathbf{x}_t\) | Trajectory after \(t\) diffusion steps |
| \(\mathbf{c} \in \mathbb{R}^{B\times d}\) | Conditioning vector from `ContextEncoder` |
| \(\mathbf{m} \in \{0,1\}^{B\times N}\) | Active-agent mask (1 = active) |

## 1. Forward diffusion

Standard Gaussian forward process with a **bounded cumulative \(\bar\alpha\)**:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\bigl(\mathbf{x}_t;\;
\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\;
(1-\bar\alpha_t)\mathbf{I}\bigr).
$$

We support two schedules:

- **Cosine** (Nichol & Dhariwal, 2021):
  \(\bar\alpha_t = \cos^2\!\bigl((t/T_d + s)/(1+s) \cdot \tfrac{\pi}{2}\bigr)\) with \(s = 0.008\).
- **Linear**: \(\beta_t \in [10^{-5}, 2\!\cdot\!10^{-3}]\), \(\bar\alpha_t = \prod_{k\le t}(1-\beta_k)\).

After the schedule is built, \(\bar\alpha_t\) is **clamped** to
\([\,1/(1 + r^2),\; 1\,]\) where \(r\) = `max_noise_ratio` (default 3). This
guarantees a minimum signal level at the noisiest step so that the recovered
\(\hat{\mathbf{x}}_0\) still respects the start/goal boundary conditions.

→ See `src/madbp/models/scheduler.py::BoundaryAwareNoiseScheduler`.

## 2. Reverse parameterisation

We predict noise (`ε`-parameterisation):

$$
\hat\varepsilon_\theta(\mathbf{x}_t, \mathbf{c}, t) = \mathrm{UNet}\bigl(
\mathrm{Axial}(\mathbf{x}_t, \mathbf{m}),\;\; \mathbf{c} + \mathrm{PE}(t)
\bigr),
$$

and reconstruct the clean trajectory in closed form:

$$
\hat{\mathbf{x}}_0 \;=\; \frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon_\theta}{\sqrt{\bar\alpha_t}}.
$$

→ See `MultiAgentDiffusionPlanner.forward` in `src/madbp/models/diffusion.py`.

## 3. Context construction

For batch element \(b\):

$$
\mathbf{c}_b \;=\; \mathrm{LN}\!\bigl(\mathrm{Mean}\,\mathrm{MHA}(
\mathbf{f}_{\text{img}}^{(b)},\; \mathbf{f}_{\text{pose}}^{(b)},\; \mathbf{e}_{N_a^{(b)}})\bigr).
$$

with \(\mathbf{f}_{\text{img}}\) from a CNN over the env image,
\(\mathbf{f}_{\text{pose}}\) from an MLP over the (start, goal) pairs of all
\(N\) padded slots, and \(\mathbf{e}_{N_a}\) a learned embedding of the active
count. The diffusion-time embedding is added on top:
\(\mathbf{c}_b + \mathrm{PE}(t)\).

→ See `src/madbp/models/context.py`.

## 4. Axial pre-processor

Couples agents on the **agent axis** and the **time axis** with two
separate multi-head attentions (parameter-tied across the other axis):

$$
\begin{aligned}
\mathbf{h}^{(1)} &= \mathrm{LN}\!\bigl(\mathbf{x}_t W + \mathrm{MHA}_{\text{agent}}\,\bigr), \quad \text{(per-time mask } \mathbf{m}\text{)}\\
\mathbf{h}^{(2)} &= \mathrm{LN}\!\bigl(\mathbf{h}^{(1)} + \mathrm{MHA}_{\text{time}}\,\bigr), \\
\mathbf{h}^{(3)} &= \mathbf{h}^{(2)} + \mathrm{MLP}\!\bigl(\mathbf{h}^{(2)}\bigr).
\end{aligned}
$$

Inactive agents are masked via the standard `key_padding_mask` convention
(``True`` = ignore). Because the attention is permutation-equivariant in
the agent axis, the same weights work for any \(N_a \le N\).

→ See `src/madbp/models/preprocessor.py::AxialPreprocessor`.

## 5. Per-agent U-Net

The U-Net runs on each agent's per-timestep feature stream as a 1-D
sequence. FiLM conditioning is applied at every residual block:

$$
\mathbf{h}\leftarrow \mathbf{h}\cdot(1+\boldsymbol\gamma) + \boldsymbol\beta, \qquad
[\boldsymbol\gamma; \boldsymbol\beta] = W_{\text{film}}(\mathbf{c} + \mathrm{PE}(t)).
$$

Down-/up-sampling halves/doubles the time axis with stride-2 1-D
convolutions; the bottleneck applies a self-attention along the time axis.

→ See `src/madbp/models/unet.py::TrajectoryUNet`.

## 6. Total training loss

$$
\mathcal{L}_{\text{total}} \;=\;
w_\varepsilon\,\mathcal{L}_\varepsilon
+ w_b\,\mathcal{L}_{\text{bound}}
+ w_t\,\mathcal{L}_{\text{temp}}
+ w_v\,\mathcal{L}_{\text{vel}}
+ w_c\,\mathcal{L}_{\text{coll}}
+ w_g\,\mathcal{L}_{\text{goal}}.
$$

### 6.1 Min-SNR-weighted denoising

$$
\mathcal{L}_\varepsilon \;=\;
\frac{1}{|\mathcal{A}|}\sum_{(b,i,t,k)\in\mathcal{A}}
\bigl(w_t^{\text{snr}}\bigr)^2 \bigl(\hat\varepsilon_\theta - \varepsilon\bigr)^2_{(b,i,t,k)},
$$

with \(w_t^{\text{snr}} = \min(\mathrm{SNR}(t), \gamma)/\mathrm{SNR}(t)\),
\(\gamma = 5\), and \(\mathcal{A}\) the index set of *active* agents.

### 6.2 Boundary

$$
\mathcal{L}_{\text{bound}} \;=\;
\mathbb{E}\!\bigl[\|\hat{\mathbf{x}}_0[\cdot,0] - \mathbf{s}\|_2^2 +
\|\hat{\mathbf{x}}_0[\cdot,H-1] - \mathbf{g}\|_2^2 +
0.1\,\|\hat{\mathbf{x}}_0 - \mathbf{x}_0\|_2^2\bigr].
$$

### 6.3 Temporal consistency

$$
\mathcal{L}_{\text{temp}} \;=\; \mathbb{E}\!\bigl[\|\Delta^2 \hat{\mathbf{x}}_0\|_2\bigr] +
0.1\,\mathbb{E}\!\bigl[\|\Delta^3 \hat{\mathbf{x}}_0\|_2\bigr].
$$

### 6.4 Velocity / acceleration constraint

$$
\mathcal{L}_{\text{vel}} \;=\; \mathbb{E}\!\bigl[(\|\dot{\hat{\mathbf{x}}}_0\| - v_{\max})_+\bigr] +
0.1\,\mathbb{E}\!\bigl[\|\ddot{\hat{\mathbf{x}}}_0\|\bigr].
$$

### 6.5 Inter-agent collision

$$
\mathcal{L}_{\text{coll}} \;=\; \frac{1}{|\mathcal{P}|}\sum_{(b,t,i\ne j)\in\mathcal{P}}
\bigl(d_{\min} - \|\hat{\mathbf{x}}_0^{(b,i,t)} - \hat{\mathbf{x}}_0^{(b,j,t)}\|_2\bigr)_+,
$$

over the active-agent pair set \(\mathcal{P}\).

### 6.6 Goal

$$
\mathcal{L}_{\text{goal}} \;=\; \mathbb{E}\!\bigl[\|\hat{\mathbf{x}}_0[\cdot,H-1] - \mathbf{g}\|_2^2\bigr].
$$

→ All implementations in `src/madbp/models/losses.py`.

## 7. DDIM sampling with hard endpoint constraints

We use deterministic DDIM (η = 0):

$$
\mathbf{x}_{t-1} \;=\; \sqrt{\bar\alpha_{t-1}}\,\hat{\mathbf{x}}_0 + \sqrt{1-\bar\alpha_{t-1}}\,\hat\varepsilon_\theta.
$$

After every step we (a) **clip** inter-step displacements
\(\|\mathbf{x}^{i,k} - \mathbf{x}^{i,k-1}\|\) to `max_step_size` and
(b) **re-pin** the start (\(k=0\)) and goal (\(k=H-1\)) for active agents.

Pinning is enforced with `torch.where(constraint_mask, constraint_values, x)`
so it is fully vectorised.

→ See `MultiAgentDiffusionPlanner.sample` in `src/madbp/models/diffusion.py`.

## 8. Classifier-free guidance (optional)

With ``cfg_dropout_prob = p`` set at construction:

- Training: \(\mathbf{c}_b \leftarrow \mathbf{c}_\varnothing\) with probability \(p\), where
  \(\mathbf{c}_\varnothing\) is a learnable null token.
- Sampling:
  \(\hat\varepsilon \leftarrow \hat\varepsilon_\varnothing + w\,(\hat\varepsilon_c - \hat\varepsilon_\varnothing)\)
  for a guidance scale \(w \ge 1\). \(w=1\) recovers the conditional model.

→ Same file: `MultiAgentDiffusionPlanner.{_build_context, sample}`.

## 9. Curriculum

The active-agent count follows a sigmoid ramp:

$$
N_a(\text{epoch}) \;=\; \mathrm{round}\!\Bigl(N_{\max}\cdot \sigma\bigl(10\,(\tfrac{\text{epoch}}{\text{total}} - 0.5)\bigr)\Bigr).
$$

The schedule lives in `src/madbp/training/curriculum.py` and is consumed
exactly once, at the start of each epoch, to build the `collate_fn`.
