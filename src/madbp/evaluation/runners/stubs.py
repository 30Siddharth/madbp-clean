"""Baseline runners not yet ported to the clean codebase.

Porting checklist (see also inline class docstrings below):

  MAPPORunner
  -----------
  - Train or locate a pretrained MAPPO checkpoint (.pth).
  - Checkpoint format should match MADPRunner::

        torch.save({"model": policy.state_dict(), "cfg": {...}}, path)

  - Implement a per-step rollout loop using the VMAS env (mirror MADPRunner).
  - Use ``get_device()`` from ``madbp.utils`` for device selection.
  - Return ``EpisodeMetrics`` from ``run_episode(seed)``.

  CLFQPRunner
  -----------
  - Port the CLF-QP controller class from the legacy codebase.
  - No checkpoint needed; instantiate the controller once in ``__init__``.
  - At each step call controller.act(positions, velocities, goals) → forces.

  MMDRunner  (new variable-agent baseline)
  -----------------------------------------
  - Parameter-sharing MAPPO trained with random agent masking.
  - Train with: ``scripts/train_mmd.py --max-agents N --mask-prob 0.3``
  - Same checkpoint interface as MAPPORunner.

  ProjectedDiffusionRunner
  ------------------------
  - Requires a projected-diffusion checkpoint.
  - See the original projected-diffusion paper for the rollout protocol.

All runners must accept the same keyword arguments as ``MADPRunner.__init__``
(scenario, n_agents, max_steps, collision_radius, goal_radius, checkpoint,
device) so the ``get_runner()`` factory in ``__init__.py`` can instantiate
any runner uniformly.
"""

from __future__ import annotations

from madbp.evaluation.metrics import EpisodeMetrics


class _NotPortedRunner:
    """Sentinel base that always raises NotImplementedError with helpful guidance."""

    name = "stub"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            f"The {self.name!r} runner has not been ported yet. "
            "See the module docstring in "
            "src/madbp/evaluation/runners/stubs.py for the porting checklist "
            "and the expected runner interface."
        )

    def run_episode(self, seed: int) -> EpisodeMetrics:  # pragma: no cover
        raise NotImplementedError


class MAPPORunner(_NotPortedRunner):
    """Wrap a pretrained MAPPO policy for per-step rollout in VMAS.

    Expected checkpoint format (same as MADPRunner)::

        torch.save({"model": policy.state_dict(), "cfg": {...}}, path)

    The policy network should output a continuous action (force) vector
    given the concatenated [pos, vel, goal_rel] observation for each agent.
    """

    name = "mappo"


class CLFQPRunner(_NotPortedRunner):
    """Wrap the CLF-QP decentralised controller from the legacy codebase.

    No checkpoint is required; the controller is deterministic.
    At each step it solves a small QP::

        min  ||u||^2
        s.t. dV/dx * f(x, u) <= -alpha * V(x)   (CLF constraint)
             ||u|| <= max_force                  (input bound)
    """

    name = "clf_qp"


class MMDRunner(_NotPortedRunner):
    """Parameter-sharing MAPPO with random agent masking (variable-agent baseline).

    This is the recommended new baseline for the ``train-small / deploy-large``
    ablation (Table 2 in the paper). Train with::

        python scripts/train_mmd.py --max-agents 10 --mask-prob 0.3

    The mask-prob argument randomly zeros out agents during training so the
    policy learns to handle variable team sizes at inference time.
    """

    name = "mmd"


class ProjectedDiffusionRunner(_NotPortedRunner):
    """Single-agent diffusion policy with projected multi-agent feasibility.

    Requires a pretrained projected-diffusion checkpoint.
    """

    name = "projected_diffusion"
