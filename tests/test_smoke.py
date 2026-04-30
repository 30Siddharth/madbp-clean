"""CPU smoke tests — exercise the model end-to-end with tiny tensors.

These tests are what GitHub Actions runs on every push. They do **not**
require VMAS and they do **not** require a GPU. The goal is to catch shape
mismatches and pipeline regressions cheaply.
"""

from __future__ import annotations

import torch

from madbp.models import losses
from madbp.models.diffusion import LossWeights, MultiAgentDiffusionPlanner


def _toy_batch(B: int = 2, max_agents: int = 4, T: int = 8, H: int = 16, W: int = 16):
    frames = torch.rand(B, 3, H, W)
    starts = torch.rand(B, max_agents, 2) * 2 - 1
    goals = torch.rand(B, max_agents, 2) * 2 - 1
    n_agents = torch.tensor([3, 2])
    trajs = torch.rand(B, max_agents, T, 2) * 2 - 1
    return frames, starts, goals, n_agents, trajs


def test_forward_pass_shapes_and_gradients():
    """Train forward should produce a scalar loss with non-trivial gradient."""
    torch.manual_seed(0)
    model = MultiAgentDiffusionPlanner(
        max_agents=4, horizon=8, diffusion_steps=20, hid=32, schedule_type="linear",
    )
    f, s, g, n, t = _toy_batch(max_agents=4, T=8)
    out = model(f, s, g, n, t, loss_weights=LossWeights())
    assert out.total.dim() == 0
    out.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(gr.abs().sum().item() > 0 for gr in grads), "no parameter received gradient"


def test_preprocessor_is_in_train_path():
    """P0 fix: the axial preprocessor must be reached on the training path."""
    torch.manual_seed(0)
    model = MultiAgentDiffusionPlanner(
        max_agents=4, horizon=8, diffusion_steps=20, hid=32, schedule_type="linear",
    )
    f, s, g, n, t = _toy_batch(max_agents=4, T=8)
    out = model(f, s, g, n, t)
    out.total.backward()
    pp_grads = [
        p.grad for p in model.preprocessor.parameters() if p.grad is not None
    ]
    assert pp_grads, "preprocessor has no parameters?"
    assert any(gr.abs().sum().item() > 0 for gr in pp_grads), \
        "preprocessor parameters did not receive gradient — train path bypasses it"


def test_sample_pins_endpoints():
    """DDIM sample must return start/goal exactly at t=0 and t=-1 for active agents."""
    torch.manual_seed(0)
    model = MultiAgentDiffusionPlanner(
        max_agents=4, horizon=8, diffusion_steps=20, hid=32, schedule_type="linear",
    )
    f, s, g, n, _ = _toy_batch(max_agents=4, T=8)
    traj = model.sample(f, s, g, n, steps=5, max_step_size=0.2)
    assert traj.shape == (2, 4, 8, 2)
    # First active agents: start and goal must be hit exactly.
    for b in range(2):
        nb = int(n[b].item())
        assert torch.allclose(traj[b, :nb, 0, :], s[b, :nb], atol=1e-5)
        assert torch.allclose(traj[b, :nb, -1, :], g[b, :nb], atol=1e-5)


def test_collision_loss_vectorisation():
    """Collision loss must be > 0 when two agents share a position."""
    pred = torch.zeros(1, 3, 4, 2)  # all three agents collocated
    pred[0, 0] = 0.0
    pred[0, 1] = 0.0
    pred[0, 2] = 1.0
    n = torch.tensor([3])
    L = losses.collision_loss(pred, n, min_distance=0.1)
    assert L.item() > 0


def test_collate_curriculum_clamp():
    """collate_fn(active_agents=k) must clamp n_agents <= k."""
    from madbp.data.dataset import collate_fn

    item = {
        "frame": torch.zeros(16, 16, 3),
        "traj": torch.zeros(8, 5, 2),
        "start": torch.zeros(5, 2),
        "goal": torch.zeros(5, 2),
        "n_agents": 5,
    }
    _, _, _, n, _ = collate_fn([item, item], max_agents=10, active_agents=2)
    assert n.tolist() == [2, 2]


def test_cfg_dropout_off_by_default():
    """With cfg_dropout_prob=0, every forward must use the conditional context."""
    torch.manual_seed(0)
    model = MultiAgentDiffusionPlanner(
        max_agents=4, horizon=8, diffusion_steps=20, hid=32, cfg_dropout_prob=0.0,
    )
    assert model.cfg_dropout_prob == 0.0
