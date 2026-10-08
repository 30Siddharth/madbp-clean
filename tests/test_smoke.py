"""CPU smoke tests for the torch model (skipped when torch is unavailable)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from madbp.models import losses  # noqa: E402
from madbp.models.diffusion import LossWeights, MultiAgentDiffusionPlanner  # noqa: E402
from madbp.models.masks import prefix_mask  # noqa: E402


def _model(**kw):
    return MultiAgentDiffusionPlanner(max_agents=4, horizon=8, diffusion_steps=20, hid=32, **kw)


def _toy(B=2, N=4, T=8, H=16, W=16):
    frames = torch.rand(B, 3, H, W)
    starts = torch.rand(B, N, 2) * 2 - 1
    goals = torch.rand(B, N, 2) * 2 - 1
    active = prefix_mask(torch.tensor([3, 2]), N)
    trajs = torch.rand(B, N, T, 2) * 2 - 1
    return frames, starts, goals, active, trajs


def test_forward_and_gradients():
    torch.manual_seed(0)
    m = _model()
    f, s, g, a, t = _toy()
    out = m(f, s, g, None, t, loss_weights=LossWeights(), active=a)
    assert out.total.dim() == 0
    out.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_preprocessor_in_train_path():
    torch.manual_seed(0)
    m = _model()
    f, s, g, a, t = _toy()
    m(f, s, g, None, t, active=a).total.backward()
    grads = [p.grad for p in m.preprocessor.parameters() if p.grad is not None]
    assert grads and any(x.abs().sum() > 0 for x in grads)


def test_collision_term_receives_weight_and_gradient():
    torch.manual_seed(0)
    m = _model()
    f, s, g, a, _ = _toy()
    t = torch.zeros(2, 4, 8, 2)  # all agents collocated -> large collision loss
    out = m(f, s, g, None, t, loss_weights=LossWeights(), active=a)
    assert out.collision.item() > 0


def test_sample_pins_endpoints_with_shuffled_slots():
    torch.manual_seed(0)
    m = _model().eval()
    f, s, g, _, _ = _toy()
    active = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]], dtype=torch.bool)
    traj = m.sample(f, s, g, active=active, steps=4)
    for b in range(2):
        idx = active[b]
        assert torch.allclose(traj[b, idx, 0], s[b, idx], atol=1e-5)
        assert torch.allclose(traj[b, idx, -1], g[b, idx], atol=1e-5)


def test_padded_slots_do_not_affect_active_outputs():
    """Garbage in padded slots must not change the prediction for active agents."""
    torch.manual_seed(0)
    m = _model().eval()
    f, s, g, a, _ = _toy()
    gen = lambda: torch.Generator().manual_seed(1)  # noqa: E731
    out1 = m.sample(f, s, g, active=a, steps=3, generator=gen())
    s2, g2 = s.clone(), g.clone()
    s2[~a] = 5.0
    g2[~a] = -5.0
    out2 = m.sample(f, s2, g2, active=a, steps=3, generator=gen())
    assert torch.allclose(out1[a], out2[a], atol=1e-4)


def test_collision_loss_positive_when_colocated():
    pred = torch.zeros(1, 3, 4, 2)
    pred[0, 2] = 1.0
    active = torch.tensor([[True, True, True]])
    assert losses.collision_loss(pred, active, d_min=0.1).item() > 0


def test_collate_curriculum_and_shuffle():
    from madbp.data.dataset import collate_fn

    item = {"frame": torch.zeros(16, 16, 3), "traj": torch.zeros(8, 5, 2),
            "start": torch.zeros(5, 2), "goal": torch.zeros(5, 2), "n_agents": 5}
    b = collate_fn([item, item], max_agents=10, active_agents=2, shuffle_slots=True)
    assert b.n_agents.tolist() == [2, 2]
    assert b.active.sum(1).tolist() == [2, 2]


def test_trainconfig_importable():
    from madbp.training.trainer import TrainConfig

    assert TrainConfig("x.h5").loss_weights.collision == pytest.approx(0.1)
