import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import VOCAB, make_model
from mtp import compute_losses, train_step
from mtp.loss import depth_loss


def test_depth_loss_targets_the_token_k_steps_ahead():
    B, T = 2, 9
    targets = torch.randint(0, VOCAB, (B, T))
    for k in (1, 2, 3):
        # a head that already "knows" token t+1+k at slot t should be scored (almost) perfectly
        aligned = F.one_hot(targets[:, k:], VOCAB).float() * 50.0
        assert depth_loss(aligned, targets, k) < 1e-4
        # ... while an off-by-one head (predicting the next token instead) must not be
        off_by_one = F.one_hot(targets[:, : T - k], VOCAB).float() * 50.0
        assert depth_loss(off_by_one, targets, k) > 1.0
    with pytest.raises(ValueError):
        depth_loss(torch.zeros(B, T, VOCAB), targets, T)


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize(
    "kind,n_future,kw",
    [
        ("parallel", 3, {}),
        ("parallel", 2, {"head_arch": "mlp"}),
        ("parallel", 2, {"detach_trunk": True}),
        ("sequential", 3, {}),
        ("sequential", 2, {"share_weights": True}),
        ("sequential", 2, {"detach_trunk": True}),
    ],
)
def test_memory_efficient_train_step_matches_naive_backward(kind, n_future, kw):
    model = make_model(kind, n_future, **kw)
    if kind == "parallel" and kw.get("head_arch") == "mlp":
        # zero-init heads would make the comparison trivial; give them signal
        for p in model.heads.parameters():
            torch.nn.init.normal_(p, std=0.05)
    ref = copy.deepcopy(model)
    idx = torch.randint(0, VOCAB, (2, 16))
    targets = torch.randint(0, VOCAB, (2, 16))

    naive = compute_losses(ref, idx, targets)
    naive.total.backward()
    efficient = train_step(model, idx, targets, memory_efficient=True)

    for a, b in zip(naive.as_floats().values(), efficient.as_floats().values()):
        assert abs(a - b) < 1e-5
    g_ref, g_eff = _grads(ref), _grads(model)
    assert g_ref.keys() == g_eff.keys()
    for name in g_ref:
        assert torch.allclose(g_ref[name], g_eff[name], atol=1e-6, rtol=1e-4), name


def test_detach_trunk_keeps_mtp_gradients_out_of_the_trunk():
    model = make_model("sequential", 2, detach_trunk=True, loss_weight=100.0)
    ntp_only = copy.deepcopy(model)
    idx = torch.randint(0, VOCAB, (2, 16))
    targets = torch.randint(0, VOCAB, (2, 16))
    train_step(model, idx, targets)
    compute_losses(ntp_only, idx, targets).ntp.backward()
    for (name, p), (_, q) in zip(model.trunk.blocks.named_parameters(), ntp_only.trunk.blocks.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6), name
    assert all(p.grad is not None for p in model.heads.parameters())
