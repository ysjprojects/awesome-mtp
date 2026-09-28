import copy

import pytest
import torch

from conftest import VOCAB, make_model, sample_chain, trained_model, transition_matrix
from mtp import SpeculativeDecoder, generate, speculative_generate

CONFIGS = [
    # (kind, n_future, head_arch, share_weights, draft_len)
    ("parallel", 3, "block", False, 3),
    ("parallel", 2, "block", False, 1),
    ("parallel", 2, "mlp", False, 2),
    ("sequential", 2, "block", False, 2),
    ("sequential", 3, "block", False, 2),
    ("sequential", 1, "block", True, 3),  # one shared module drafted recursively beyond its trained depth
]


def _prompts(n: int, length: int = 10):
    g = torch.Generator().manual_seed(123)
    return [sample_chain(transition_matrix(), 1, length - 1, g) for _ in range(n)]


def _with_random_heads(model):
    """Trained trunk, scrambled heads: drafts are mostly wrong, so the reject/rewind path runs."""
    bad = copy.deepcopy(model)
    g = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for p in bad.heads.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return bad.eval()


def test_generate_matches_uncached_argmax_loop():
    model = trained_model("none", 0)
    prompt = _prompts(1)[0]
    out = generate(model, prompt, 15)
    seq = prompt
    with torch.no_grad():
        for _ in range(15):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat((seq, nxt), dim=1)
    assert torch.equal(out, seq)


@pytest.mark.parametrize("kind,n_future,head_arch,share,K", CONFIGS)
def test_speculative_greedy_decoding_is_lossless(kind, n_future, head_arch, share, K):
    model = trained_model(kind, n_future, head_arch, share)
    for candidate, expect_rejections in ((model, False), (_with_random_heads(model), True)):
        accepted = rejected = 0
        for prompt in _prompts(3):
            greedy = generate(candidate, prompt, 40)
            spec, stats = speculative_generate(candidate, prompt, 40, K)
            assert spec.shape == greedy.shape
            assert torch.equal(spec, greedy), (kind, n_future, K)
            assert stats.generated >= 40 and stats.rounds >= 1
            accepted += sum(stats.accepted)
            rejected += stats.rounds * K - sum(stats.accepted)
        # make sure the test actually exercised what it claims to: trained heads get drafts
        # accepted, untrained heads get drafts rejected (and the caches rewound)
        if expect_rejections:
            assert rejected > 0
        else:
            assert accepted > 0


@pytest.mark.parametrize("kind,n_future,head_arch,share,K", CONFIGS)
def test_drafter_state_matches_teacher_forced_forward(kind, n_future, head_arch, share, K):
    """After several rounds (with rewinds), the incremental drafter must produce exactly the
    logits a from-scratch teacher-forced forward over the same tokens produces."""
    model = _with_random_heads(trained_model(kind, n_future, head_arch, share))
    prompt = _prompts(1)[0]
    dec = SpeculativeDecoder(model, prompt, max_new_tokens=60, draft_len=K)
    for _ in range(6):
        dec.round()
    with torch.no_grad():
        drafts, _ = dec.drafter.draft(dec)
        n = dec.n
        if kind == "parallel":
            seq = dec.tokens[:n][None]
        else:
            seq = torch.cat((dec.tokens[:n], dec.t1[None], drafts[0, : K - 1]))[None]
        h0 = model.trunk(seq)
        hs = model.heads.forward_train(h0, seq, model.trunk, n_depths=K)
        for k in range(1, K + 1):
            ref = model.heads.logits(k, hs[k - 1][:, n - 1], model.trunk)
            assert torch.allclose(dec.drafter.last_logits[k - 1], ref, atol=1e-4), k


def test_speculative_sampling_matches_target_distribution():
    """With temperature > 0 the (t1, x2) pair must follow the trunk's own two-step distribution."""
    model = _with_random_heads(trained_model("sequential", 1, share_weights=True))
    prompt = _prompts(1)[0]
    with torch.no_grad():
        p1 = torch.softmax(model(prompt)[0, -1], -1)
        p2 = torch.stack([torch.softmax(model(torch.cat((prompt, torch.tensor([[a]])), 1))[0, -1], -1) for a in range(VOCAB)])
    exact = p1[:, None] * p2  # joint over (x1, x2)

    g = torch.Generator().manual_seed(0)
    counts = torch.zeros(VOCAB, VOCAB)
    N = 3000
    for _ in range(N):
        out, _ = speculative_generate(model, prompt, 2, draft_len=2, temperature=1.0, generator=g)
        counts[out[0, -2], out[0, -1]] += 1
    tv = 0.5 * (counts / N - exact).abs().sum()
    assert tv < 0.1, float(tv)


def test_invalid_speculative_configurations_are_rejected():
    par = make_model("parallel", 2)
    seq = make_model("sequential", 2)
    prompt = torch.randint(0, VOCAB, (1, 5))
    with pytest.raises(ValueError):
        SpeculativeDecoder(par, prompt, 5, draft_len=3)  # parallel heads have fixed offsets
    with pytest.raises(ValueError):
        SpeculativeDecoder(seq, prompt, 5, draft_len=3)  # recursion needs share_weights
    with pytest.raises(ValueError):
        SpeculativeDecoder(seq, torch.randint(0, VOCAB, (2, 5)), 5, draft_len=1)  # batch > 1
    with pytest.raises(ValueError):
        SpeculativeDecoder(make_model(), prompt, 5, draft_len=1)  # no heads at all
