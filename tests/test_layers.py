import torch

from conftest import VOCAB, make_model


def test_kv_cache_chunked_forward_matches_full_forward():
    model = make_model()
    trunk = model.trunk
    idx = torch.randint(0, VOCAB, (2, 12))
    with torch.no_grad():
        full = trunk.logits(trunk(idx))
        caches = trunk.new_caches(2, 12)
        pieces = [trunk.logits(trunk(chunk, caches=caches)) for chunk in idx.split([3, 1, 4, 4], dim=1)]
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)


def test_kv_cache_truncate_then_append_matches_fresh_sequence():
    model = make_model()
    trunk = model.trunk
    a = torch.randint(0, VOCAB, (1, 6))
    b = torch.randint(0, VOCAB, (1, 3))
    with torch.no_grad():
        caches = trunk.new_caches(1, 16)
        trunk(a, caches=caches)
        for c in caches:
            c.truncate(4)
        rewound = trunk.logits(trunk(b, caches=caches))
        assert caches[0].length == 7
        fresh = trunk.logits(trunk(torch.cat((a[:, :4], b), dim=1)))[:, 4:]
    assert torch.allclose(rewound, fresh, atol=1e-5)
