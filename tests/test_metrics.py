import pytest

from mtp.metrics import expected_accepted_length, speedup_estimate


def test_expected_accepted_length_uses_conditional_rates():
    # depth 2 is only attempted when depth 1 was accepted: 0.5 + 0.5 * 0.8, not 0.5 + 0.8
    assert expected_accepted_length([0.5, 0.8]) == pytest.approx(0.9)
    assert expected_accepted_length([]) == 0.0
    assert expected_accepted_length([1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert expected_accepted_length([0.0, 1.0]) == 0.0


def test_speedup_estimate_counts_the_bonus_token_and_draft_cost():
    # no drafts accepted, free draft -> 1x; one always-accepted draft costing half a step -> 2/1.5
    assert speedup_estimate([0.0], draft_cost=0.0) == pytest.approx(1.0)
    assert speedup_estimate([1.0], draft_cost=0.5) == pytest.approx(2 / 1.5)
