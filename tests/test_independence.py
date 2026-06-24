"""Independence / visibility certification tests."""
import pytest

from orchestration import independence

ROUTING = {
    "visibility": {
        "panel": {"mode": "blind", "members_see": "original_prompt_only", "enforce": "hard"},
        "soft": {"mode": "blind", "members_see": "original_prompt_only", "enforce": "soft"},
    }
}


def test_self_test_passes():
    assert independence.self_test() is True


def test_blind_panel_certified():
    p = "original question"
    vis = independence.certify(ROUTING, "panel", p, [p, p, p])
    assert vis["independence_certified"] is True
    assert vis["mode"] == "blind"


def test_contamination_raises_under_hard_enforce():
    p = "original question"
    with pytest.raises(ValueError):
        independence.certify(ROUTING, "panel", p, [p, p + " member A: YES", p])


def test_contamination_does_not_raise_under_soft_enforce():
    p = "original question"
    vis = independence.certify(ROUTING, "soft", p, [p, p + " member A: YES", p])
    assert vis["independence_certified"] is False


def test_undeclared_task_class_returns_none():
    assert independence.certify(ROUTING, "undeclared", "p", ["p"]) is None


def test_assert_blind_true_for_identical_inputs():
    assert independence.assert_blind("q", ["q", "q"]) is True
