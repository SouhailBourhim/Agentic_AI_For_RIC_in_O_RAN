"""Structural checks of the RL objective schema (feasibility checks come in M03)."""

import pytest
from pydantic import ValidationError

from verifier.objective import DEFAULT_OBJECTIVE, RLObjective, SliceValues


def test_valid_objective_is_accepted() -> None:
    obj = RLObjective(
        weights=SliceValues(embb=0.2, urllc=0.5, be=0.3),
        min_thr_mbps=SliceValues(urllc=10.0),
        max_latency_ms=10.0,
        valid_from=0.0,
        valid_until=600.0,
    )
    assert obj.min_thr_mbps.urllc == 10.0


def test_default_objective_is_valid() -> None:
    assert sum(DEFAULT_OBJECTIVE.weights.as_list()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"weights": SliceValues(embb=0.5, urllc=0.5, be=0.5)}, id="weights-sum-1.5"),
        pytest.param({"weights": SliceValues(embb=0.2, urllc=0.2, be=0.2)}, id="weights-sum-0.6"),
        pytest.param({"weights": {"embb": 1.2, "urllc": -0.2, "be": 0.0}}, id="negative-weight"),
        pytest.param(
            {"weights": SliceValues(embb=1.0), "min_thr_mbps": {"urllc": -1.0}}, id="negative-target"
        ),
        pytest.param({"weights": SliceValues(embb=1.0), "max_latency_ms": 0.0}, id="zero-latency-bound"),
        pytest.param(
            {"weights": SliceValues(embb=1.0), "valid_from": 100.0, "valid_until": 100.0},
            id="empty-window",
        ),
        pytest.param({"weights": {"embb": 1.0, "mmtc": 0.0}}, id="unknown-slice"),
    ],
)
def test_invalid_objective_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        RLObjective(**kwargs)


def test_objective_is_immutable() -> None:
    """A verified objective must not be changed afterwards."""
    with pytest.raises(ValidationError):
        DEFAULT_OBJECTIVE.max_latency_ms = 5.0  # type: ignore[misc]
