"""merge(): the one objective the RL xApp follows when several intents are active (M03)."""

import pytest

from verifier.merge import ActiveIntent, merge
from verifier.objective import DEFAULT_OBJECTIVE, RLObjective, SliceValues


def intent(
    intent_id: str,
    issued_at: float,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    targets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bound: float | None = None,
    window: tuple[float, float] = (0.0, 1000.0),
) -> ActiveIntent:
    """A short way to write an active intent in the tables below (slices in order embb, urllc, be)."""
    objective = RLObjective(
        weights=SliceValues(embb=weights[0], urllc=weights[1], be=weights[2]),
        min_thr_mbps=SliceValues(embb=targets[0], urllc=targets[1], be=targets[2]),
        max_latency_ms=bound,
        valid_from=window[0],
        valid_until=window[1],
    )
    return ActiveIntent(intent_id=intent_id, issued_at=issued_at, objective=objective)


A = intent("a", 10.0, weights=(0.6, 0.2, 0.2), targets=(20.0, 5.0, 0.0), bound=20.0, window=(10.0, 600.0))
B = intent("b", 50.0, weights=(0.1, 0.8, 0.1), targets=(0.0, 10.0, 8.0), bound=10.0, window=(50.0, 300.0))
B_SAME_TIME = intent("b", 10.0, weights=(0.1, 0.8, 0.1), window=(10.0, 600.0))
LATER = intent("later", 5.0, weights=(0.2, 0.2, 0.6), window=(500.0, 900.0))  # issued early, starts late


@pytest.mark.parametrize(
    ("intents", "at_s", "weights", "targets", "bound", "until", "ids"),
    [
        pytest.param([], 100.0, DEFAULT_OBJECTIVE.weights.as_list(), [0, 0, 0], None, None, (),
                     id="nothing-active-default-weights"),
        pytest.param([A], 100.0, [0.6, 0.2, 0.2], [20, 5, 0], 20.0, 600.0, ("a",), id="one-intent-as-is"),
        pytest.param([A, B], 100.0, [0.1, 0.8, 0.1], [20, 10, 8], 10.0, 300.0, ("a", "b"),
                     id="max-target-tightest-bound-newest-weights"),
        pytest.param([B, A], 100.0, [0.1, 0.8, 0.1], [20, 10, 8], 10.0, 300.0, ("a", "b"),
                     id="list-order-does-not-matter"),
        pytest.param([A, B], 300.0, [0.6, 0.2, 0.2], [20, 5, 0], 20.0, 600.0, ("a",),
                     id="newest-expired-previous-weights-back"),
        pytest.param([A, B], 20.0, [0.6, 0.2, 0.2], [20, 5, 0], 20.0, 600.0, ("a",),
                     id="newest-not-started-yet"),
        pytest.param([A, B], 700.0, DEFAULT_OBJECTIVE.weights.as_list(), [0, 0, 0], None, None, (),
                     id="all-expired-default"),
        pytest.param([A, B_SAME_TIME], 100.0, [0.1, 0.8, 0.1], [20, 5, 0], 20.0, 600.0, ("a", "b"),
                     id="tie-on-issued-at-broken-by-id"),
        # LATER starts after A but was issued before it: A is the newest, its weights apply.
        pytest.param([A, LATER], 550.0, [0.6, 0.2, 0.2], [20, 5, 0], 20.0, 600.0, ("a", "later"),
                     id="newest-means-issued-last-not-started-last"),
    ],
)
def test_merge_table(intents, at_s, weights, targets, bound, until, ids) -> None:
    merged = merge(intents, at_s)
    obj = merged.objective
    assert obj.weights.as_list() == pytest.approx(weights)
    assert obj.min_thr_mbps.as_list() == pytest.approx(targets)
    assert obj.max_latency_ms == bound
    assert obj.valid_from == at_s
    assert obj.valid_until == until  # the earliest end: the next moment to merge again
    assert merged.intent_ids == ids


def test_window_end_is_excluded() -> None:
    """At exactly valid_until the intent is no longer active, so touching windows never overlap."""
    first = intent("first", 0.0, targets=(30.0, 0.0, 0.0), window=(0.0, 100.0))
    second = intent("second", 1.0, targets=(0.0, 30.0, 0.0), window=(100.0, 200.0))
    assert merge([first, second], 100.0).intent_ids == ("second",)
