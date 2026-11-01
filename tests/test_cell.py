"""Properties of the per-slice cell model in sim/cell.py."""

import numpy as np
import pytest

from sim.cell import CellConfig, shares_from_action, slice_latency_ms, step_slice

# A fixed 100 Mb/s cell, so the arithmetic in these tests stays readable (share 0.1 -> 10 Mb/s)
# whatever the calibrated default capacity is.
CFG = CellConfig(capacity_mbps=100.0)
BASE_MS = 2.0


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.2, 0.5, 0.3]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([5.0, -1.0, 0.5]),  # out of range: clipped to [0, 1] first
    ],
)
def test_shares_sum_to_one_and_stay_in_range(action: np.ndarray) -> None:
    shares = shares_from_action(action)
    assert shares.sum() == pytest.approx(1.0)
    assert np.all(shares >= 0.0) and np.all(shares <= 1.0)


def test_all_zero_action_gives_equal_split() -> None:
    assert shares_from_action(np.zeros(3)) == pytest.approx(np.full(3, 1 / 3))


@pytest.mark.parametrize("load", [0.0, 5.0, 30.0, 80.0, 150.0])
@pytest.mark.parametrize("backlog", [0.0, 10.0, 50.0])
def test_more_prbs_never_lower_throughput(load: float, backlog: float) -> None:
    shares = np.linspace(0.0, 1.0, 101)
    thr = [step_slice(s, load, backlog, BASE_MS, CFG).throughput_mbps for s in shares]
    assert all(b >= a for a, b in zip(thr, thr[1:]))


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("load", [0.0, 20.0, 200.0])
def test_never_serves_more_than_waiting(share: float, load: float) -> None:
    backlog = 3.0
    step = step_slice(share, load, backlog, BASE_MS, CFG)
    assert step.throughput_mbps * CFG.dt_s <= backlog + load * CFG.dt_s + 1e-9
    assert 0.0 <= step.prb_used <= 1.0


def test_overloaded_slice_shows_rising_latency() -> None:
    """Offer 30 Mb/s to a slice that can carry 10 Mb/s: the queue grows, so does latency."""
    share, load = 0.1, 30.0
    backlog, latencies = 0.0, []
    while backlog < CFG.max_backlog_mb:
        step = step_slice(share, load, backlog, BASE_MS, CFG)
        latencies.append(step.latency_ms)
        backlog = step.backlog_mb
    assert len(latencies) > 2
    assert all(b > a for a, b in zip(latencies, latencies[1:]))


def test_overflow_is_dropped_and_counted() -> None:
    step = step_slice(0.1, 30.0, CFG.max_backlog_mb, BASE_MS, CFG)
    assert step.backlog_mb == CFG.max_backlog_mb
    assert step.dropped_mb == pytest.approx(20.0)  # 30 in, 10 out, buffer already full


def test_underloaded_latency_follows_mm1() -> None:
    """No backlog: latency = base / (1 - rho)."""
    share, load = 0.5, 25.0  # capacity 50 Mb/s, rho = 0.5
    step = step_slice(share, load, 0.0, BASE_MS, CFG)
    assert step.backlog_mb == 0.0
    assert step.latency_ms == pytest.approx(BASE_MS / (1 - 0.5))


def test_latency_rises_with_utilisation() -> None:
    loads = np.linspace(0.0, 45.0, 10)  # up to rho = 0.9 on a 50 Mb/s slice
    lat = [slice_latency_ms(50.0, load, 0.0, BASE_MS, CFG) for load in loads]
    assert all(b > a for a, b in zip(lat, lat[1:]))
