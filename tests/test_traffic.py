"""Shapes of the traffic generators in sim/traffic.py."""

import numpy as np

from sim.traffic import TrafficPattern, sample_traffic
from verifier.objective import SLICES

TIMES = np.arange(0.0, 100.0, 1.0)


def test_steady_is_constant() -> None:
    p = TrafficPattern("steady", base_mbps=20.0, peak_mbps=99.0)
    assert {p.mean_load_at(t) for t in TIMES} == {20.0}


def test_spike_is_higher_only_inside_its_window() -> None:
    p = TrafficPattern("spike", base_mbps=10.0, peak_mbps=40.0, start_s=30.0, end_s=50.0)
    for t in TIMES:
        expected = 40.0 if 30.0 <= t < 50.0 else 10.0
        assert p.mean_load_at(t) == expected


def test_ramp_never_goes_down() -> None:
    p = TrafficPattern("ramp", base_mbps=10.0, peak_mbps=60.0, start_s=20.0, end_s=70.0)
    loads = [p.mean_load_at(t) for t in TIMES]
    assert all(b >= a for a, b in zip(loads, loads[1:]))
    assert loads[0] == 10.0 and loads[-1] == 60.0


def test_noise_is_seeded_and_never_negative() -> None:
    p = TrafficPattern("steady", base_mbps=1.0, peak_mbps=1.0, noise_std=2.0)
    a = [p.load_at(t, np.random.default_rng(7)) for t in TIMES]
    b = [p.load_at(t, np.random.default_rng(7)) for t in TIMES]
    assert a == b
    assert min(a) >= 0.0


def test_sample_traffic_covers_every_slice() -> None:
    traffic = sample_traffic(np.random.default_rng(0), episode_s=200.0)
    assert set(traffic) == set(SLICES)
