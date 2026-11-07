"""Traffic generators: the offered load of one slice over time.

Four shapes, enough to build the evaluation scenarios ("steady then spike", ramps):
- steady: constant base load
- spike:  base load, then base x factor between start_s and end_s
- ramp:   base load until start_s, then linear up to peak at end_s, then peak
- steps:  piecewise constant, (start_s, Mb/s) pairs; plays the testbed's iperf3 segments (M09a)

Random noise is added on top, using the environment's seeded random generator so a
run can be repeated exactly.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from verifier.objective import SLICES

PatternKind = Literal["steady", "spike", "ramp", "steps"]

# Loads are given as fractions of the cell capacity, so the scenarios keep their meaning when
# the capacity is recalibrated (it was 100 Mb/s before the M06 spike measured 75 Mb/s).
# Typical mean load per slice: about 90 % busy in total, so any spike causes overload.
DEFAULT_LOAD_FRACTION: dict[str, float] = {"embb": 0.50, "urllc": 0.10, "be": 0.30}

# Ranges used when sampling random scenarios for training (fractions of capacity).
SAMPLE_LOAD_FRACTION_RANGE: dict[str, tuple[float, float]] = {
    "embb": (0.20, 0.60),
    "urllc": (0.02, 0.15),
    "be": (0.10, 0.40),
}


@dataclass(frozen=True)
class TrafficPattern:
    """Offered load of one slice as a function of time."""

    kind: PatternKind = "steady"
    base_mbps: float = 10.0
    peak_mbps: float = 10.0  # load at the top of the spike or the end of the ramp
    start_s: float = 0.0
    end_s: float = 0.0
    noise_std: float = 0.05  # multiplicative: 0.05 means about +/- 5 %
    steps: tuple[tuple[float, float], ...] = ()  # kind "steps" only: (start_s, Mb/s), sorted by start; 0 before the first

    def mean_load_at(self, t: float) -> float:
        """Load at time t without noise. Kept separate so tests can check the shape."""
        if self.kind == "steady":
            return self.base_mbps
        if self.kind == "spike":
            inside = self.start_s <= t < self.end_s
            return self.peak_mbps if inside else self.base_mbps
        if self.kind == "ramp":
            if t <= self.start_s:
                return self.base_mbps
            if t >= self.end_s:
                return self.peak_mbps
            progress = (t - self.start_s) / (self.end_s - self.start_s)
            return self.base_mbps + progress * (self.peak_mbps - self.base_mbps)
        if self.kind == "steps":
            current = 0.0
            for start, mbps in self.steps:
                if t < start:
                    break
                current = mbps
            return current
        raise ValueError(f"unknown traffic pattern kind: {self.kind}")

    def load_at(self, t: float, rng: np.random.Generator) -> float:
        """Load at time t with multiplicative Gaussian noise, never negative."""
        noise = 1.0 + self.noise_std * rng.standard_normal()
        return max(0.0, self.mean_load_at(t) * noise)


def default_traffic(capacity_mbps: float) -> dict[str, TrafficPattern]:
    """Steady load on every slice at the default levels."""
    loads = {s: DEFAULT_LOAD_FRACTION[s] * capacity_mbps for s in SLICES}
    return {s: TrafficPattern("steady", loads[s], loads[s]) for s in SLICES}


# Peak / base factor of sampled spikes and ramps. Training in M02b widens it to (1.5, 5.0) to
# cover the operating envelope of the evaluation scenarios ("steady then spike", up to x4).
DEFAULT_SURGE_RANGE = (1.5, 3.0)


def sample_traffic(
    rng: np.random.Generator,
    episode_s: float,
    capacity_mbps: float,
    surge_range: tuple[float, float] = DEFAULT_SURGE_RANGE,
) -> dict[str, TrafficPattern]:
    """A random scenario: each slice gets a random shape, level and timing.

    Used in training (M02) so the policy sees many situations, not one scripted scenario.
    The random generator is called in the same order whatever surge_range is, so the
    default reproduces exactly the episodes defined before the option existed.
    """
    traffic = {}
    for s in SLICES:
        low, high = SAMPLE_LOAD_FRACTION_RANGE[s]
        base = float(rng.uniform(low, high)) * capacity_mbps
        kind: PatternKind = str(rng.choice(["steady", "spike", "ramp"]))  # type: ignore[assignment]
        peak = base * float(rng.uniform(*surge_range))
        start = float(rng.uniform(0.1, 0.6) * episode_s)
        end = float(start + rng.uniform(0.1, 0.4) * episode_s)
        traffic[s] = TrafficPattern(kind, base, peak, start, end)
    return traffic
