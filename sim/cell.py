"""Cell model: how a PRB share turns into throughput, PRB use and latency for one slice.

Each slice is a "fluid queue": traffic is treated like water poured into a bucket
(the backlog) and drained at the rate the slice's PRBs allow. No packets, no radio
channel. Every formula is explained in docs/learn/M01-simulator.md.
"""

from dataclasses import dataclass, field

import numpy as np

from verifier.objective import SLICES


@dataclass(frozen=True)
class CellConfig:
    """Fixed parameters of the simulated cell."""

    n_prb: int = 106  # 40 MHz at 30 kHz SCS, the usual OAI rfsim configuration
    # Measured on the OAI rfsim testbed in the M06 spike (OAI v3.0.0, 106 PRB, 30 kHz, TDD 7/2):
    # one UE alone 76 Mb/s, three UEs together 74.8 Mb/s in total. See docs/learn/M06-spike.md.
    capacity_mbps: float = 75.0
    dt_s: float = 1.0  # one step = one RL decision
    # Service time per slice (ms): the latency of an almost empty slice.
    base_latency_ms: dict[str, float] = field(
        default_factory=lambda: {"embb": 10.0, "urllc": 2.0, "be": 20.0}
    )
    max_backlog_mb: float = 50.0  # buffer size per slice; traffic above it is dropped
    # Caps rho so base / (1 - rho) stays finite and not too jumpy near 1 (at most 20 x base).
    # Above it, overload shows up through the growing backlog instead.
    max_utilisation: float = 0.95

    @property
    def mbps_per_prb(self) -> float:
        """Every PRB carries the same rate (simplification: no MCS difference per slice)."""
        return self.capacity_mbps / self.n_prb


@dataclass(frozen=True)
class SliceStep:
    """What happened to one slice during one step."""

    throughput_mbps: float
    backlog_mb: float  # left in the queue at the end of the step
    dropped_mb: float  # did not fit in the buffer
    prb_used: float  # fraction of the slice's own PRBs that carried data, 0..1
    latency_ms: float


def shares_from_action(action: np.ndarray) -> np.ndarray:
    """Turn the policy's raw action (3 numbers in [0, 1]) into PRB shares that sum to 1.

    We divide by the sum. If the policy outputs all zeros, we fall back to an equal split
    rather than dividing by zero.
    """
    clipped = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
    total = clipped.sum()
    if total <= 1e-9:
        return np.full(len(SLICES), 1.0 / len(SLICES))
    return clipped / total


def smooth_split(previous: np.ndarray | None, new: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential smoothing of the PRB split: alpha * new + (1 - alpha) * previous.

    Both splits sum to 1, so the result does too. With no previous split (first step of an
    episode) the new one is applied as is. Used in two places that must stay identical: the
    deployed policy (rl/policy.py) and, during training, the environment itself, so the
    policy learns with the same actuator lag it will have when deployed.
    """
    new = np.asarray(new, dtype=np.float64)
    if previous is None:
        return new
    return alpha * new + (1.0 - alpha) * np.asarray(previous, dtype=np.float64)


def apply_min_share(shares: np.ndarray, floor: float) -> np.ndarray:
    """Guarantee every slice at least `floor` of the PRBs: the last step before a split is applied.

    Deterministic, and valid for ANY input (a policy can output NaN, negatives or all zeros):
    1. invalid values (NaN, inf, negative) count as 0; if nothing is left, use an equal split;
    2. scale to sum 1;
    3. raise every slice below the floor to exactly the floor, and take the PRBs this needs
       from the other slices in proportion to how far they are above the floor.
    A split already respecting the floor is returned unchanged (up to float rounding in step 2),
    so the floor only acts when it must, like the 3GPP minimum PRB ratio (verifier.objective.MIN_SLICE_SHARE). Step 3 always
    has enough to take from: the shares sum to 1 and n * floor <= 1, so the slices above the
    floor hold at least the deficit.

    Works on one split (shape (3,)) or many (shape (n, 3), one split per row: the oracle scores
    its 861 candidates at once). Used by the training env (sim/env.py), the RL runner
    (rl/runner.py) and the oracle (rl/oracle.py), so they all apply the same rule.
    """
    n = len(SLICES)
    if not 0.0 <= floor <= 1.0 / n:
        raise ValueError(f"floor must be in [0, 1/{n}], got {floor}")
    values = np.array(shares, dtype=np.float64)  # a copy
    values[~np.isfinite(values) | (values < 0)] = 0.0
    total = values.sum(axis=-1, keepdims=True)
    empty = total <= 1e-12
    values = np.where(empty, 1.0 / n, values / np.where(empty, 1.0, total))
    deficit = np.maximum(floor - values, 0.0).sum(axis=-1, keepdims=True)
    excess = np.maximum(values - floor, 0.0)
    taken = deficit * excess / np.maximum(excess.sum(axis=-1, keepdims=True), 1e-12)
    return np.where(values < floor, floor, values - taken)


def step_slice(
    share: float,
    load_mbps: float,
    backlog_mb: float,
    base_latency_ms: float,
    cfg: CellConfig,
) -> SliceStep:
    """Advance one slice by one step (dt seconds).

    1. capacity of the slice = its share of the cell capacity
    2. arrivals during the step = load x dt
    3. served = what is waiting, up to what the PRBs can carry
    4. new backlog = what is left, capped by the buffer (the rest is dropped)
    5. PRB use = served / what the PRBs could have carried
    6. latency = M/M/1 time in system + time to drain the leftover backlog
    """
    cap_mbps = share * cfg.capacity_mbps
    arrivals_mb = load_mbps * cfg.dt_s
    waiting_mb = backlog_mb + arrivals_mb
    servable_mb = cap_mbps * cfg.dt_s

    served_mb = min(waiting_mb, servable_mb)
    leftover_mb = waiting_mb - served_mb
    new_backlog_mb = min(leftover_mb, cfg.max_backlog_mb)
    dropped_mb = leftover_mb - new_backlog_mb

    prb_used = served_mb / servable_mb if servable_mb > 0 else 0.0
    latency_ms = slice_latency_ms(cap_mbps, load_mbps, new_backlog_mb, base_latency_ms, cfg)

    return SliceStep(
        throughput_mbps=served_mb / cfg.dt_s,
        backlog_mb=new_backlog_mb,
        dropped_mb=dropped_mb,
        prb_used=prb_used,
        latency_ms=latency_ms,
    )


def slice_latency_ms(
    cap_mbps: float,
    load_mbps: float,
    backlog_mb: float,
    base_latency_ms: float,
    cfg: CellConfig,
) -> float:
    """Mean latency of a slice, in ms.

    First term, M/M/1: mean time in system T = (1/mu) / (1 - rho). base_latency_ms plays
    the role of the service time 1/mu, and rho = load / capacity is how full the slice is.
    Latency grows smoothly as the slice fills up.

    Second term: time to send what is still queued at the slice's rate. It keeps growing
    while the slice is overloaded, because the backlog grows.

    A slice with zero PRBs can send nothing: we treat that as full utilisation and drain
    time computed at a tiny rate, so latency is very large but finite.
    """
    safe_cap = max(cap_mbps, 1e-6)
    rho = min(load_mbps / safe_cap, cfg.max_utilisation)
    queueing_ms = base_latency_ms / (1.0 - rho)
    drain_ms = 1000.0 * backlog_mb / safe_cap
    return queueing_ms + drain_ms


def step_slices_batch(
    shares: np.ndarray,
    loads_mbps: np.ndarray,
    backlog_mb: np.ndarray,
    base_latency_ms: np.ndarray,
    cfg: CellConfig,
) -> dict[str, np.ndarray]:
    """step_slice for many candidate splits at once (numpy), used by the oracle (rl/oracle.py).

    Inputs broadcast together, e.g. shares (n, 3) against loads (3,). Same six steps and the
    same formulas as step_slice and slice_latency_ms above; a test checks they agree.
    Returns arrays "throughput_mbps", "backlog_mb", "latency_ms".
    """
    cap_mbps = shares * cfg.capacity_mbps
    waiting_mb = backlog_mb + loads_mbps * cfg.dt_s
    served_mb = np.minimum(waiting_mb, cap_mbps * cfg.dt_s)
    new_backlog_mb = np.minimum(waiting_mb - served_mb, cfg.max_backlog_mb)
    safe_cap = np.maximum(cap_mbps, 1e-6)
    rho = np.minimum(loads_mbps / safe_cap, cfg.max_utilisation)
    latency_ms = base_latency_ms / (1.0 - rho) + 1000.0 * new_backlog_mb / safe_cap
    return {"throughput_mbps": served_mb / cfg.dt_s, "backlog_mb": new_backlog_mb, "latency_ms": latency_ms}
