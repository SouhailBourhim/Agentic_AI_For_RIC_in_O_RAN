"""Reward: how good one step was, judged against the current objective.

The same KPIs give different rewards under different objectives. That is what makes the
policy goal-conditioned: it must read the objective to know what earns reward.

    r = sum_s w_s * thr_s / C                                  weighted useful throughput
        - 2 * sum_s max(0, min(target_s, load_s) - thr_s) / C  shortfall on minimum targets
        - 1 * log(1 + max(0, lat_urllc - bound) / bound)        URLLC latency over the bound

C is the cell capacity, so every term is a fraction and the reward does not change scale
when the capacity is recalibrated to the testbed.

Reward versions. "v2" (default since M02b) uses the log latency term above: 0 at the bound,
0.69 at twice the bound, 2.3 at 10x, 5.1 at 170x, so it keeps growing and always gives the
policy a direction. "v1" (M01 and M02 rounds 1-2) capped it: min(1, excess / bound). Beyond
twice the bound v1 is flat, so 20 ms and 1 700 ms cost the same, and nothing leads a policy
towards draining a URLLC queue after a surge. v1 is kept for the M02b ablation and the record.

"v3" (M09c) is v1 plus an over-serving term:

    - LAMBDA * min(sum_s surplus_s, sum_s shortfall_s) / C

surplus_s is the capacity a slice was given above its need (share x C - need, for slices not
short), its need being min(target, load), its load if it has no target, and for URLLC under a
bound its load plus the verifier's margin, load / (1 - base / bound). shortfall_s is what the
targeted slices miss (the term above). So surplus costs only while some slice is short, and
only up to what it misses: giving spare capacity away is free, starving a target isn't. M09b
found v7 giving URLLC 0.43-0.53 of the cell where 0.34-0.36 was needed, while BE missed its
target (F4a). It needs the split and the URLLC base latency the verifier sizes with.

The work is done by compute_reward_batch (numpy, many objectives at once), which the
objective relabeling in rl/relabel.py and the oracle in rl/oracle.py need. compute_reward is
the same calculation for one step, so there is a single implementation.
"""

from collections.abc import Mapping
from typing import Literal

import numpy as np

from sim.cell import CellConfig, SliceStep
from verifier.objective import SLICES, RLObjective

SHORTFALL_PENALTY = 2.0  # missing a target costs more than the throughput it would bring
LATENCY_PENALTY = 1.0
OVERSERVE_PENALTY = 1.0  # LAMBDA of reward v3
RewardVersion = Literal["v1", "v2", "v3"]
DEFAULT_REWARD_VERSION: RewardVersion = "v2"


def compute_reward_batch(
    weights: np.ndarray,
    targets_mbps: np.ndarray,
    bound_ms: np.ndarray,
    throughput_mbps: np.ndarray,
    loads_mbps: np.ndarray,
    urllc_latency_ms: np.ndarray,
    capacity_mbps: float,
    version: RewardVersion = DEFAULT_REWARD_VERSION,
    shares: np.ndarray | None = None,
    urllc_base_ms: float | None = None,
) -> np.ndarray:
    """Reward of n steps (or of one step under n objectives), vectorised.

    Shapes: weights, targets_mbps, throughput_mbps, loads_mbps (n, 3) in SLICES order;
    bound_ms and urllc_latency_ms (n,). A bound of NaN means "no latency bound".
    Version "v3" also needs shares (n, 3), the PRB split applied, and urllc_base_ms.
    """
    utility = (weights * throughput_mbps).sum(axis=1) / capacity_mbps
    # Only the part of a target that the slice actually asks for can be missed:
    # a 10 Mb/s target on a slice offering 4 Mb/s is met at 4 Mb/s.
    needed = np.minimum(targets_mbps, loads_mbps)
    shortfall = np.maximum(0.0, needed - throughput_mbps).sum(axis=1) / capacity_mbps

    has_bound = ~np.isnan(bound_ms)
    safe_bound = np.where(has_bound, bound_ms, 1.0)
    excess = np.maximum(0.0, urllc_latency_ms - safe_bound) / safe_bound
    latency_term = np.log1p(excess) if version == "v2" else np.minimum(1.0, excess)  # v3 keeps v1's
    latency_term = np.where(has_bound, latency_term, 0.0)

    reward = utility - SHORTFALL_PENALTY * shortfall - LATENCY_PENALTY * latency_term
    if version == "v3":
        if shares is None or urllc_base_ms is None:
            raise ValueError("reward v3 needs the split (shares) and the URLLC base latency")
        reward = reward - OVERSERVE_PENALTY * overserving(targets_mbps, bound_ms, throughput_mbps, loads_mbps,
                                                          shares, capacity_mbps, urllc_base_ms)
    return reward


def overserving(targets_mbps: np.ndarray, bound_ms: np.ndarray, throughput_mbps: np.ndarray,
                loads_mbps: np.ndarray, shares: np.ndarray, capacity_mbps: float, urllc_base_ms: float) -> np.ndarray:
    """min(surplus, shortfall) / C per row: the capacity given above need while some target is missed."""
    need = np.where(targets_mbps > 0, np.minimum(targets_mbps, loads_mbps), loads_mbps)
    u = SLICES.index("urllc")
    has_bound = ~np.isnan(bound_ms)
    margin = 1.0 - urllc_base_ms / np.where(has_bound, bound_ms, np.inf)
    need[:, u] = np.where(has_bound, np.maximum(need[:, u], loads_mbps[:, u] / np.maximum(margin, 1e-3)), need[:, u])
    short = np.maximum(0.0, np.minimum(targets_mbps, loads_mbps) - throughput_mbps)
    surplus = np.where(short > 0, 0.0, np.maximum(0.0, shares * capacity_mbps - need))
    return np.minimum(surplus.sum(axis=1), short.sum(axis=1)) / capacity_mbps


def compute_reward(
    objective: RLObjective,
    steps: Mapping[str, SliceStep],
    loads_mbps: Mapping[str, float],
    cfg: CellConfig,
    version: RewardVersion = DEFAULT_REWARD_VERSION,
    shares: np.ndarray | None = None,
    urllc_base_ms: float | None = None,
) -> float:
    """Reward of one step. Pure function: same inputs, same output."""
    bound = np.nan if objective.max_latency_ms is None else objective.max_latency_ms
    reward = compute_reward_batch(
        weights=np.array([objective.weights.as_list()]),
        targets_mbps=np.array([objective.min_thr_mbps.as_list()]),
        bound_ms=np.array([bound]),
        throughput_mbps=np.array([[steps[s].throughput_mbps for s in SLICES]]),
        loads_mbps=np.array([[loads_mbps[s] for s in SLICES]]),
        urllc_latency_ms=np.array([steps["urllc"].latency_ms]),
        capacity_mbps=cfg.capacity_mbps,
        version=version,
        shares=None if shares is None else np.asarray(shares, dtype=np.float64)[None, :],
        urllc_base_ms=urllc_base_ms,
    )
    return float(reward[0])
