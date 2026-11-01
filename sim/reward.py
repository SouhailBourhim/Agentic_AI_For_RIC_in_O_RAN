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
RewardVersion = Literal["v1", "v2"]
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
) -> np.ndarray:
    """Reward of n steps (or of one step under n objectives), vectorised.

    Shapes: weights, targets_mbps, throughput_mbps, loads_mbps (n, 3) in SLICES order;
    bound_ms and urllc_latency_ms (n,). A bound of NaN means "no latency bound".
    """
    utility = (weights * throughput_mbps).sum(axis=1) / capacity_mbps
    # Only the part of a target that the slice actually asks for can be missed:
    # a 10 Mb/s target on a slice offering 4 Mb/s is met at 4 Mb/s.
    needed = np.minimum(targets_mbps, loads_mbps)
    shortfall = np.maximum(0.0, needed - throughput_mbps).sum(axis=1) / capacity_mbps

    has_bound = ~np.isnan(bound_ms)
    safe_bound = np.where(has_bound, bound_ms, 1.0)
    excess = np.maximum(0.0, urllc_latency_ms - safe_bound) / safe_bound
    latency_term = np.log1p(excess) if version == "v2" else np.minimum(1.0, excess)
    latency_term = np.where(has_bound, latency_term, 0.0)

    return utility - SHORTFALL_PENALTY * shortfall - LATENCY_PENALTY * latency_term


def compute_reward(
    objective: RLObjective,
    steps: Mapping[str, SliceStep],
    loads_mbps: Mapping[str, float],
    cfg: CellConfig,
    version: RewardVersion = DEFAULT_REWARD_VERSION,
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
    )
    return float(reward[0])
