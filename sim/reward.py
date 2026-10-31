"""Reward: how good one step was, judged against the current objective.

The same KPIs give different rewards under different objectives. That is what makes the
policy goal-conditioned: it must read the objective to know what earns reward.

    r = sum_s w_s * thr_s / C                                  weighted useful throughput
        - 2 * sum_s max(0, min(target_s, load_s) - thr_s) / C  shortfall on minimum targets
        - 1 * min(1, max(0, lat_urllc - bound) / bound)         URLLC latency over the bound

C is the cell capacity, so every term is a fraction and the reward does not change scale
when the capacity is recalibrated to the testbed.
"""

from collections.abc import Mapping

from sim.cell import CellConfig, SliceStep
from verifier.objective import SLICES, RLObjective

SHORTFALL_PENALTY = 2.0  # missing a target costs more than the throughput it would bring
LATENCY_PENALTY = 1.0


def compute_reward(
    objective: RLObjective,
    steps: Mapping[str, SliceStep],
    loads_mbps: Mapping[str, float],
    cfg: CellConfig,
) -> float:
    """Reward of one step. Pure function: same inputs, same output."""
    capacity = cfg.capacity_mbps
    weights = dict(zip(SLICES, objective.weights.as_list()))
    targets = dict(zip(SLICES, objective.min_thr_mbps.as_list()))

    utility = sum(weights[s] * steps[s].throughput_mbps for s in SLICES) / capacity

    # Only the part of a target that the slice actually asks for can be missed:
    # a 10 Mb/s target on a slice offering 4 Mb/s is met at 4 Mb/s.
    shortfall = sum(
        max(0.0, min(targets[s], loads_mbps[s]) - steps[s].throughput_mbps) for s in SLICES
    ) / capacity

    latency_excess = 0.0
    if objective.max_latency_ms is not None:
        bound = objective.max_latency_ms
        latency_excess = min(1.0, max(0.0, steps["urllc"].latency_ms - bound) / bound)

    return utility - SHORTFALL_PENALTY * shortfall - LATENCY_PENALTY * latency_excess
