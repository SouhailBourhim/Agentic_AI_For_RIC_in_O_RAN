"""Non-learning controllers to compare the RL policy against.

Both have the same shape as the policy (`observation -> action`), so the evaluation and
the plots treat all controllers the same way.
- equal_split: 1/3 of the PRBs to each slice, whatever happens (asked for by M02).
- proportional_to_load: each slice gets a share equal to its share of the offered load.
  A simple heuristic with no learning: checks that RL beats something smarter than equal.
"""

from collections import deque

import numpy as np

from sim.env import KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS
from verifier.objective import MIN_SLICE_SHARE, SLICES

N_SLICES = len(SLICES)
PEAK_WINDOW_S = 30  # LLM + formula sizes a bounded URLLC to its peak load over the last 30 s (DECISIONS)
SIZING_FRACTION = 74.0 / 79.0  # LLM + formula sizes against 74 Mb/s of a cell told 79, as LLM only v2's prompt (DECISIONS)


class EqualSplitPolicy:
    """Stands where the RL policy stands in the runner, for the controllers with no RL (M08a):
    static, and LLM only between two of the LLM's splits. Same `shares` call as SlicePolicy."""

    def shares(self, _obs: np.ndarray) -> np.ndarray:
        return np.full(N_SLICES, 1.0 / N_SLICES)


class FormulaPolicy:
    """LLM + formula (M09b's ablation): the verified objective turned into a split by arithmetic, with no RL.

    Stands where the RL policy stands, reads the same observation (loads and the objective in force, both as
    fractions of the capacity the cell reports), and the Runner applies its split as it applies the policy's.
    1. Each slice with a target needs min(target, load) of the cell: that is what "met" means (eval.metrics).
    2. URLLC under a latency bound needs its whole load, plus the verifier's margin (verifier.verify.
       required_shares): a bound holds only while the slice isn't saturated, whatever its target. Its load here
       is the peak of the last PEAK_WINDOW_S seconds, not the current one: the usual sizing for a latency bound
       under bursty load, and the rule LLM only v2's prompt gives the model ("above the highest recent load").
    3. Every need is sized against 74 Mb/s, not the 79 the cell reports: the cell carries 73.5-80.2 Mb/s
       (M08d, M09a), and a share sized to the load exactly misses "met" on a smaller cell.
    4. While URLLC's observed latency is over its bound, it gets all the slack the other needs leave, until
       the queue has drained. The observation clips latency at 1 s, so the queue's size can't be estimated
       from it; giving the slack doesn't need the size.
    5. If the needs add up to more than the cell, a bounded URLLC is served first and the throughput needs share
       what's left in proportion: a bound is all or nothing, since a saturated slice's queue grows without
       limit, while a throughput shortfall degrades gradually. Without a bound, all needs scale alike.
    6. What is left goes to the slices in proportion to the load they still offer beyond their need.
    The peak in step 2 and steps 3-5 were added after the dev benchmark, within a logged budget of revisions
    (DECISIONS 2026-11-08, docs/learn/data/M09b-formula-revisions.csv): without them, a queue built during a
    load jump never drained, and the verifier's implied fair-share targets scaled URLLC below its own load.
    No smoothing: arithmetic doesn't oscillate.
    """

    def __init__(self, urllc_base_latency_ms: float) -> None:
        self.base_ms = urllc_base_latency_ms
        self.recent_urllc: deque[float] = deque(maxlen=max(PEAK_WINDOW_S, 1))  # one observation per tick (1 s)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        k = KPI_FEATURES_PER_SLICE * N_SLICES
        loads = obs[0:k:KPI_FEATURES_PER_SLICE].astype(np.float64)
        targets = obs[k + N_SLICES:k + 2 * N_SLICES].astype(np.float64)
        bound_ms = float(obs[k + 2 * N_SLICES]) * LATENCY_SCALE_MS
        need = np.minimum(targets, loads) / SIZING_FRACTION
        urllc = SLICES.index("urllc")
        self.recent_urllc.append(float(loads[urllc]))
        if bound_ms > 0:
            peak = max(self.recent_urllc) if PEAK_WINDOW_S > 0 else float(loads[urllc])
            need[urllc] = peak / max(1.0 - self.base_ms / bound_ms, 1e-3) / SIZING_FRACTION
            latency_ms = float(obs[urllc * KPI_FEATURES_PER_SLICE + 2]) * LATENCY_SCALE_MS
            if latency_ms > bound_ms:
                need[urllc] = max(need[urllc], 1.0 - (need.sum() - need[urllc]))  # all the slack: drain the queue
        need = np.minimum(need, 1.0)
        if need.sum() >= 1.0:
            if bound_ms <= 0:
                return need / need.sum()
            first = min(need[urllc], 1.0 - MIN_SLICE_SHARE * (N_SLICES - 1))
            others = np.delete(need, urllc)
            others = others / others.sum() * (1.0 - first) if others.sum() > 1e-9 else np.full(N_SLICES - 1, (1.0 - first) / 2)
            return np.insert(others, urllc, first)
        rest = np.maximum(loads - need, 0.0)
        extra = (1.0 - need.sum()) * (rest / rest.sum() if rest.sum() > 1e-9 else np.full(N_SLICES, 1.0 / N_SLICES))
        return need + extra


def equal_split(_obs: np.ndarray) -> np.ndarray:
    return np.full(N_SLICES, 1.0 / N_SLICES, dtype=np.float32)


def proportional_to_load(obs: np.ndarray) -> np.ndarray:
    """Load/C is the first KPI feature of each slice's block in the observation.

    We normalise the loads to sum to 1 ourselves: a slice can offer more than the whole cell
    during a spike (load/C > 1), and the action space stops at 1, so clipping first would
    distort the proportions. With no load at all, fall back to an equal split.
    """
    loads = obs[0 : KPI_FEATURES_PER_SLICE * N_SLICES : KPI_FEATURES_PER_SLICE].astype(np.float64)
    total = loads.sum()
    if total <= 1e-9:
        return equal_split(obs)
    return (loads / total).astype(np.float32)
