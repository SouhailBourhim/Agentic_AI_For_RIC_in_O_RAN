"""Non-learning controllers to compare the RL policy against.

Both have the same shape as the policy (`observation -> action`), so the evaluation and
the plots treat all controllers the same way.
- equal_split: 1/3 of the PRBs to each slice, whatever happens (asked for by M02).
- proportional_to_load: each slice gets a share equal to its share of the offered load.
  A simple heuristic with no learning: checks that RL beats something smarter than equal.
"""

import numpy as np

from sim.env import KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS
from verifier.objective import SLICES

N_SLICES = len(SLICES)


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
       required_shares): a bound holds only while the slice isn't saturated, whatever its target.
    3. What is left goes to the slices in proportion to the load they still offer beyond step 1.
    No smoothing: arithmetic doesn't oscillate, and this is the strongest version of the ablation.
    """

    def __init__(self, urllc_base_latency_ms: float) -> None:
        self.base_ms = urllc_base_latency_ms

    def shares(self, obs: np.ndarray) -> np.ndarray:
        k = KPI_FEATURES_PER_SLICE * N_SLICES
        loads = obs[0:k:KPI_FEATURES_PER_SLICE].astype(np.float64)
        targets = obs[k + N_SLICES:k + 2 * N_SLICES].astype(np.float64)
        bound_ms = float(obs[k + 2 * N_SLICES]) * LATENCY_SCALE_MS
        need = np.minimum(targets, loads)
        urllc = SLICES.index("urllc")
        if bound_ms > 0:
            need[urllc] = loads[urllc] / max(1.0 - self.base_ms / bound_ms, 1e-3)
        need = np.minimum(need, 1.0)
        if need.sum() >= 1.0:
            return need / need.sum()
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
