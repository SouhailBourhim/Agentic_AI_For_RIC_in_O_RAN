"""Non-learning controllers to compare the RL policy against.

Both have the same shape as the policy (`observation -> action`), so the evaluation and
the plots treat all controllers the same way.
- equal_split: 1/3 of the PRBs to each slice, whatever happens (asked for by M02).
- proportional_to_load: each slice gets a share equal to its share of the offered load.
  A simple heuristic with no learning: checks that RL beats something smarter than equal.
"""

import numpy as np

from sim.env import KPI_FEATURES_PER_SLICE
from verifier.objective import SLICES

N_SLICES = len(SLICES)


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
