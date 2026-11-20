"""The shielded RL executor (M11d): URLLC never gets less than the formula would give it; the RL splits the rest.

Why: RL loses F3 (URLLC load flipping between ~10 % and ~45 % of the cell, with a latency bound) mainly by
under-sizing URLLC: about 0.42 of the cell where the formula gives 0.63 (M10c′), so every upward flip overloads the
slice and the bound breaks. The formula sizes a bounded URLLC to its 30 s peak load (rl.baselines.FormulaPolicy,
step 2). The shield borrows only that decision and leaves the rest of the split to the policy.

Every tick:
1. the RL policy gives its split (after its own filter);
2. the formula gives its split, every tick, so its 30 s peak window stays fed even while it isn't used;
3. if the objective has a URLLC latency bound and the RL gives URLLC less than the formula does, URLLC gets the
   formula's share and eMBB and best effort share the rest in the RL's own ratio;
4. otherwise the RL's split stands, exactly.
The runner's 5 % floor comes after, as for every executor. A safety layer of this kind (a classical controller
bounding a learned one) is what OnSlicing (CoNEXT 2021) and SafeSlice (ICMLCN 2025) do.

Pure arithmetic on the observation: no I/O, no LLM (hard invariant 5).
"""

import numpy as np

from rl.baselines import FormulaPolicy
from sim.env import KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS
from verifier.objective import SLICES

URLLC = SLICES.index("urllc")
OTHERS = [i for i in range(len(SLICES)) if i != URLLC]


def urllc_bound_ms(obs: np.ndarray) -> float:
    """The objective's URLLC latency bound in ms, 0 if none: where FormulaPolicy.shares reads it."""
    k = KPI_FEATURES_PER_SLICE * len(SLICES)
    return float(obs[k + 2 * len(SLICES)]) * LATENCY_SCALE_MS


def shield(rl_split: np.ndarray, formula_split: np.ndarray, bound_ms: float) -> tuple[np.ndarray, bool]:
    """(the split to apply, whether the shield changed it). Pure: tests drive it directly."""
    rl_split = np.asarray(rl_split, dtype=np.float64)
    floor = float(formula_split[URLLC])
    if bound_ms <= 0 or rl_split[URLLC] >= floor:
        return rl_split, False
    out = np.empty(len(SLICES))
    out[URLLC] = floor
    rest = rl_split[OTHERS]
    ratio = rest / rest.sum() if rest.sum() > 1e-12 else np.full(len(OTHERS), 1.0 / len(OTHERS))
    out[OTHERS] = (1.0 - floor) * ratio
    return out, True


class ShieldedPolicy:
    """Stands where the RL policy stands in the Runner: same `shares(obs)` call."""

    def __init__(self, rl, formula: FormulaPolicy) -> None:
        self.rl = rl
        self.formula = formula
        self.last_shielded = False  # did the shield change the last split (tests, figures)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        rl_split = self.rl.shares(obs)
        formula_split = self.formula.shares(obs)  # every tick, bound or not: its peak window
        split, self.last_shielded = shield(rl_split, formula_split, urllc_bound_ms(obs))
        return split if self.last_shielded else rl_split

    def restart_from(self, split: np.ndarray) -> None:
        """After the safe state or a monitor fallback, the RL's filter restarts from the split in force (M11a)."""
        self.rl.restart_from(split)
