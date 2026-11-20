"""The runtime monitor (M11b): the formula takes over an intent the RL policy keeps missing.

The verifier checks that an objective is feasible; it can't check that the policy then meets it (review §3.4: M08d's
S3, M09b's F4a and M10c′'s F3 are accepted objectives the RL missed). This monitor watches the outcome, in the fast
loop, every tick:

1. The formula executor (rl.baselines.FormulaEstPolicy, frozen since M10b) runs in shadow on the same observation as
   the policy. It is arithmetic: microseconds, no LLM, no network (hard invariant 5).
2. A slice is *missed* when a target in force isn't met: its throughput under min(target, load) by more than
   max(1 Mb/s, 5 %) (M10d's tolerance: measured throughput is noisy), or URLLC's latency over the bound.
3. The miss *qualifies* when the formula would fix it: the formula grants that slice its whole sized need (it isn't
   cut by an overload) and gives it at least one PRB more than the policy does. Otherwise the channel, not the
   policy, is the problem, and switching wouldn't help.
4. After FALLBACK_AFTER_S qualifying seconds in a row, the formula executes the split until the a1_policy segment
   ends. Any tick that doesn't qualify resets the count. No switching back within a segment: no flapping.
5. When the segment changes (new objective or expiry), the policy decides again.

Pure: no I/O, no clock read, no randomness. The runner logs what it decides (rl/runner.py).
"""

from dataclasses import dataclass

import numpy as np

from rl.baselines import FormulaPolicy
from verifier.objective import SLICES, RLObjective

# Revision 1 of M11b's one allowed revision (DECISIONS 2026-11-20): rule A failed at 20 s, the time the served filter
# (alpha 0.10) takes to cover 90 % of a step. The misses come in short runs (longest qualifying run 9 s on m09 F2), so
# 20 in a row was rarely reached: the fallback fired in 13 of 160 and 18 of 200 dev runs.
FALLBACK_AFTER_S = 10.0
MISS_TOLERANCE_MBPS = 1.0  # M10d's tolerance on a measured rate: max(1 Mb/s, 5 %)
MISS_TOLERANCE_FRACTION = 0.05
NEED_TOLERANCE = 1e-9  # float slack on "the formula grants the whole need"


def missed_slices(records: list, objective: RLObjective, loads: dict[str, float]) -> dict[str, str]:
    """{slice: 'throughput' | 'latency'} for each target in force that this tick's KPI records miss.

    records: the tick's KpiRecord list (store.kpi, not imported here: the monitor stays free of the store).
    loads: the loads the controller sees (rl.runner.seen_loads). An idle slice is never missed: met = min(target, load).
    """
    by_slice = {r.slice: r for r in records}
    missed = {}
    for s, target in zip(SLICES, objective.min_thr_mbps.as_list()):
        if target <= 0.0:
            continue
        need = min(target, loads[s])
        tolerance = max(MISS_TOLERANCE_MBPS, MISS_TOLERANCE_FRACTION * need)
        if by_slice[s].dl_thr_mbps < need - tolerance:
            missed[s] = "throughput"
    if objective.max_latency_ms is not None and "urllc" not in missed:
        latency = by_slice["urllc"].latency_ms
        if latency is not None and latency > objective.max_latency_ms:
            missed["urllc"] = "latency"
    return missed


@dataclass(frozen=True)
class Fallback:
    """Why the formula took over: what the runner logs and shows."""

    slice: str
    reason: str  # 'throughput' | 'latency'
    missed_s: float
    rl_share: float
    formula_share: float


class FallbackMonitor:
    """Decides, each tick, whether the policy's split or the formula's is applied. The runner calls `segment` first,
    then `step` while the policy decides, or `hold` on a tick where neither does (safe state, LLM only's rows)."""

    def __init__(self, formula: FormulaPolicy, n_prb: int, dt_s: float = 1.0,
                 after_s: float = FALLBACK_AFTER_S) -> None:
        self.formula = formula
        self.min_gain = 1.0 / n_prb  # "more" means at least one PRB more
        self.dt_s = dt_s
        self.after_ticks = max(1, round(after_s / dt_s))
        self.count = 0  # qualifying ticks in a row
        self.active = False  # the formula executes the current segment
        self.policy_id: int | None = None  # the segment seen last tick
        self.fired: Fallback | None = None  # set on the tick the fallback starts, for the runner to log

    def segment(self, policy_id: int | None) -> bool:
        """Note the segment in force. True if a fallback was active and has just ended because the segment changed."""
        if policy_id == self.policy_id:
            return False
        self.policy_id, self.count = policy_id, 0
        ended, self.active = self.active, False
        return ended

    def hold(self) -> None:
        """A tick where the policy doesn't decide: the count starts again (misses must be the policy's, in a row)."""
        self.count = 0

    def step(self, obs: np.ndarray, objective: RLObjective, records: list, loads: dict[str, float],
             rl_shares: np.ndarray) -> np.ndarray | None:
        """The formula's split if the formula executes this tick, else None (the policy's split stands)."""
        self.fired = None
        formula_shares = np.asarray(self.formula.shares(obs), dtype=np.float64)  # in shadow, every tick
        if self.active:
            return formula_shares
        need = self.formula.last_need
        qualifying = [(s, why) for s, why in missed_slices(records, objective, loads).items()
                      if formula_shares[SLICES.index(s)] >= need[SLICES.index(s)] - NEED_TOLERANCE
                      and formula_shares[SLICES.index(s)] >= rl_shares[SLICES.index(s)] + self.min_gain]
        if not qualifying:
            self.count = 0
            return None
        self.count += 1
        if self.count < self.after_ticks:
            return None
        s, why = qualifying[0]
        i = SLICES.index(s)
        self.active = True
        self.fired = Fallback(s, why, self.count * self.dt_s, float(rl_shares[i]), float(formula_shares[i]))
        return formula_shares
