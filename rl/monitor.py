"""The runtime monitor (M11b, M11b′): the formula takes over an intent the RL policy keeps missing.

The verifier checks that an objective is feasible; it can't check that the policy then meets it (review §3.4: M08d's
S3, M09b's F4a and M10c′'s F3 are accepted objectives the RL missed). This monitor watches the outcome, in the fast
loop, every tick:

1. The formula executor (rl.baselines.FormulaEstPolicy, frozen since M10b) runs in shadow on the same observation as
   the policy. It is arithmetic: microseconds, no LLM, no network (hard invariant 5).
2. A slice is *missed* when a target in force isn't met: its throughput under min(target, load) by more than the
   trigger's tolerance, or URLLC's latency over the bound.
3. The miss *qualifies* when the formula would fix it: the formula grants that slice its whole sized need (it isn't
   cut by an overload) and gives it at least one PRB more than the policy does. Otherwise the channel, not the
   policy, is the problem, and switching wouldn't help.
4. When at least `missed_s` of the last `window_s` seconds qualified, the formula executes the split until the
   a1_policy segment ends. No switching back within a segment: no flapping.
5. When the segment changes (new objective or expiry), the policy decides again. A tick where the policy doesn't
   decide (safe state, LLM only's rows) clears the window: the misses counted must be the policy's.

Two triggers were pre-registered (DECISIONS 2026-11-20), both this one mechanism: M11b's "N in a row" is k = W = N.

Pure: no I/O, no clock read, no randomness. The runner logs what it decides (rl/runner.py).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from rl.baselines import FormulaPolicy
from verifier.objective import SLICES, RLObjective

NEED_TOLERANCE = 1e-9  # float slack on "the formula grants the whole need"
# The scorer's own met rule (eval.metrics.THROUGHPUT_TOLERANCE), copied here so the monitor doesn't import eval.
EXACT_TOLERANCE_MBPS = 1e-9


@dataclass(frozen=True)
class Trigger:
    """When the monitor hands a segment to the formula.

    window_s, missed_s: fire when at least missed_s of the last window_s seconds had a qualifying miss.
    tolerance_mbps, tolerance_fraction: a throughput counts as missed below min(target, load) minus
    max(tolerance_mbps, tolerance_fraction x that need)."""

    name: str
    window_s: float
    missed_s: float
    tolerance_mbps: float
    tolerance_fraction: float


# M11b, revision 1: 10 s in a row, M10d's tolerance on a measured rate (max(1 Mb/s, 5 %)). Revision 0 was 20 s in a
# row. Both failed rule A on dev; the misses came in short runs (longest qualifying run 9 s on m09 F2), and on m09 F2
# about half the exact misses were within the tolerance.
M11B = Trigger("m11b", window_s=10.0, missed_s=10.0, tolerance_mbps=1.0, tolerance_fraction=0.05)
# M11b′: half of the last 20 s (the served filter's settle time: 0.9^22 = 0.1 at alpha 0.10), and the scorer's own met
# rule, so the monitor watches exactly what intent_met counts. A false alarm hands the segment to the executor that did
# better on dev: it costs little, and rule B measures it.
M11B_PRIME = Trigger("m11b_prime", window_s=20.0, missed_s=10.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                     tolerance_fraction=0.0)
TRIGGERS = {t.name: t for t in (M11B, M11B_PRIME)}
# What the served runner uses (demo, testbed). M11b's until M11b′ is scored on its test split (DECISIONS 2026-11-20).
SERVED_TRIGGER = M11B


def missed_slices(records: list, objective: RLObjective, loads: dict[str, float],
                  trigger: Trigger = SERVED_TRIGGER) -> dict[str, str]:
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
        tolerance = max(trigger.tolerance_mbps, trigger.tolerance_fraction * need)
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
    missed_s: float  # qualifying seconds in the window when it fired
    rl_share: float
    formula_share: float


class FallbackMonitor:
    """Decides, each tick, whether the policy's split or the formula's is applied. The runner calls `segment` first,
    then `step` while the policy decides, or `hold` on a tick where neither does (safe state, LLM only's rows)."""

    def __init__(self, formula: FormulaPolicy, n_prb: int, dt_s: float = 1.0,
                 trigger: Trigger = SERVED_TRIGGER) -> None:
        self.formula = formula
        self.trigger = trigger
        self.min_gain = 1.0 / n_prb  # "more" means at least one PRB more
        self.dt_s = dt_s
        self.missed_ticks = max(1, round(trigger.missed_s / dt_s))
        self.window: deque[bool] = deque(maxlen=max(1, round(trigger.window_s / dt_s)))  # qualifying, per tick
        self.active = False  # the formula executes the current segment
        self.policy_id: int | None = None  # the segment seen last tick
        self.fired: Fallback | None = None  # set on the tick the fallback starts, for the runner to log

    @property
    def count(self) -> int:
        """Qualifying ticks in the window."""
        return sum(self.window)

    def segment(self, policy_id: int | None) -> bool:
        """Note the segment in force. True if a fallback was active and has just ended because the segment changed."""
        if policy_id == self.policy_id:
            return False
        self.policy_id = policy_id
        self.window.clear()
        ended, self.active = self.active, False
        return ended

    def hold(self) -> None:
        """A tick where the policy doesn't decide: the window starts again (the misses must be the policy's)."""
        self.window.clear()

    def step(self, obs: np.ndarray, objective: RLObjective, records: list, loads: dict[str, float],
             rl_shares: np.ndarray) -> np.ndarray | None:
        """The formula's split if the formula executes this tick, else None (the policy's split stands)."""
        self.fired = None
        formula_shares = np.asarray(self.formula.shares(obs), dtype=np.float64)  # in shadow, every tick
        if self.active:
            return formula_shares
        need = self.formula.last_need
        qualifying = [(s, why) for s, why in missed_slices(records, objective, loads, self.trigger).items()
                      if formula_shares[SLICES.index(s)] >= need[SLICES.index(s)] - NEED_TOLERANCE
                      and formula_shares[SLICES.index(s)] >= rl_shares[SLICES.index(s)] + self.min_gain]
        self.window.append(bool(qualifying))
        if not qualifying or self.count < self.missed_ticks:
            return None
        s, why = qualifying[0]
        i = SLICES.index(s)
        self.active = True
        self.fired = Fallback(s, why, self.count * self.dt_s, float(rl_shares[i]), float(formula_shares[i]))
        return formula_shares
