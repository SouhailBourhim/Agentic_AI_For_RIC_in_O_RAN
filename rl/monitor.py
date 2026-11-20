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

from rl.baselines import FORMULA_EST_MARGIN, FormulaPolicy
from verifier.objective import SLICES, RLObjective

NEED_TOLERANCE = 1e-9  # float slack on "the formula grants the whole need"
# The scorer's own met rule (eval.metrics.THROUGHPUT_TOLERANCE), copied here so the monitor doesn't import eval.
EXACT_TOLERANCE_MBPS = 1e-9


@dataclass(frozen=True)
class Trigger:
    """When the monitor hands a segment to the formula.

    window_s, missed_s: fire when at least missed_s of the last window_s seconds had a qualifying miss.
    tolerance_mbps, tolerance_fraction: a throughput counts as missed below min(target, load) minus
    max(tolerance_mbps, tolerance_fraction x that need).
    stated_only: watch only the throughput targets an intent stated (the a1_policy row's stated_slices), not the ones
    the verifier implied for the other slices. The latency bound is always stated (nothing implies one).
    per_intent: the formula keeps executing while any intent it took over is still in force, across segments (another
    intent starting or ending makes a new segment); False = until the segment ends.
    need_margin: the share of its sized need the formula may leave out and still "meet" the target: its own sizing
    margin (rl.baselines.FORMULA_EST_MARGIN), so the question becomes "would its split meet the target at the measured
    rate". 0 = the whole sized need."""

    name: str
    window_s: float
    missed_s: float
    tolerance_mbps: float
    tolerance_fraction: float
    stated_only: bool = False
    per_intent: bool = False
    need_margin: float = 0.0


# M11b, revision 1: 10 s in a row, M10d's tolerance on a measured rate (max(1 Mb/s, 5 %)). Revision 0 was 20 s in a
# row. Both failed rule A on dev; the misses came in short runs (longest qualifying run 9 s on m09 F2), and on m09 F2
# about half the exact misses were within the tolerance.
M11B = Trigger("m11b", window_s=10.0, missed_s=10.0, tolerance_mbps=1.0, tolerance_fraction=0.05)
# M11b′: half of the last 20 s (the served filter's settle time: 0.9^22 = 0.1 at alpha 0.10), and the scorer's own met
# rule, so the monitor watches exactly what intent_met counts. A false alarm hands the segment to the executor that did
# better on dev: it costs little, and rule B measures it.
M11B_PRIME = Trigger("m11b_prime", window_s=20.0, missed_s=10.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                     tolerance_fraction=0.0)
# M11b′ revision 3: the same, watching only the stated throughput targets. On m10 F2 (stated target on eMBB or BE),
# 28 of r2's 33 fallbacks fired on URLLC's implied target, which nobody stated and nothing scores; there the policy
# alone beat the formula, and rule B failed.
M11B_PRIME_STATED = Trigger("m11b_prime_stated", window_s=20.0, missed_s=10.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                            tolerance_fraction=0.0, stated_only=True)
# M11b′ revision 4: stated targets only, 5 of the last 20 s. r3 fixed B (m10 F2: rl_mon = rl) but A still failed where
# the policy is far below the formula (m10 F3; m09 F2, F3, F4): the cost left is the time before the handoff.
M11B_PRIME_K5 = Trigger("m11b_prime_k5", window_s=20.0, missed_s=5.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                        tolerance_fraction=0.0, stated_only=True)
# M11b′ revision 5: r4, and the handoff lasts for the intent, as the review words it ("the formula executes that
# intent"). r4 failed A only on m09 F4, whose two intents overlap: when the second one started, the new segment gave the
# policy back, it missed the first intent's BE target again, and the formula couldn't qualify in the merged segment.
M11B_PRIME_INTENT = Trigger("m11b_prime_intent", window_s=20.0, missed_s=5.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                            tolerance_fraction=0.0, stated_only=True, per_intent=True)
# M11b′ revision 6: r5, and the formula qualifies when its split covers its need less its own sizing margin. r5 failed
# A only on m09 F4, where 3 of 15 intents never handed over: with an implied URLLC target the formula's needs added up
# to just over the cell, its overload branch scaled every slice down a hair (BE 0.3257 of a 0.326 need), so it never
# "granted the whole need", yet formula_est alone met those intents (its 0.15 margin).
M11B_PRIME_MEETS = Trigger("m11b_prime_meets", window_s=20.0, missed_s=5.0, tolerance_mbps=EXACT_TOLERANCE_MBPS,
                           tolerance_fraction=0.0, stated_only=True, per_intent=True, need_margin=FORMULA_EST_MARGIN)
TRIGGERS = {t.name: t for t in (M11B, M11B_PRIME, M11B_PRIME_STATED, M11B_PRIME_K5, M11B_PRIME_INTENT,
                                M11B_PRIME_MEETS)}
# What the served runner uses (demo, testbed). M11b's until M11b′ is scored on its test split (DECISIONS 2026-11-20).
SERVED_TRIGGER = M11B


def missed_slices(records: list, objective: RLObjective, loads: dict[str, float],
                  trigger: Trigger = SERVED_TRIGGER, stated_slices: tuple[str, ...] | None = None) -> dict[str, str]:
    """{slice: 'throughput' | 'latency'} for each watched target in force that this tick's KPI records miss.

    records: the tick's KpiRecord list (store.kpi, not imported here: the monitor stays free of the store).
    loads: the loads the controller sees (rl.runner.seen_loads). An idle slice is never missed: met = min(target, load).
    stated_slices: the segment's stated throughput targets; with a stated_only trigger only those are watched
    (None = unknown: every target is).
    """
    by_slice = {r.slice: r for r in records}
    watched = stated_slices if trigger.stated_only and stated_slices is not None else SLICES
    missed = {}
    for s, target in zip(SLICES, objective.min_thr_mbps.as_list()):
        if target <= 0.0 or s not in watched:
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
        self.taken_over: frozenset[str] = frozenset()  # the intents the formula executes (per_intent triggers)
        self.fired: Fallback | None = None  # set on the tick the fallback starts, for the runner to log

    @property
    def count(self) -> int:
        """Qualifying ticks in the window."""
        return sum(self.window)

    def segment(self, policy_id: int | None, intent_ids: tuple[str, ...] = ()) -> bool:
        """Note the segment in force. True if a fallback was active and has just ended because the segment changed.

        With a per_intent trigger, a new segment that still holds an intent the formula took over keeps the formula,
        and the intents of the new segment join the takeover (the formula executes the segment, so all of it)."""
        if policy_id == self.policy_id:
            return False
        self.policy_id = policy_id
        self.window.clear()
        if self.active and self.trigger.per_intent and self.taken_over & set(intent_ids):
            self.taken_over = frozenset(intent_ids)
            return False
        ended, self.active, self.taken_over = self.active, False, frozenset()
        return ended

    def hold(self) -> None:
        """A tick where the policy doesn't decide: the window starts again (the misses must be the policy's)."""
        self.window.clear()

    def step(self, obs: np.ndarray, objective: RLObjective, records: list, loads: dict[str, float],
             rl_shares: np.ndarray, stated_slices: tuple[str, ...] | None = None,
             intent_ids: tuple[str, ...] = ()) -> np.ndarray | None:
        """The formula's split if the formula executes this tick, else None (the policy's split stands)."""
        self.fired = None
        formula_shares = np.asarray(self.formula.shares(obs), dtype=np.float64)  # in shadow, every tick
        if self.active:
            return formula_shares
        need = self.formula.last_need
        missed = missed_slices(records, objective, loads, self.trigger, stated_slices)
        qualifying = [(s, why) for s, why in missed.items()
                      if formula_shares[SLICES.index(s)] >= need[SLICES.index(s)] * (1.0 - self.trigger.need_margin)
                      - NEED_TOLERANCE
                      and formula_shares[SLICES.index(s)] >= rl_shares[SLICES.index(s)] + self.min_gain]
        self.window.append(bool(qualifying))
        if not qualifying or self.count < self.missed_ticks:
            return None
        s, why = qualifying[0]
        i = SLICES.index(s)
        self.active, self.taken_over = True, frozenset(intent_ids)
        self.fired = Fallback(s, why, self.count * self.dt_s, float(rl_shares[i]), float(formula_shares[i]))
        return formula_shares
