"""Non-learning controllers to compare the RL policy against.

Both have the same shape as the policy (`observation -> action`), so the evaluation and
the plots treat all controllers the same way.
- equal_split: 1/3 of the PRBs to each slice, whatever happens (asked for by M02).
- proportional_to_load: each slice gets a share equal to its share of the offered load.
  A simple heuristic with no learning: checks that RL beats something smarter than equal.
"""

from collections import deque

import numpy as np

from sim.cell import smooth_split
from sim.env import EFFICIENCY_FEATURES, KPI_FEATURES_PER_SLICE, LATENCY_SCALE_MS, OBS_SIZE_WITH_APPLIED
from verifier.objective import MIN_SLICE_SHARE, SLICES

N_SLICES = len(SLICES)
PEAK_WINDOW_S = 30  # LLM + formula sizes a bounded URLLC to its peak load over the last 30 s (DECISIONS)
SIZING_FRACTION = 74.0 / 79.0  # LLM + formula sizes against 74 Mb/s of a cell told 79, as LLM only v2's prompt (DECISIONS)
# M10b: formula_est's rate estimate, tuned on the even dev indices within a logged budget of 3 revisions
# (docs/learn/data/M10b-formula-revisions.csv). Rev 1: the average, margin 1 - 74/79; rev 2: the min of latest and
# average, margin 0.15 (a bound is all or nothing, so under-sizing costs more than over-sizing); rev 3: the same
# estimate, margin 0.10 + 0.5 x the slice's relative spread (fast channels get more headroom, slow ones less).
# FROZEN at rev 2, the best of the three on the even half (0.738 against 0.694 and 0.728); budget spent.
ESTIMATES = ("latest", "ewma", "min")
FORMULA_EST_ESTIMATE = "min"
FORMULA_EST_MARGIN = 0.15
FORMULA_EST_VOLATILITY = 0.0  # extra margin per unit of |latest - average| / average
MIN_RATE = 1e-3  # a slice's estimated rate never reaches 0 (the tracker holds its last value while idle anyway)


def estimated_rates(obs: np.ndarray, estimate: str = FORMULA_EST_ESTIMATE, margin: float = FORMULA_EST_MARGIN,
                    volatility: float = FORMULA_EST_VOLATILITY) -> np.ndarray:
    """What one unit of share carries for each slice, as a fraction of the told capacity, from the observation's
    EfficiencyTracker features (sim/env.py, M10a), less a safety margin. formula_est and mpc_est both size with it.

    estimate: "latest" (the newest measurement, 2-3 s old), "ewma" (the average) or "min" (the more cautious of the
    two). The margin grows by `volatility` x the relative gap between latest and average: a slice whose channel
    moves fast gets more headroom."""
    if obs.shape[-1] < OBS_SIZE_WITH_APPLIED + EFFICIENCY_FEATURES:
        raise ValueError("estimated_rates needs the 34-value observation with the efficiency features (rl.runner)")
    latest = obs[OBS_SIZE_WITH_APPLIED:OBS_SIZE_WITH_APPLIED + N_SLICES].astype(np.float64)
    average = obs[OBS_SIZE_WITH_APPLIED + N_SLICES:OBS_SIZE_WITH_APPLIED + 2 * N_SLICES].astype(np.float64)
    value = {"latest": latest, "ewma": average, "min": np.minimum(latest, average)}[estimate]
    spread = np.abs(latest - average) / np.maximum(average, MIN_RATE)
    return np.maximum(value * (1.0 - margin - volatility * spread), MIN_RATE)


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
       (M08d, M09a), and a share sized to the load exactly misses "met" on a smaller cell (`rates`).
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

    def rates(self, obs: np.ndarray) -> np.ndarray:
        """What one unit of share carries per slice, as a fraction of the told capacity: here the fixed 74/79 for
        every slice (step 3). formula_est (M10b) overrides it with a measured estimate."""
        return np.full(N_SLICES, SIZING_FRACTION)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        k = KPI_FEATURES_PER_SLICE * N_SLICES
        loads = obs[0:k:KPI_FEATURES_PER_SLICE].astype(np.float64)
        targets = obs[k + N_SLICES:k + 2 * N_SLICES].astype(np.float64)
        bound_ms = float(obs[k + 2 * N_SLICES]) * LATENCY_SCALE_MS
        rate = self.rates(obs)
        need = np.minimum(targets, loads) / rate
        urllc = SLICES.index("urllc")
        self.recent_urllc.append(float(loads[urllc]))
        sized_urllc = None  # URLLC's sized need, kept when step 4 asks for all the slack on top of it
        if bound_ms > 0:
            peak = max(self.recent_urllc) if PEAK_WINDOW_S > 0 else float(loads[urllc])
            need[urllc] = peak / max(1.0 - self.base_ms / bound_ms, 1e-3) / rate[urllc]
            latency_ms = float(obs[urllc * KPI_FEATURES_PER_SLICE + 2]) * LATENCY_SCALE_MS
            if latency_ms > bound_ms:
                sized_urllc = need[urllc]
                need[urllc] = max(need[urllc], 1.0 - (need.sum() - need[urllc]))  # all the slack: drain the queue
        need = np.minimum(need, 1.0)
        # M11b: what each slice needs to meet its target, for the runtime monitor (rl/monitor.py). Without step 4's
        # drain: all the slack is a request, not a need, and the split can never grant it in full (the floor).
        self.last_need = need.copy()
        if sized_urllc is not None:
            self.last_need[urllc] = min(sized_urllc, 1.0)
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


class FormulaEstPolicy(FormulaPolicy):
    """formula_est (M10b): rev 4's rules, but each slice sized on what its PRBs were measured to carry
    (estimated_rates), instead of the fixed 74/79. Reads only the observation, like rev 4, so it gets exactly the
    information an RL policy with the efficiency inputs gets. On a cell where every PRB carries the same rate the
    estimate is the true-to-told capacity ratio, so it also adapts to the 73.5-80.2 Mb/s spread rev 4 sized for."""

    def __init__(self, urllc_base_latency_ms: float, estimate: str = FORMULA_EST_ESTIMATE,
                 margin: float = FORMULA_EST_MARGIN, volatility: float = FORMULA_EST_VOLATILITY) -> None:
        super().__init__(urllc_base_latency_ms)
        if estimate not in ESTIMATES:
            raise ValueError(f"estimate must be one of {ESTIMATES}, got {estimate!r}")
        self.estimate, self.margin, self.volatility = estimate, margin, volatility

    def rates(self, obs: np.ndarray) -> np.ndarray:
        return estimated_rates(obs, self.estimate, self.margin, self.volatility)


class SmoothedPolicy:
    """Any `.shares()` controller behind the same output filter as the served RL policy (M09g, diagnostic D1).

    The formula acts unfiltered, while LLM + RL is served behind alpha 0.10 (rl.runner.SERVED_SMOOTHING_ALPHA):
    a split it asks for takes ~10 s to arrive. Wrapping the formula in the same filter tells how much of its
    lead on dev comes from acting at once rather than from what it asks for. Same function as SlicePolicy
    (sim.cell.smooth_split), same place (inside the policy, before the runner's per-slice floor)."""

    def __init__(self, inner, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.inner, self.alpha = inner, alpha
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def shares(self, obs: np.ndarray) -> np.ndarray:
        self._previous = smooth_split(self._previous, self.inner.shares(obs), self.alpha)
        return self._previous.astype(np.float32)


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
