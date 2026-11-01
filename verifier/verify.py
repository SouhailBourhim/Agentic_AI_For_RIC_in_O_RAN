"""The deterministic verifier: accept or reject a proposed RL objective, with a reason.

Every objective the LLM proposes goes through verify() before it can reach a1_policy (hard
invariant 3). The checks run in a fixed order and the first failure is returned, so the same
inputs always give the same decision and the same reason:

    1 schema           the proposal is a valid RLObjective (shape, ranges, weights sum to 1)
    2 window           the validity window is bounded, long enough, not over, not too far ahead
    3 latency_physics  the URLLC bound is above the latency of an empty URLLC slice
    4 budget           the targets fit in the cell's PRBs
    5 floor            ... while leaving every slice at least a minimum PRB share
    6 training_range   the objective looks like the ones the policy was trained on
    7 conflict         4 and 5 still hold for the merge with the other active intents
    8 step             the objective that runs does not jump too far at once

Plain Python: no LLM, no randomness, no network, no clock read (the time is in the state).
tests/test_verifier_purity.py checks this. Each check is explained in docs/learn/M03-verifier.md.
"""

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verifier.merge import ActiveIntent, MergedObjective, merge
from verifier.objective import (
    MIN_SLICE_SHARE,
    SLICES,
    TRAINED_LATENCY_BOUND_MS,
    TRAINED_MAX_TARGET_FRACTION,
    RLObjective,
)

# Float sums like 0.8 - 0.3 give 0.5000000000000001; a value exactly on a limit is accepted.
TOLERANCE = 1e-9

CheckName = Literal[
    "ok", "schema", "window", "latency_physics", "budget", "floor", "training_range", "conflict", "step"
]


class VerifierConfig(BaseModel):
    """The limits of the verifier. Defaults and their reasons are in docs/learn/M03-verifier.md."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_share_floor: float = Field(default=MIN_SLICE_SHARE, ge=0.0, le=1.0 / len(SLICES))  # same as the runner
    max_weight_step: float = Field(default=0.5, gt=0.0)  # per slice, between two merged objectives
    max_share_step: float = Field(default=0.3, gt=0.0)  # per slice, between two direct PRB splits (LLM only)
    max_target_step_fraction: float = Field(default=0.3, gt=0.0)  # per slice, fraction of capacity
    min_duration_s: float = Field(default=10.0, gt=0.0)  # the policy needs ~10-20 steps to act
    max_duration_s: float = Field(default=86_400.0, gt=0.0)  # one day
    max_start_delay_s: float = Field(default=3_600.0, ge=0.0)  # one hour
    max_target_fraction: float = TRAINED_MAX_TARGET_FRACTION
    latency_bound_range_ms: tuple[float, float] = TRAINED_LATENCY_BOUND_MS


class VerifierState(BaseModel):
    """What the verifier needs to know about the cell and the intents already accepted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now_s: float = Field(ge=0.0)  # backend clock
    capacity_mbps: float = Field(gt=0.0)  # cell capacity with all PRBs (75 Mb/s, M06 spike)
    urllc_base_latency_ms: float = Field(default=2.0, gt=0.0)  # empty URLLC slice (sim/cell.py)
    active: tuple[ActiveIntent, ...] = ()  # accepted intents; expired ones are ignored


class Decision(BaseModel):
    """The verifier's answer. `check` names the first failing check, or "ok"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    check: CheckName
    reason: str
    merged: MergedObjective | None = None  # what runs now if accepted; None if rejected


DEFAULT_CONFIG = VerifierConfig()


def _reject(check: CheckName, reason: str) -> Decision:
    return Decision(accepted=False, check=check, reason=reason)


# ---------------------------------------------------------------- the checks, in order


def check_schema(proposal: RLObjective | Mapping[str, Any]) -> RLObjective | Decision:
    """1. Build the RLObjective from raw input (an LLM's JSON). Pydantic does the work.

    Pydantic's lax mode is kept: "10" becomes 10.0, so a number sent as a string by the
    LLM is not a rejection. Unknown fields, inf, NaN and negative values are rejected.
    """
    if isinstance(proposal, RLObjective):
        return proposal
    try:
        return RLObjective.model_validate(proposal)
    except ValidationError as err:
        problems = []
        for e in err.errors():
            where = ".".join(str(part) for part in e["loc"])  # empty for whole-object checks
            message = e["msg"].removeprefix("Value error, ")
            problems.append(f"{where}: {message}" if where else message)
        return _reject("schema", f"invalid objective: {'; '.join(problems)}")


def check_window(obj: RLObjective, state: VerifierState, config: VerifierConfig) -> Decision | None:
    """2. The validity window must be bounded, not over, long enough, not too long or too far ahead.

    The length that counts is what is left from now: valid_until - max(valid_from, now).
    """
    if obj.valid_until is None:
        return _reject("window", "the objective has no end (valid_until); every intent must expire")
    if obj.valid_until <= state.now_s:
        return _reject("window", f"the window ended at {obj.valid_until:g} s, now is {state.now_s:g} s")
    start = max(obj.valid_from, state.now_s)
    length = obj.valid_until - start
    if length < config.min_duration_s - TOLERANCE:
        return _reject(
            "window",
            f"the objective lasts {length:g} s, less than the minimum {config.min_duration_s:g} s "
            "the policy needs to act",
        )
    if length > config.max_duration_s + TOLERANCE:
        return _reject("window", f"the objective lasts {length:g} s, more than the maximum {config.max_duration_s:g} s")
    delay = obj.valid_from - state.now_s
    if delay > config.max_start_delay_s + TOLERANCE:
        return _reject(
            "window", f"the objective starts in {delay:g} s, later than the maximum {config.max_start_delay_s:g} s"
        )
    return None


def check_latency_physics(obj: RLObjective, state: VerifierState) -> Decision | None:
    """3. A URLLC bound at or below the latency of an empty URLLC slice can never be met."""
    bound = obj.max_latency_ms
    if bound is not None and bound <= state.urllc_base_latency_ms:
        return _reject(
            "latency_physics",
            f"URLLC latency bound {bound:g} ms is not above the latency of an empty URLLC slice "
            f"({state.urllc_base_latency_ms:g} ms)",
        )
    return None


def required_shares(obj: RLObjective, state: VerifierState) -> dict[str, float]:
    """The PRB share each slice needs to meet its minimum target.

    A target of T Mb/s needs T / C of the cell (every PRB carries the same rate, sim/cell.py).
    For URLLC with a latency bound B, the cell model gives latency = base / (1 - rho), with
    rho = load / slice capacity. Latency <= B means rho <= 1 - base / B, so carrying T Mb/s
    within the bound needs a capacity of T / (1 - base / B): the share is T / (C (1 - base / B)).
    """
    need = {s: getattr(obj.min_thr_mbps, s) / state.capacity_mbps for s in SLICES}
    if obj.max_latency_ms is not None and need["urllc"] > 0:
        max_rho = 1.0 - state.urllc_base_latency_ms / obj.max_latency_ms
        need["urllc"] = need["urllc"] / max_rho if max_rho > 0 else float("inf")
    return need


def feasibility_problem(
    obj: RLObjective, state: VerifierState, config: VerifierConfig
) -> tuple[CheckName, str] | None:
    """4 and 5. Do the targets fit in the PRB budget while every slice keeps the floor?

    Budget: the shares needed must sum to at most 1.
    Floor: a slice with no target (or a small one) still gets at least min_share_floor, so the
    sum of max(need, floor) must be at most 1. Otherwise the slices below the floor would be
    left with less than the floor: they are named as starved.
    """
    need = required_shares(obj, state)
    detail = ", ".join(f"{s} {need[s]:.0%}" for s in SLICES if need[s] > 0)
    total = sum(need.values())
    if total > 1.0 + TOLERANCE:
        return "budget", f"the minimum targets need {total:.0%} of the PRBs ({detail}), more than the cell has"
    floor = config.min_share_floor
    if sum(max(n, floor) for n in need.values()) > 1.0 + TOLERANCE:
        starved = [s for s in SLICES if need[s] < floor]
        left = 1.0 - sum(need[s] for s in SLICES if need[s] >= floor)
        return (
            "floor",
            f"the minimum targets need {total:.0%} of the PRBs ({detail}), which leaves {left:.1%} "
            f"for {' and '.join(starved)}, below the floor of {floor:.0%} per slice",
        )
    return None


def check_training_range(obj: RLObjective, state: VerifierState, config: VerifierConfig) -> Decision | None:
    """6. The policy was trained on targets up to 40 % of capacity and bounds of 5-50 ms.

    Outside that range the network gets inputs it has never seen; its behaviour is unknown,
    even when the objective is physically feasible.
    """
    max_target = config.max_target_fraction * state.capacity_mbps
    for s in SLICES:
        target = getattr(obj.min_thr_mbps, s)
        if target > max_target + TOLERANCE:
            return _reject(
                "training_range",
                f"{s} target {target:g} Mb/s is above {max_target:g} Mb/s "
                f"({config.max_target_fraction:.0%} of capacity), the largest the policy was trained on",
            )
    low, high = config.latency_bound_range_ms
    bound = obj.max_latency_ms
    if bound is not None and not (low - TOLERANCE <= bound <= high + TOLERANCE):
        return _reject(
            "training_range",
            f"URLLC latency bound {bound:g} ms is outside {low:g}-{high:g} ms, the range the policy was trained on",
        )
    return None


def check_instants(new: RLObjective, state: VerifierState) -> list[float]:
    """The moments where the merged constraints can get tighter during the new window.

    Constraints only tighten when an intent starts (expiries only loosen them), so it is enough
    to check the start of the new objective and each start of an active intent inside its window.
    """
    first = max(new.valid_from, state.now_s)
    end = new.valid_until if new.valid_until is not None else float("inf")
    later = [i.objective.valid_from for i in state.active if first < i.objective.valid_from < end]
    return sorted({first, *later})


def check_merges(
    intents: list[ActiveIntent],
    instants: list[float],
    state: VerifierState,
    config: VerifierConfig,
    new_intent_id: str | None = None,
) -> Decision | None:
    """7. Budget and floor on the merged objective at each instant.

    The reason names the other intents in the merge (not the new one itself).
    """
    for t in instants:
        merged = merge(intents, t)
        problem = feasibility_problem(merged.objective, state, config)
        if problem is not None:
            _, why = problem
            others = [i for i in merged.intent_ids if i != new_intent_id]
            return _reject("conflict", f"together with active intent(s) {', '.join(others)} at t={t:g} s, {why}")
    return None


def check_step(before: RLObjective, after: RLObjective, state: VerifierState, config: VerifierConfig) -> Decision | None:
    """8. The objective that runs may not jump too far in one change (per slice weight and target)."""
    for s in SLICES:
        dw = abs(getattr(after.weights, s) - getattr(before.weights, s))
        if dw > config.max_weight_step + TOLERANCE:
            return _reject(
                "step",
                f"{s} weight would change by {dw:.2f} ({getattr(before.weights, s):.2f} -> "
                f"{getattr(after.weights, s):.2f}), more than the maximum step {config.max_weight_step:.2f}",
            )
        dt = abs(getattr(after.min_thr_mbps, s) - getattr(before.min_thr_mbps, s))
        max_dt = config.max_target_step_fraction * state.capacity_mbps
        if dt > max_dt + TOLERANCE:
            return _reject(
                "step",
                f"{s} target would change by {dt:g} Mb/s, more than the maximum step {max_dt:g} Mb/s "
                f"({config.max_target_step_fraction:.0%} of capacity)",
            )
    return None


# ---------------------------------------------------------------- entry points


def verify(
    proposal: RLObjective | Mapping[str, Any],
    state: VerifierState,
    config: VerifierConfig = DEFAULT_CONFIG,
    new_intent_id: str = "new",
    replacing: str | None = None,
) -> Decision:
    """Accept or reject a proposed objective. Checks 1-8 in order; the first failure wins.

    On acceptance, Decision.merged is merge(active + new) at state.now_s: the objective the RL
    xApp follows from now on, with the intent_ids it comes from. The new intent is treated as
    issued now, so it is the newest one and its weights are used.

    `replacing` (M04b): the id of an active intent this proposal replaces (a revision). Conflicts
    are checked WITHOUT it, since it leaves when the revision is applied, but the step is
    measured from what runs now, WITH it: a revision is a change from the objective in force.
    """
    obj = check_schema(proposal)
    if isinstance(obj, Decision):
        return obj

    # Checks 2-6 look at the new objective alone.
    decision = check_window(obj, state, config)
    if decision is not None:
        return decision
    decision = check_latency_physics(obj, state)
    if decision is not None:
        return decision
    problem = feasibility_problem(obj, state, config)
    if problem is not None:
        return _reject(*problem)
    decision = check_training_range(obj, state, config)
    if decision is not None:
        return decision

    # Checks 7-8 look at the merge with the intents already active.
    running_now = list(state.active)
    current = [i for i in running_now if i.intent_id != replacing]
    with_new = [*current, ActiveIntent(intent_id=new_intent_id, issued_at=state.now_s, objective=obj)]
    decision = check_merges(with_new, check_instants(obj, state), state, config, new_intent_id)
    if decision is not None:
        return decision

    start = max(obj.valid_from, state.now_s)
    decision = check_step(merge(running_now, start).objective, merge(with_new, start).objective, state, config)
    if decision is not None:
        return decision

    merged = merge(with_new, state.now_s)
    return Decision(accepted=True, check="ok", reason=_accept_reason(merged), merged=merged)


def reverify_after_removal(state: VerifierState, config: VerifierConfig = DEFAULT_CONFIG) -> Decision:
    """Re-check the merge after an intent expired or was cancelled (state.active without it).

    Runs budget, floor and conflict on the remaining intents, now and at each later start.
    The step check is skipped: an expiry cannot be refused. With the same cell, removing an
    intent only loosens the targets and the bound, so this accepts; it can reject only if the
    state changed meanwhile (for example a recalibrated capacity). Weights may jump back to
    an older intent's weights: the caller logs it.
    """
    later = [i.objective.valid_from for i in state.active if i.objective.valid_from > state.now_s]
    instants = sorted({state.now_s, *later})
    decision = check_merges(list(state.active), instants, state, config)
    if decision is not None:
        return decision
    merged = merge(list(state.active), state.now_s)
    return Decision(accepted=True, check="ok", reason=_accept_reason(merged), merged=merged)


def _accept_reason(merged: MergedObjective) -> str:
    if not merged.intent_ids:
        return "no intent active: default objective"
    return f"accepted; once applied, the RL xApp follows the merge of {', '.join(merged.intent_ids)}"


def verify_shares(shares: list[float], previous: list[float] | None,
                  config: VerifierConfig = DEFAULT_CONFIG) -> Decision:
    """Check a PRB split proposed directly (set_slice_share, the "LLM only" baseline of M08a).

    Only that baseline sets shares itself; in the proposed controller the RL policy does
    (hard invariant 4). Checks, in order: 3 finite numbers (schema), summing to 1 (budget),
    every slice at or above the floor (floor), no slice moving more than max_share_step from the
    split applied now (step). Same Decision as verify(); `merged` stays None.
    """
    values = list(shares)
    if len(values) != len(SLICES) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return _reject("schema", f"a PRB split is {len(SLICES)} finite numbers (embb, urllc, be), got {shares}")
    if abs(sum(values) - 1.0) > 1e-6:
        return _reject("budget", f"PRB shares must sum to 1, got {sum(values):.4f}")
    for s, v in zip(SLICES, values):
        if v < config.min_share_floor - TOLERANCE:
            return _reject("floor", f"{s} share {v:.3f} is below the floor of {config.min_share_floor:.0%} per slice")
    if previous is not None:
        for s, v, before in zip(SLICES, values, previous):
            if abs(v - before) > config.max_share_step + TOLERANCE:
                return _reject("step", f"{s} share would change by {abs(v - before):.2f} ({before:.2f} -> {v:.2f}),"
                                       f" more than the maximum step {config.max_share_step:.2f}")
    return Decision(accepted=True, check="ok", reason="PRB split accepted")
