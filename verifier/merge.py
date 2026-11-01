"""Merge of the active intents into the one objective the RL xApp actually follows.

Several intents can be active at once ("URLLC >= 10 Mb/s for 10 min", then "eMBB >= 30 Mb/s
for 5 min"), but the RL policy reads a single objective. merge() builds it:

- per slice, the highest minimum throughput asked by any active intent;
- the tightest URLLC latency bound (no bound counts as "no constraint");
- the weights of the newest active intent (default weights if none is active).

The verifier checks merge(active + new), not the new objective alone, so what is verified is
exactly what runs (docs/PLAN.md, conflict rule). Pure and deterministic: no LLM, no randomness,
no network, no clock read; the time is an argument.
"""

from pydantic import BaseModel, ConfigDict, Field

from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues


class ActiveIntent(BaseModel):
    """An intent whose objective was accepted, as the verifier needs to know it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_id: str
    issued_at: float = Field(ge=0.0)  # backend clock, s; decides which intent is "newest"
    objective: RLObjective


class MergedObjective(BaseModel):
    """The objective the RL xApp follows, and the intents it comes from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: RLObjective
    intent_ids: tuple[str, ...]  # sorted; empty when nothing is active (default objective)


def is_active(intent: ActiveIntent, at_s: float) -> bool:
    """True if the intent's window contains at_s: valid_from <= at_s < valid_until.

    The end is excluded, so two windows that touch (one ends when the next starts) never
    overlap.
    """
    obj = intent.objective
    ends_after = obj.valid_until is None or at_s < obj.valid_until
    return obj.valid_from <= at_s and ends_after


def newest(intents: list[ActiveIntent]) -> ActiveIntent:
    """The most recently issued intent. Ties on issued_at are broken by intent_id, so the
    answer never depends on the order of the list."""
    return max(intents, key=lambda i: (i.issued_at, i.intent_id))


def merge(intents: list[ActiveIntent] | tuple[ActiveIntent, ...], at_s: float) -> MergedObjective:
    """The objective that applies at time at_s, given every known intent.

    1. keep the intents active at at_s;
    2. per slice, the highest minimum target; the tightest latency bound;
    3. the weights of the newest active intent (default weights if none);
    4. the merged window runs from at_s to the earliest end among the active intents, so the
       next expiry is also the moment to merge again.
    """
    active = [i for i in intents if is_active(i, at_s)]
    if not active:
        default = RLObjective(weights=DEFAULT_OBJECTIVE.weights, valid_from=at_s)
        return MergedObjective(objective=default, intent_ids=())

    targets = {s: max(getattr(i.objective.min_thr_mbps, s) for i in active) for s in SLICES}
    bounds = [i.objective.max_latency_ms for i in active if i.objective.max_latency_ms is not None]
    ends = [i.objective.valid_until for i in active if i.objective.valid_until is not None]

    objective = RLObjective(
        weights=newest(active).objective.weights,
        min_thr_mbps=SliceValues(**targets),
        max_latency_ms=min(bounds) if bounds else None,
        valid_from=at_s,
        valid_until=min(ends) if ends else None,
    )
    return MergedObjective(objective=objective, intent_ids=tuple(sorted(i.intent_id for i in active)))
