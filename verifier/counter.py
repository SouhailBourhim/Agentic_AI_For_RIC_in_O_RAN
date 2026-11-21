"""M13: answer a rejection with the largest version of the proposal that fits (an offer).

When the verifier rejects a proposal because its targets don't fit (budget, floor, training range, conflict with
the intents in force, or too big a step), the operator learns that it doesn't fit, not what would. with_offer looks
for the largest scaled-down version that verify() accepts:

    for p = 99 %, 98 %, ... 1 %: every stated target x p, rounded down to whole Mb/s (operators state whole numbers);
    the bound, the window and the weights stay the proposal's; every stated slice keeps at least 1 Mb/s;
    the first candidate verify() accepts twice, in a cell OFFER_MARGIN smaller and in the real one, is the offer.

Why the smaller cell: the verifier sizes against the told capacity (79 Mb/s), but the testbed measured 73.5-80.2 Mb/s
depending on the split (M08d). An offer that fills the told cell exactly would squeeze the guarantees already in force
whenever the real cell is smaller; one that fits 73.5 Mb/s does not.

Why the bound is never loosened: it is the qualitative part of a URLLC request; trading it for throughput changes
the service, which is the operator's call, not the verifier's.

No offer for schema, window or latency rejections: scaling the targets down does not fix those.
An offer is information. Nothing here applies anything: to run, it is proposed and verified again like any objective.

Plain Python, like verify.py: no LLM, no randomness, no network, no clock (tests/test_verifier_purity.py).
"""

import math

from verifier.objective import SLICES, RLObjective, SliceValues
from verifier.verify import DEFAULT_CONFIG, Decision, Offer, VerifierConfig, VerifierState, verify

OFFER_CHECKS = ("budget", "floor", "training_range", "conflict", "step")
LOWEST_MEASURED_CAPACITY_MBPS = 73.5  # M08d, the testbed cell's capacity under its least favourable split
TOLD_CAPACITY_MBPS = 79.0  # what the verifier is told on the testbed and the simulator profiles from M09b on
OFFER_MARGIN = 1.0 - LOWEST_MEASURED_CAPACITY_MBPS / TOLD_CAPACITY_MBPS  # ~0.07


def scaled(obj: RLObjective, percent: int) -> RLObjective | None:
    """The proposal with every stated target x percent / 100, rounded down; None if a stated slice would get 0."""
    targets = {}
    for s in SLICES:
        stated = getattr(obj.min_thr_mbps, s)
        if stated > 0:
            targets[s] = float(math.floor(stated * percent / 100))
            if targets[s] < 1.0:
                return None
    return obj.model_copy(update={"min_thr_mbps": SliceValues(**targets)})


def with_offer(decision: Decision, obj: RLObjective, state: VerifierState, config: VerifierConfig = DEFAULT_CONFIG,
               new_intent_id: str = "new", replacing: str | None = None) -> Decision:
    """The rejection, with the largest scaled-down proposal that fits as its offer (unchanged if there is none)."""
    if decision.accepted or decision.check not in OFFER_CHECKS or not any(obj.min_thr_mbps.as_list()):
        return decision
    smaller = state.model_copy(update={"capacity_mbps": state.capacity_mbps * (1.0 - OFFER_MARGIN)})
    previous = None
    for percent in range(99, 0, -1):
        candidate = scaled(obj, percent)
        if candidate is None:  # a stated slice would lose its target: stop, smaller ones lose it too
            return decision
        if candidate == previous:  # rounding down gave the same targets as the last percent
            continue
        previous = candidate
        if (verify(candidate, smaller, config, new_intent_id, replacing).accepted
                and verify(candidate, state, config, new_intent_id, replacing).accepted):
            offer = Offer(min_thr_mbps=candidate.min_thr_mbps, max_latency_ms=obj.max_latency_ms,
                          fraction=percent / 100)
            return decision.model_copy(update={"offer": offer})
    return decision


def offer_text(offer: Offer) -> str:
    """One line for the operator, e.g. 'what fits now: urllc 16 Mb/s with latency below 5 ms (about 70 % of the
    request), for the same duration'."""
    targets = " and ".join(f"{s} {getattr(offer.min_thr_mbps, s):g} Mb/s" for s in SLICES
                           if getattr(offer.min_thr_mbps, s) > 0)
    bound = f" with latency below {offer.max_latency_ms:g} ms" if offer.max_latency_ms is not None else ""
    return f"what fits now: {targets}{bound} (about {offer.fraction:.0%} of the request), for the same duration"
