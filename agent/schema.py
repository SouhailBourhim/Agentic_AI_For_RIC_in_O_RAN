"""What the agent extracts from an intent (facts) and how a decision becomes an RL objective.

ParsedIntent is the parse step's output: only what the operator said, nothing decided. Its
JSON schema is given to the LLM (Ollama's `format`), so the model's reply always has this
shape; M08b scores it against the hand-annotated intents (eval/intents/).

objective_from_decision turns the plan step's decision (an importance from 0 to 10 per
slice, plus constraints) into an RLObjective: integer importances are easy for a small LLM,
weights summing exactly to 1 are not. The verifier then checks the result like any other.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

SliceName = Literal["embb", "urllc", "be"]
DEFAULT_DURATION_S = 600.0  # an intent with no stated duration lasts 10 minutes


class SliceNumbers(BaseModel):
    """One number per slice, null = not stated. Every key is required: see ParsedIntent."""

    model_config = ConfigDict(extra="forbid")

    embb: float | None
    urllc: float | None
    be: float | None


class ParsedIntent(BaseModel):
    """The facts of an operator intent. Not stated = null. See eval/intents/README.md.

    Every field is required (null is allowed, absence is not): the JSON schema then lists them
    all, Ollama's constrained decoding writes them all, and a reply that drops fields is
    invalid (retried) instead of silently read as "nothing stated" (seen with granite4.2).
    """

    model_config = ConfigDict(extra="forbid")

    understood: bool
    clarification: str | None  # one question, when understood is false
    priority: SliceName | None  # the slice explicitly favoured or protected
    min_thr_mbps: SliceNumbers
    max_latency_ms: float | None  # URLLC only
    duration_s: float | None

    def states_something(self) -> bool:
        """True when a priority, a throughput target or a latency bound is set. An understood
        request with none of them (a duration alone, "switch the eMBB slice off") gives the
        agent nothing to do: it is out of scope."""
        numbers = [*self.min_thr_mbps.model_dump().values(), self.max_latency_ms]
        return self.priority is not None or any(v is not None for v in numbers)


NOTHING_TO_DO = ("Je n'ai trouvé aucun objectif exprimable (débit minimum, latence, priorité). "
                 "Que faut-il changer ? / I found no objective I can set (minimum throughput, "
                 "latency, priority). What should change?")

# Added to every "needs clarification" reply (M05a live demo: for "Fixe le partage des PRB à
# 90/5/5" the model only asked "what does PRB mean?"). Fixed text, not the LLM's: it tells the
# operator what can be asked, and it cannot change the extracted facts or their score.
SCOPE_HINT = ("Je peux fixer des objectifs (débit minimum par slice, borne de latence URLLC, priorité, "
              "durée) ; le partage des PRB est décidé par le contrôleur RL. / I can set goals (minimum "
              "throughput per slice, URLLC latency bound, priority, duration); the PRB split itself is "
              "decided by the RL controller.")


def clarify(question: str) -> ParsedIntent:
    """A not-understood parse with every fact null."""
    return ParsedIntent(understood=False, clarification=question, priority=None,
                        min_thr_mbps=SliceNumbers(embb=None, urllc=None, be=None),
                        max_latency_ms=None, duration_s=None)


def stated_changes(facts: dict, arguments: dict) -> list[str]:
    """The operator's stated values that a proposed objective changes (empty = all kept).

    Stated = not null in the parse: each minimum throughput, the latency bound, the duration.
    Used in demo mode, where the agent may choose the importances but not relax what the
    operator asked for (decision of 2026-11-02: relaxing is allowed in eval mode only).
    """
    changes = []
    proposed_thr = arguments.get("min_thr_mbps") or {}
    for s in SLICES:
        stated = facts["min_thr_mbps"][s]
        if stated is not None and not _same(stated, proposed_thr.get(s)):
            changes.append(f"{s} minimum {stated:g} Mb/s (proposed {proposed_thr.get(s)})")
    for key, unit in (("max_latency_ms", "ms"), ("duration_s", "s")):
        stated = facts[key]
        if stated is not None and not _same(stated, arguments.get(key)):
            changes.append(f"{key} {stated:g} {unit} (proposed {arguments.get(key)})")
    return changes


def _same(a: float, b: float | None) -> bool:
    return b is not None and abs(float(a) - float(b)) <= 1e-6


def weights_from_importance(importance: dict[str, float]) -> SliceValues:
    """Importance 0..10 per slice -> weights summing to 1. All zero -> the default weights.

    Values outside 0..10 are clipped, missing slices count as 0: the plan step's numbers are
    an LLM's, so they are made well-formed here, and the verifier checks the outcome.
    """
    clipped = {s: min(10.0, max(0.0, float(importance.get(s, 0.0) or 0.0))) for s in SLICES}
    total = sum(clipped.values())
    if total == 0.0:
        return DEFAULT_OBJECTIVE.weights
    weights = {s: clipped[s] / total for s in SLICES}
    weights["be"] = 1.0 - weights["embb"] - weights["urllc"]  # exact sum of 1 despite rounding
    return SliceValues(**{s: max(0.0, w) for s, w in weights.items()})


def objective_from_decision(
    importance: dict[str, float],
    min_thr_mbps: dict[str, float | None] | None,
    max_latency_ms: float | None,
    duration_s: float | None,
    now_s: float,
) -> RLObjective:
    """The RLObjective for a decision taken at backend time now_s. Raises ValueError (pydantic)
    on values the schema refuses (negative targets, inf...), which the agent reports as a
    rejection, like the verifier's."""
    targets = {s: float(v) for s, v in (min_thr_mbps or {}).items() if s in SLICES and v is not None}
    duration = DEFAULT_DURATION_S if duration_s is None else float(duration_s)
    return RLObjective(
        weights=weights_from_importance(importance),
        min_thr_mbps=SliceValues(**targets),
        max_latency_ms=max_latency_ms,
        valid_from=now_s,
        valid_until=now_s + duration,
    )
