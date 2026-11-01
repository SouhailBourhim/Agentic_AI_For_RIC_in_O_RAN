"""What the agent extracts from an intent (facts) and how a decision becomes an RL objective.

ParsedIntent is the parse step's output: only what the operator said, nothing decided. Its
JSON schema is given to the LLM (Ollama's `format`), so the model's reply always has this
shape; M08b scores it against the hand-annotated intents (eval/intents/).

objective_from_decision turns the plan step's decision (an importance from 0 to 10 per
slice, plus constraints) into an RLObjective: integer importances are easy for a small LLM,
weights summing exactly to 1 are not. The verifier then checks the result like any other.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

SliceName = Literal["embb", "urllc", "be"]
DEFAULT_DURATION_S = 600.0  # an intent with no stated duration lasts 10 minutes


class SliceNumbers(BaseModel):
    """One optional number per slice (null = not stated)."""

    model_config = ConfigDict(extra="forbid")

    embb: float | None = None
    urllc: float | None = None
    be: float | None = None


class ParsedIntent(BaseModel):
    """The facts of an operator intent. Not stated = null. See eval/intents/README.md."""

    model_config = ConfigDict(extra="forbid")

    understood: bool
    clarification: str | None = None  # one question, when understood is false
    priority: SliceName | None = None  # the slice explicitly favoured or protected
    min_thr_mbps: SliceNumbers = Field(default_factory=SliceNumbers)
    max_latency_ms: float | None = None  # URLLC only
    duration_s: float | None = None


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
