"""The RL objective: what the LLM may change, and what the RL policy follows.

An objective says what matters right now: a weight per slice, an optional minimum
throughput per slice, an optional latency bound for URLLC, and a validity window.
The LLM proposes one, the verifier checks it, the RL policy reads it as part of its
input (goal-conditioned policy, see CLAUDE.md).

This module only checks the *shape* of an objective (types, ranges, weights summing
to 1). Whether it is feasible on the current cell is the verifier's job (M03).
Pure data: no LLM, no randomness, no network.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Order matters: observation vectors, actions and shares always use this order.
SLICES: tuple[str, ...] = ("embb", "urllc", "be")

WEIGHT_SUM_TOLERANCE = 1e-6

# The range of objectives the RL policy was trained on (sampled in sim/env.py). The verifier
# rejects objectives outside it (M03): a goal-conditioned policy is only trustworthy on goals
# like the ones it learned from. Changing these values means retraining the policy.
TRAINED_MAX_TARGET_FRACTION = 0.4  # each minimum target is at most 40 % of the cell capacity
TRAINED_LATENCY_BOUND_MS = (5.0, 50.0)  # URLLC latency bound, when there is one


class SliceValues(BaseModel):
    """One non-negative number per slice (a weight, or a throughput in Mb/s)."""

    # allow_inf_nan=False: an infinite target would pass "ge=0" and break every sum later.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    embb: float = Field(default=0.0, ge=0.0)
    urllc: float = Field(default=0.0, ge=0.0)
    be: float = Field(default=0.0, ge=0.0)

    def as_list(self) -> list[float]:
        """Values in SLICES order, for building vectors."""
        return [self.embb, self.urllc, self.be]


class RLObjective(BaseModel):
    """An objective for the RL xApp.

    Frozen (immutable) so that an objective that has been verified cannot be
    modified afterwards; a change means building a new object and verifying it again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    weights: SliceValues
    min_thr_mbps: SliceValues = SliceValues()
    max_latency_ms: float | None = Field(default=None, gt=0.0)  # URLLC bound
    valid_from: float = Field(default=0.0, ge=0.0)  # backend clock, seconds
    valid_until: float | None = None  # None = no end

    @model_validator(mode="after")
    def _check_weights_and_window(self) -> "RLObjective":
        """Weights are shares of importance, so each is at most 1 and they sum to 1."""
        weights = self.weights.as_list()
        if any(w > 1.0 for w in weights):
            raise ValueError(f"each weight must be in [0, 1], got {weights}")
        if abs(sum(weights) - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1, got sum={sum(weights):.6f}")
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError(
                f"valid_until ({self.valid_until}) must be after valid_from ({self.valid_from})"
            )
        return self


# Used when no objective is active: a mild preference for eMBB, no hard targets.
DEFAULT_OBJECTIVE = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3))
