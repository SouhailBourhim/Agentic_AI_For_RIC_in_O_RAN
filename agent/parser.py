"""Rule-based intent parser (M04a stand-in; M04b replaces it with the LLM).

Turns operator text, in French or English, into an RLObjective with a few regular expressions:
    "garantir 10 Mb/s au slice URLLC pendant 10 minutes"
    "prioritise eMBB for 30 min"
    "latence URLLC < 5 ms et 20 Mb/s pour le best effort pendant 1 h"

It extracts, it does not judge: an impossible value is passed on, and the verifier rejects it
with a reason. When it cannot find anything it understands, it says so instead of guessing.

Rules, applied to the text in lower case and without accents, split into clauses at
"," (not a decimal comma), ";", " et ", " and ":
- slice names: embb / urllc / "best effort" (or "slice be");
- "X Mb/s" in a clause = minimum throughput for the slice named in that clause (or the only
  slice named in the whole text);
- "latence/latency ... Y ms" = the URLLC latency bound;
- prioriser/prioritise/protect/favour <slice> = that slice gets the priority weights;
  otherwise the slice with a guarantee, if exactly one; otherwise the default weights;
- "pendant/for/durant N min|s|h" = duration (default 10 minutes).
Priority weights: 0.6 for the priority slice, 0.2 for each other.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

DEFAULT_DURATION_S = 600.0
PRIORITY_WEIGHT = 0.6

SLICE_PATTERNS = {
    "embb": r"\bembb\b|\bhaut debit\b|\bbroadband\b",
    "urllc": r"\burllc\b",
    "be": r"\bbest[ -]?effort\b|\b(?:slice|tranche)\s+be\b|\bbe\s+slice\b",
}
NUMBER = r"(\d+(?:[.,]\d+)?)"
THROUGHPUT = re.compile(NUMBER + r"\s*(?:mb/?s|mbit/?s|mbps)\b")
LATENCY = re.compile(r"\b(?:latence|latency|delai|delay)\b[^0-9]*" + NUMBER + r"\s*ms\b")
PRIORITY = re.compile(
    r"\b(?:prioriti[sz]e[sd]?|prioriser|prioritaire|priorite|priority|proteger|protect|favoriser|favou?r)\b"
)
DURATION = re.compile(r"\b(?:pendant|durant|for)\s+" + NUMBER + r"\s*(h|heures?|hours?|min|minutes?|s|sec|secondes?|seconds?)\b")
CLAUSE_SPLIT = re.compile(r"(?<!\d),|,(?!\d)|;|\bet\b|\band\b")  # not the decimal comma of "15,5"


@dataclass(frozen=True)
class ParseResult:
    objective: RLObjective | None  # None when the text was not understood
    parsed: dict = field(default_factory=dict)  # what was extracted, for the record and the logs
    problem: str | None = None  # why it was not understood


def normalise(text: str) -> str:
    """Lower case, accents removed ("Latence élevée" -> "latence elevee")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def number(raw: str) -> float:
    return float(raw.replace(",", "."))


def slices_in(text: str) -> list[str]:
    return [s for s in SLICES if re.search(SLICE_PATTERNS[s], text)]


def duration_s(text: str) -> float:
    match = DURATION.search(text)
    if match is None:
        return DEFAULT_DURATION_S
    value, unit = number(match.group(1)), match.group(2)
    if unit.startswith("h"):
        return value * 3600.0
    if unit.startswith("m"):
        return value * 60.0
    return value


def parse_intent(text: str, now_s: float) -> ParseResult:
    """Parse operator text into an objective valid from now_s (backend clock)."""
    norm = normalise(text)
    named = slices_in(norm)
    targets: dict[str, float] = {}
    priority: list[str] = []
    bound: float | None = None

    for clause in CLAUSE_SPLIT.split(norm):
        here = slices_in(clause)
        slice_ = here[0] if len(here) == 1 else (named[0] if len(named) == 1 else None)
        thr = THROUGHPUT.search(clause)
        if thr is not None:
            if slice_ is None:
                return ParseResult(None, {"text": norm}, f"a throughput ({thr.group(0)}) without a clear slice")
            targets[slice_] = number(thr.group(1))
        lat = LATENCY.search(clause)
        if lat is not None:
            if slice_ not in (None, "urllc"):
                return ParseResult(None, {"text": norm}, f"latency bounds exist only for URLLC, not {slice_}")
            bound = number(lat.group(1))
        if PRIORITY.search(clause) and slice_ is not None:
            priority.append(slice_)

    if not targets and bound is None and not priority:
        return ParseResult(None, {"text": norm, "slices": named},
                           "not understood: no throughput target, latency bound or priority found")

    favoured = priority[0] if priority else (next(iter(targets)) if len(targets) == 1 else None)
    if favoured is None:
        weights = DEFAULT_OBJECTIVE.weights
    else:
        other = (1.0 - PRIORITY_WEIGHT) / (len(SLICES) - 1)
        weights = SliceValues(**{s: PRIORITY_WEIGHT if s == favoured else other for s in SLICES})
    duration = duration_s(norm)
    parsed = {"slices": named, "targets_mbps": targets, "latency_bound_ms": bound,
              "priority": favoured, "duration_s": duration}
    objective = RLObjective(
        weights=weights,
        min_thr_mbps=SliceValues(**targets),
        max_latency_ms=bound,
        valid_from=now_s,
        valid_until=now_s + duration,
    )
    return ParseResult(objective, parsed)
