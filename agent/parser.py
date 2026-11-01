"""Rule-based intent parser: the regex twin of the LLM's parse step (M04a; kept in M04b).

Turns operator text, in French or English, into a ParsedIntent (agent/schema.py): the facts,
nothing decided. Used by the rule-based agent (RIC_AGENT=rules: CI, fallback without Ollama)
and as the baseline of the extraction score (eval/extraction.py).
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
- prioriser/prioritise/protect/favour <slice> = the priority slice;
- "pendant/for/durant N min|s|h" = duration (null when not stated).
"""

import re
import unicodedata

from agent.schema import ParsedIntent, SliceNumbers
from verifier.objective import SLICES


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


def normalise(text: str) -> str:
    """Lower case, accents removed ("Latence élevée" -> "latence elevee")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def number(raw: str) -> float:
    return float(raw.replace(",", "."))


def slices_in(text: str) -> list[str]:
    return [s for s in SLICES if re.search(SLICE_PATTERNS[s], text)]


def duration_s(text: str) -> float | None:
    """The stated duration in seconds, or None when none is stated."""
    match = DURATION.search(text)
    if match is None:
        return None
    value, unit = number(match.group(1)), match.group(2)
    if unit.startswith("h"):
        return value * 3600.0
    if unit.startswith("m"):
        return value * 60.0
    return value


def parse_facts(text: str) -> ParsedIntent:
    """The facts of an intent (agent/schema.py ParsedIntent), by regular expressions.

    The rule-based twin of the LLM's parse step: same output, so the same graph and the same
    scorer (eval/extraction.py) work with both. Priority is set only by an explicit priority
    word; a guarantee alone is a fact, not a priority (the plan step decides the weights).
    """
    norm = normalise(text)
    named = slices_in(norm)
    targets: dict[str, float] = {}
    priority: list[str] = []
    bound: float | None = None

    def not_understood(question: str) -> ParsedIntent:
        return ParsedIntent(understood=False, clarification=question)

    for clause in CLAUSE_SPLIT.split(norm):
        here = slices_in(clause)
        slice_ = here[0] if len(here) == 1 else (named[0] if len(named) == 1 else None)
        thr = THROUGHPUT.search(clause)
        if thr is not None:
            if slice_ is None:
                return not_understood(f"which slice should get {thr.group(0)}?")
            targets[slice_] = number(thr.group(1))
        lat = LATENCY.search(clause)
        if lat is not None:
            if slice_ not in (None, "urllc"):
                return not_understood(f"latency bounds exist only for URLLC, not {slice_}: which slice?")
            bound = number(lat.group(1))
        if PRIORITY.search(clause) and slice_ is not None:
            priority.append(slice_)

    if not targets and bound is None and not priority:
        return not_understood("no throughput target, latency bound or priority found: what should change?")
    return ParsedIntent(
        understood=True,
        priority=priority[0] if priority else None,
        min_thr_mbps=SliceNumbers(**targets),
        max_latency_ms=bound,
        duration_s=duration_s(norm),
    )
