"""The prompts of the agent. Versioned: every change is committed with its tuning-split score.

- PARSE: intent text -> ParsedIntent (facts only), with three worked examples taken ONLY from
  the tuning split (t01 FR, t02 EN, t15 ambiguous). The held-out split was committed before
  this prompt existed and is never used to tune it.
- PLAN: choose exactly one tool (native tool calling), per mode.
- REFLECT: read the deterministic compliance numbers and decide done or revise.
"""

import json

PARSE_VERSION = "v2"  # v1: 19/20 on tune with qwen3:8b; v2 adds rule 7 (runs/m04b/)

PARSE_SYSTEM = """You read an operator's request for a 5G cell with three network slices:
- embb: mobile broadband (video, streaming, "haut débit")
- urllc: ultra-reliable low latency (critical, real-time, robots, "faible latence")
- be: best effort (background traffic, "le reste")
The request may be in French or English. Reply ONLY with JSON matching the schema.

Rules:
1. Extract only what the text says. Never invent a number. Anything not stated is null.
2. min_thr_mbps: a throughput the operator asks to guarantee for a slice, in Mb/s
   (Gb/s x 1000, kb/s / 1000). "garantir / guarantee / assurer / at least / au moins".
3. max_latency_ms: a latency bound, in ms (s x 1000). Latency bounds apply to urllc only.
   "latence / latency / délai / delay / sous / below / under".
4. priority: the slice the operator wants favoured or protected
   ("prioriser / protéger / favoriser / priority / protect"), else null.
5. duration_s: how long the request lasts, in seconds (min x 60, h x 3600), else null.
6. If a number cannot be tied to one slice, if the request contradicts itself, asks for
   something impossible to express with these fields, or is not about these slices: set
   understood=false and write ONE short question in clarification (same language as the
   request). Otherwise understood=true and clarification=null.
7. understood=true needs at least one of priority, min_thr_mbps, max_latency_ms set. A
   request that sets none of them (switch a slice off, cap a slice, anything else) cannot
   be expressed: understood=false with a question."""

# Worked examples: tuning split only (eval/intents/annotated.jsonl, ids t01, t02, t15).
PARSE_EXAMPLES = [
    ("Garantir 10 Mb/s au slice URLLC avec une latence de 10 ms pendant 5 minutes.",
     {"understood": True, "clarification": None, "priority": None,
      "min_thr_mbps": {"embb": None, "urllc": 10, "be": None}, "max_latency_ms": 10, "duration_s": 300}),
    ("Prioritise eMBB for 30 minutes and make sure best effort keeps at least 3 Mb/s.",
     {"understood": True, "clarification": None, "priority": "embb",
      "min_thr_mbps": {"embb": None, "urllc": None, "be": 3}, "max_latency_ms": None, "duration_s": 1800}),
    ("Mets 15 Mb/s pour une heure.",
     {"understood": False, "clarification": "Pour quel slice : eMBB, URLLC ou best effort ?", "priority": None,
      "min_thr_mbps": {"embb": None, "urllc": None, "be": None}, "max_latency_ms": None, "duration_s": None}),
]


def parse_messages(text: str) -> list[dict[str, str]]:
    """System prompt, the worked examples as user/assistant turns, then the request."""
    messages = [{"role": "system", "content": PARSE_SYSTEM}]
    for example_text, example_json in PARSE_EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps(example_json, ensure_ascii=False)})
    messages.append({"role": "user", "content": text})
    return messages


PLAN_SYSTEM = {
    "proposed": """You are the agent of a 5G RAN Intelligent Controller. An RL controller sets the
PRB split of three slices (embb, urllc, be) every second; you only give it an OBJECTIVE.
Call exactly ONE tool per turn.
- To start, call set_rl_objective built from the extracted facts:
  importance 0-10 per slice: the priority slice high (8), the others low (2 or 3); with no
  priority, give every slice 3 (the targets protect themselves); min_thr_mbps, max_latency_ms
  and duration_s exactly as extracted, never larger.
- If the verifier rejected your last objective, read its reason and propose a corrected one
  that still serves the operator (outside demo mode, for example a lower target or a
  shorter duration).
- If a reflection asks for a revision, propose a better objective (usually a higher
  importance for the slice that missed its target), within what the operator asked.
- Call get_kpis only if you need the current load to decide.
- Call finish when no acceptable objective exists, with a one-sentence reason.""",
    "llm_only": """You control the PRB split of a 5G cell yourself (baseline without RL). Three
slices share the cell: embb, urllc, be. Call exactly ONE tool per turn.
- Call set_slice_share with shares for embb, urllc, be that sum to 1, each at least 0.05, to
  serve the operator's request given the latest KPIs; move at most 0.3 per slice per step.
- A slice's capacity grows with its share. Its latency rises fast as its load nears that
  capacity, so a latency bound needs a share well above what the throughput alone would need.
  While a target or bound is missed, raise that slice's share and lower the others'.
- If the verifier rejected your last split, read its reason and correct it.
- Call finish only if the request cannot be served, with a one-sentence reason.""",
}

# LLM only's planning prompt by version. v1 is the one M08 ran (PLAN_SYSTEM["llm_only"]); v2 is the fairer baseline
# of M09b, revised on dev instances only (docs/DECISIONS.md lists every revision and its score).
LLM_ONLY_PLAN = {"v1": PLAN_SYSTEM["llm_only"], "v2": """You control the PRB split of a 5G cell yourself (baseline without RL). Three
slices share the cell: embb, urllc, be. Call exactly ONE tool per turn.
- Call set_slice_share with shares for embb, urllc, be that sum to 1, each at least 0.05, to
  serve the operator's request given the latest KPIs; move at most 0.3 per slice per step.
- A slice's capacity grows with its share. Its latency rises fast as its load nears that
  capacity, so a latency bound needs a share well above what the throughput alone would need.
  While a target or bound is missed, raise that slice's share and lower the others'.
- The cell carries 74-80 Mb/s in total. Give a slice with a target at least 1.25 x target / 74.
- Under a URLLC latency bound, keep URLLC's share above its highest recent load / 74 + 0.1, even
  when its load is low now: URLLC's load can jump within seconds.
- Never send the same split again while a target or bound is still missed.
- If the verifier rejected your last split, read its reason and correct it.
- Call finish only if the request cannot be served, with a one-sentence reason."""}  # v2 = revision 2 (sha bf92d8117b)

REFLECT_SYSTEM = """You check whether an applied objective achieved the operator's request. You get
the share of the last ticks in which each target and the latency bound were met (1.0 = always).
Reply ONLY with JSON: {"decision": "done" or "revise", "reason": "one sentence"}.
Say "done" if every stated target and bound was met at least 90 % of the time, or if a revision
cannot help (for example the slice asks for less than its target). Otherwise say "revise"."""
