"""The measures of the interface evaluation, from the logs only (M08b). Pure functions, no radio KPI:
this evaluation stops at the verifier's decision, as the proposal asks, so its results measure the
interface and can't be confused with how well the RL controls the cell.

Each run's event log is cut into items by `item_start` and `item_end` markers. File order is used
rather than timestamps: the gateway, the agent and the harness are separate processes on one log.
"""

import math
from pathlib import Path

from agent.schema import stated_changes
from store.events import read_events
from verifier.objective import SLICES

GATEWAY_TOOLS = ("submit_intent", "get_kpis", "get_status")
ERRORED = "errored"
REFUSED = "refused_by_gateway"
REJECTED = "rejected_by_verifier"
SUCCEEDED = "succeeded"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval of a share. Chosen over the plain +/- formula because it stays honest at
    the edges: 20/20 gives 0.84-1.00, not a meaningless 1.00 +/- 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- extraction


def stated(facts: dict) -> dict:
    """The proposal's four fields, from a ParsedIntent: which slices, which metrics, which values, which window."""
    targets = {s: facts["min_thr_mbps"][s] for s in SLICES if facts["min_thr_mbps"][s] is not None}
    latency = facts["max_latency_ms"]
    slices = set(targets) | ({facts["priority"]} if facts["priority"] else set()) | ({"urllc"} if latency is not None else set())
    metrics = ({"throughput"} if targets else set()) | ({"latency"} if latency is not None else set()) | (
        {"priority"} if facts["priority"] else set())
    return {"slice": slices, "metric": metrics, "value": (targets, latency), "window": facts["duration_s"]}


def proposal_fields(expected: dict, got: dict | None) -> dict[str, bool] | None:
    """Per proposal field, right or wrong; None for an intent meant to be refused, where no field is scored
    (annotation rule 5). A model that refuses a clear request has every field wrong."""
    if not expected["understood"]:
        return None
    if got is None or not got["understood"]:
        return {"slice": False, "metric": False, "value": False, "window": False}
    want, have = stated(expected), stated(got)
    return {field: want[field] == have[field] for field in want}


def summarise_extraction(items: list[dict]) -> dict:
    """One repeat of one parser. `items` are rows with expected, got, correct, ambiguous."""
    n = len(items)
    correct = sum(i["correct"] for i in items)
    ambiguous = [i for i in items if i["ambiguous"]]
    clear = [i for i in items if not i["ambiguous"]]
    detected = sum(not (i["got"] or {}).get("understood", False) for i in ambiguous)
    false_alarms = sum(not (i["got"] or {}).get("understood", False) for i in clear)
    fields: dict[str, list[bool]] = {f: [] for f in ("slice", "metric", "value", "window")}
    for i in clear:
        verdict = proposal_fields(i["expected"], i["got"])
        for f, ok in (verdict or {}).items():
            fields[f].append(ok)
    return {"n": n, "correct": correct, "interval": wilson(correct, n),
            "ambiguous": len(ambiguous), "ambiguous_detected": detected,
            "clear": len(clear), "clear_refused": false_alarms,
            "fields": {f: sum(v) / len(v) if v else None for f, v in fields.items()}}


# ---------------------------------------------------------------- the logs of one run


def split_items(events: list[dict]) -> list[dict]:
    """[{start, end, events}] per item: everything logged between its two markers, in file order."""
    items, current = [], None
    for e in events:
        if e["event"] == "item_start":
            current = {"start": e["data"], "events": [], "end": None, "start_wall": e["ts_wall"]}
            items.append(current)
        elif e["event"] == "item_end":
            if current is not None:
                current["end"] = e["data"]
            current = None
        elif current is not None:
            current["events"].append(e)
    return items


def gateway_calls(events: list[dict]) -> list[dict]:
    return [e for e in events if e["event"] == "tool_call" and e.get("source") == "gateway"]


def submitted_id(item: dict) -> str | None:
    """The intent this item created, from the gateway's own log. Hermes's reply is the LLM's and the
    client's business; the gateway's record is the one that says what really arrived."""
    return next((c["intent_id"] for c in gateway_calls(item["events"])
                 if c["data"]["tool"] == "submit_intent" and c["data"]["outcome"] == "ok" and c.get("intent_id")), None)


def outcome_class(end: dict) -> str:
    """What became of a submitted intent, from the store's final record, logged in item_end."""
    state = end.get("final_state")
    if state is None:
        return "no_intent"
    if state == "applied":
        return "accepted"
    if state != "rejected":
        return state  # a timeout, or a state nothing should reach here
    verifier = end.get("verifier") or {}
    if verifier and not verifier.get("accepted", True):
        return REJECTED
    if str(end.get("reason", "")).startswith("agent error"):
        return ERRORED
    return "refused_by_parse"


def tool_call_table(items: list[dict]) -> dict[str, dict[str, int]]:
    """Per tool: succeeded, refused by the gateway, rejected by the verifier, errored (proposal's table).

    A submit_intent that the gateway accepted and the verifier later rejected counts as rejected by the
    verifier. One the parse step refused ("what do you mean?") succeeded as a call: the interface worked.
    """
    table = {t: {SUCCEEDED: 0, REFUSED: 0, REJECTED: 0, ERRORED: 0} for t in GATEWAY_TOOLS}
    for item in items:
        final = outcome_class(item["end"] or {})
        for call in gateway_calls(item["events"]):
            tool, outcome = call["data"]["tool"], call["data"]["outcome"]
            if outcome == "invalid_args":
                table[tool][REFUSED] += 1
            elif outcome != "ok":
                table[tool][ERRORED] += 1
            elif tool == "submit_intent" and final == REJECTED:
                table[tool][REJECTED] += 1
            elif tool == "submit_intent" and final == ERRORED:
                table[tool][ERRORED] += 1
            else:
                table[tool][SUCCEEDED] += 1
    return table


def state_changed(item: dict) -> bool:
    """Did this item change what the RAN is told? Two independent records: a timetable written by an apply
    in the log, and the a1_policy rows before and after, read from the store by the harness."""
    wrote = any(e["event"] == "timetable_written" and str(e["data"].get("reason", "")).startswith("apply")
                for e in item["events"])
    end = item["end"] or {}
    return wrote or end.get("schedule_before") != end.get("schedule_after")


def relaxed(item: dict, expected: dict | None) -> list[str]:
    """What an accepted proposal changed in the operator's stated values. Evaluation mode allows this after
    a rejection (decision of 2026-11-02); a demo would refuse it. Without a label (a lockdown prompt or a
    probe) there are no stated values to compare with."""
    if not expected or not expected["understood"]:
        return []
    changes: list[str] = []
    for e in item["events"]:
        data = e.get("data", {})
        if e["event"] == "agent_step" and data.get("tool") == "set_rl_objective" and (data.get("verifier") or {}).get("accepted"):
            changes += [c for c in stated_changes(expected, data.get("arguments") or {}) if c not in changes]
    return changes


def fidelity(item: dict) -> tuple[str, list[str]]:
    """Hermes: exact, altered (with `punctuation` for a lost final mark, as M05b saw) or not_sent."""
    typed = item["start"]["text"]
    sent = [c["data"]["arguments"].get("text") for c in gateway_calls(item["events"])
            if c["data"]["tool"] == "submit_intent"]
    if not sent:
        return "not_sent", sent
    if sent == [typed]:
        return "exact", sent
    if len(sent) == 1 and sent[0] == typed.rstrip(".!?"):
        return "punctuation", sent
    return "altered", sent


def interface_latency(item: dict) -> tuple[float | None, float | None]:
    """(typed -> arrival at the gateway, gateway reply -> displayed), in seconds, on one clock.

    The first submit_intent's logged time is when the gateway answered; its duration gives the arrival.
    The reply's end is when Hermes's process finished. The worker's LangGraph decision is not in either:
    submit_intent returns as soon as the request is stored, as the proposal's definition requires.
    """
    end = item["end"] or {}
    calls = [c for c in gateway_calls(item["events"]) if c["data"]["tool"] == "submit_intent"]
    if not calls or end.get("turn_start") is None or end.get("turn_end") is None:
        return None, None
    answered = calls[0]["ts_wall"]
    arrived = answered - calls[0]["data"]["duration_ms"] / 1000
    return arrived - end["turn_start"], end["turn_end"] - answered


def load_items(run: Path) -> list[dict]:
    return split_items(read_events(run / "events.jsonl"))
