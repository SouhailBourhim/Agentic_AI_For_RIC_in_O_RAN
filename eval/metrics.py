"""The measures of one run, computed only from its kpi.csv and events.jsonl (M08a; M08c reuses them).

Pure functions: the numbers in the note come from here and from nothing else. A controller that
ignores intents is scored against them too, so all four are compared on what the operator asked for.
"""

import csv
import json
from pathlib import Path

import numpy as np

from agent.schema import stated_changes
from eval.extraction import score_one
from eval.scenarios import Scenario, ScenarioIntent
from store.events import read_events
from verifier.objective import SLICES

HOLD_S = 5  # a target is "reached" once it has held this many consecutive seconds
THROUGHPUT_TOLERANCE = 1e-9  # as in agent.tools.compliance


def load_run(work: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    """KPI columns per slice, and the events."""
    rows = list(csv.DictReader((work / "kpi.csv").open()))
    kpis = {}
    for s in SLICES:
        mine = [r for r in rows if r["slice"] == s]
        number = lambda k: np.array([float(r[k]) if r[k] != "" else np.nan for r in mine])  # noqa: B023
        kpis[s] = {"t": number("ts"), "thr": number("dl_thr_mbps"), "load": number("offered_load_mbps"),
                   "lat": number("latency_ms"), "share": number("prb_share_alloc"), "policy": number("policy_id")}
    return kpis, read_events(work / "events.jsonl")  # a static run logs nothing: no file, no events


def met_mask(kpis: dict, intent: ScenarioIntent) -> tuple[np.ndarray, np.ndarray]:
    """(times, met?) over the whole run. Same rule as agent.tools.compliance: a throughput target is
    met when the slice got min(target, offered load), and the latency bound on URLLC, tick by tick;
    the intent is met in a second when all of its conditions are."""
    t = kpis["urllc"]["t"]
    ok = np.ones(len(t), dtype=bool)
    for s, target in intent.targets.items():
        ok &= kpis[s]["thr"] >= np.minimum(target, kpis[s]["load"]) - THROUGHPUT_TOLERANCE
    if intent.max_latency_ms is not None:
        ok &= kpis["urllc"]["lat"] <= intent.max_latency_ms
    return t, ok


def in_window(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return (t > start) & (t <= end)  # a KPI's ts is the end of its second


def reaction_s(t: np.ndarray, ok: np.ndarray, start: float, end: float) -> float | None:
    """Seconds from `start` until the intent holds for HOLD_S seconds in a row, or None if it never
    does before `end`. None stays None: a mean over "never" would be an invented number."""
    for i in np.where(in_window(t, start, end))[0]:
        if i + HOLD_S <= len(ok) and ok[i:i + HOLD_S].all() and t[i + HOLD_S - 1] <= end:
            return max(0.0, float(t[i] - start - 1.0))  # the first second after `start` counts as 0; the testbed's clock isn't whole seconds
    return None


def over_bound_after(kpis: dict, intent: ScenarioIntent, within_s: float = 60.0) -> int | None:
    """Seconds URLLC spent over the intent's latency bound just after its window (M08d). intent_met can't see
    these: M08c's S2 swung for 23-28 s after the intent expired and still scored 1.00. None if the intent has no
    bound, or if the run ended with it."""
    if intent.max_latency_ms is None:
        return None
    t, latency = kpis["urllc"]["t"], kpis["urllc"]["lat"]
    _, end = intent.window
    after = (t > end) & (t <= end + within_s)
    return int((latency[after] > intent.max_latency_ms).sum()) if after.any() else None


def moved_per_s(kpis: dict, mask: np.ndarray | None = None) -> float | None:
    """Share of the cell that changes hands per second: half the sum of |change| over the slices, as
    one PRB leaving a slice also arrives in another. Only between neighbouring seconds both in mask."""
    shares = np.stack([kpis[s]["share"] for s in SLICES])
    steps = np.abs(np.diff(shares, axis=1)).sum(axis=0) / 2
    if mask is not None:
        steps = steps[mask[:-1] & mask[1:]]
    return float(steps.mean()) if steps.size else None


def blocked_by_check(events: list[dict]) -> tuple[int, dict[str, int]]:
    """(proposals made, proposals the verifier blocked per check), from the event log."""
    proposals, blocked = 0, {}
    for e in events:
        verifier = e.get("data", {}).get("verifier") if e["event"] == "agent_step" else None
        if verifier is None and e["event"] == "reverified":
            verifier = None if e["data"].get("accepted") else e["data"]
        if verifier is None:
            continue
        if e["event"] == "agent_step":
            proposals += 1
        if not verifier["accepted"]:
            blocked[verifier["check"]] = blocked.get(verifier["check"], 0) + 1
    return proposals, blocked


def llm_latencies(events: list[dict]) -> dict[str, list[float]]:
    """The seconds of each LLM step, per node. Only LLM steps carry a latency above 0."""
    out: dict[str, list[float]] = {"parse": [], "plan": [], "reflect": []}
    for e in events:
        data = e.get("data", {})
        if e["event"] == "agent_step" and data.get("node") in out and data.get("latency_s"):
            out[data["node"]].append(float(data["latency_s"]))
    return out


def first_apply_s(events: list[dict], intent_id: str, submitted_s: float) -> float | None:
    """Simulated seconds from submission until the intent first changed something in the cell."""
    for e in events:
        if e["event"] in ("applied", "shares_applied") and e.get("intent_id") == intent_id:
            return float(e["ts_backend"]) - submitted_s
    return None


def intent_id_of(events: list[dict], intent: ScenarioIntent) -> str | None:
    """The id the store gave this intent, found by its text: numbering by order of submission would
    be wrong after a revision, which takes the next number. None when the controller ignores intents."""
    return next((e["intent_id"] for e in events if e["event"] == "received" and e["data"]["text"] == intent.text), None)


def intent_events(events: list[dict], intent_id: str | None) -> list[dict]:
    return [e for e in events if e.get("intent_id") == intent_id and e["event"] == "agent_step"]


def relaxed(steps: list[dict], intent: ScenarioIntent) -> list[str]:
    """What the accepted proposals, a revision included, changed in the operator's stated values
    (allowed in eval mode; scoring stays against what was stated)."""
    changes: list[str] = []
    for e in steps:
        data = e["data"]
        if data.get("tool") == "set_rl_objective" and (data.get("verifier") or {}).get("accepted"):
            changes += [c for c in stated_changes(intent.expected, data.get("arguments") or {}) if c not in changes]
    return changes


def parse_correct(steps: list[dict], intent: ScenarioIntent) -> bool | None:
    """H7: did the parse step state the intended facts? None when this controller has no parse."""
    for e in steps:
        if e["data"].get("node") == "parse":
            facts = e["data"].get("facts")
            return bool(facts) and score_one(intent.expected, facts)["correct"]
    return None


def measure_intent(kpis: dict, events: list[dict], scenario: Scenario, index: int) -> dict:
    """One row per run and intent."""
    intent = scenario.intents[index]
    start, end = intent.window
    t, ok = met_mask(kpis, intent)
    window = in_window(t, start, end)
    intent_id = intent_id_of(events, intent)
    steps = intent_events(events, intent_id)
    reaction = reaction_s(t, ok, intent.reaction_from_s, end)
    row = {"intent": index + 1, "intent_met": float(ok[window].mean()),
           "reaction_s": "" if reaction is None else reaction, "reached": int(reaction is not None),
           "first_apply_s": "" if (first := first_apply_s(events, intent_id, intent.at_s)) is None else first,
           "parse_ok": parse_correct(steps, intent), "relaxed": "; ".join(relaxed(steps, intent)),
           "seconds_on_fallback": int(np.isnan(kpis["urllc"]["policy"][window]).sum())}
    for s in SLICES:
        k = kpis[s]
        row[f"{s}_thr"] = float(k["thr"][window].mean())
        row[f"{s}_unmet"] = float(np.clip(k["load"][window] - k["thr"][window], 0, None).mean())
        row[f"{s}_lat_median"] = float(np.median(k["lat"][window]))
        row[f"{s}_lat_p95"] = float(np.percentile(k["lat"][window], 95))
        row[f"{s}_share"] = float(k["share"][window].mean())
    return row


def measure_run(work: Path, scenario: Scenario) -> dict:
    """The measures that belong to the whole run."""
    kpis, events = load_run(work)
    latencies = llm_latencies(events)
    proposals, blocked = blocked_by_check(events)
    all_calls = [x for values in latencies.values() for x in values]
    no_intent = np.isnan(kpis["urllc"]["policy"])
    idle = moved_per_s(kpis, no_intent)
    run = {"moved_per_s": moved_per_s(kpis), "moved_per_s_no_intent": "" if idle is None else idle,
           "proposals": proposals, "blocked": sum(blocked.values()), "blocked_checks": json.dumps(blocked, sort_keys=True),
           "agent_errors": sum(e["event"] == "agent_error" for e in events), "llm_steps": len(all_calls)}
    for node, values in latencies.items():
        run[f"llm_{node}_median_s"] = float(np.median(values)) if values else ""
        run[f"llm_{node}_p95_s"] = float(np.percentile(values, 95)) if values else ""
    return {"run": run, "intents": [measure_intent(kpis, events, scenario, i) for i in range(len(scenario.intents))]}
