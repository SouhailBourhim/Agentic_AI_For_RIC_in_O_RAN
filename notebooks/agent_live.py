"""Watch the LangGraph agent live: which nodes each intent goes through, and what it did at each one.

Run with:  uv run marimo run notebooks/agent_live.py      (read-only; refreshes every 5 s)

Reads the end of the event log (var/events.jsonl, or RIC_EVENTS), where the agent writes one
`agent_step` per node (agent/graph.py, step()). The intent ids restart at i0001 in every eval run, so
the notebook only looks at the events after the last `scenario_start`: the run being played now.
Nothing is written anywhere: this is a viewer, never a client of the agent.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    from collections import Counter
    from pathlib import Path

    import marimo as mo

    root = mo.notebook_dir().parent
    events_file = Path(os.environ.get("RIC_EVENTS", root / "var" / "events.jsonl"))
    TAIL_BYTES = 4_000_000  # the log grows all night (MBs); one run's events fit well inside this
    NODES = ("parse", "plan", "act", "confirm", "observe", "reflect")
    return Counter, NODES, TAIL_BYTES, events_file, json, mo


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["2s", "5s", "15s"], default_interval="5s")
    refresh
    return (refresh,)


@app.cell
def _(TAIL_BYTES, events_file, json, mo, refresh):
    refresh  # re-run this cell on every tick

    def tail_events(path, n_bytes):
        """The last n_bytes of the log, as events; the first (probably cut) line is dropped."""
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        if size > n_bytes:
            lines = lines[1:]
        return [json.loads(line) for line in lines if line.strip()]

    mo.stop(not events_file.exists(), mo.md(f"**No event log at `{events_file}` yet.**"))
    events = tail_events(events_file, TAIL_BYTES)
    starts = [k for k, e in enumerate(events) if e["event"] == "scenario_start"]
    run_events = events[starts[-1]:] if starts else events
    run_info = run_events[0]["data"] if starts else {}
    t0 = run_info.get("t0", run_events[0]["ts_wall"] if run_events else 0.0)
    stopped = any(e["event"] == "scenario_stop" for e in run_events)
    intent_ids = sorted({e["intent_id"] for e in run_events if e.get("intent_id")})
    mo.md(f"**Run:** `{run_info.get('run', '?')}` · instance **{run_info.get('instance', '?')}** · "
          f"controller **{run_info.get('controller', '?')}** · {'finished' if stopped else 'playing'} · "
          f"{len(intent_ids)} intent(s) so far")
    return intent_ids, run_events, t0


@app.cell
def _(intent_ids, mo):
    mo.stop(not intent_ids, mo.md("No intent in this run yet (static and RL-only runs have none)."))
    chosen = mo.ui.dropdown(intent_ids, value=intent_ids[-1], label="intent")
    chosen
    return (chosen,)


@app.cell
def _(chosen, run_events):
    mine = [e for e in run_events if e.get("intent_id") == chosen.value]
    steps = [e["data"] for e in mine if e["event"] == "agent_step"]
    text = next((e["data"].get("text") for e in mine if e["event"] == "received"), None)
    return mine, steps, text


@app.cell
def _(Counter, NODES, mine, mo, steps, text):
    def path_of(steps):
        """The visited nodes in order; an act step that went through the verifier adds a verifier stop."""
        out = ["start"]
        for s in steps:
            out.append(s["node"])
            if s["node"] == "act" and isinstance(s.get("verifier"), dict):
                out.append("verifier_ok" if s["verifier"].get("accepted") else "verifier_no")
        return out

    def mermaid(steps, ended):
        visits = path_of(steps) + (["end"] if ended else [])
        # the two verifier outcomes are one diamond; the edge label says which
        merged = ["verifier" if v.startswith("verifier") else v for v in visits]
        edges = Counter(zip(merged, merged[1:]))
        outcome = Counter(v for v in visits if v.startswith("verifier"))
        seen, current = set(merged), merged[-1]
        lines = ["flowchart LR", "  start([intent received])", "  end_([done])"]
        lines += [f"  {n}[{n}]" for n in NODES]
        lines.append(f"  verifier{{verifier<br>{outcome['verifier_ok']} ok · {outcome['verifier_no']} rejected}}")
        for (a, b), k in edges.items():
            lines.append(f"  {a.replace('end', 'end_')} -- ×{k} --> {b.replace('end', 'end_')}")
        lines += ["  classDef seen fill:#cfe8ff,stroke:#1f6feb", "  classDef now fill:#ffd88a,stroke:#b07800,stroke-width:3px"]
        lines += [f"  class {n.replace('end', 'end_')} seen" for n in seen if n != current]
        lines.append(f"  class {current.replace('end', 'end_')} now")
        return "\n".join(lines)

    ended = any(e["event"] in ("agent_done", "expired", "agent_error") for e in mine)  # agent_done: applied or rejected
    mo.vstack([
        mo.md(f"> {text}" if text else "*(intent text not in the log window)*"),
        mo.mermaid(mermaid(steps, ended)),
        mo.md("Blue: visited. Orange: where the intent is now (or ended). Edge labels: how many times."),
    ])
    return


@app.cell
def _(json, mine, mo, t0):
    def summary(e):
        d = {k: v for k, v in e["data"].items() if k not in ("node", "backend_t")}
        return json.dumps(d, ensure_ascii=False)[:300]

    rows = [{"t (s)": round(e["ts_wall"] - t0, 1), "source": e["source"], "event": e["event"],
             "node": e["data"].get("node", ""), "LLM s": e["data"].get("latency_s", ""), "details": summary(e)}
            for e in mine]
    mo.ui.table(rows, page_size=25, selection=None)
    return


if __name__ == "__main__":
    app.run()
