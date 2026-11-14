"""Watch M09f live: each finished item's outcome, who stopped it, and the safety oracle's verdict.

Run with:  uv run marimo run notebooks/m09f_live.py      (read-only; refreshes every 5 s)

Reads the event logs under eval/results/m09f/<split>/<controller>/r<N>/events.jsonl and judges each finished item
with eval.m09f's own functions (outcome, proposals, unsafe), the same code --score uses. The item still playing has
no item_end yet and is shown apart. Nothing is written anywhere: this is a viewer.
For one item's path through the agent's nodes: RIC_EVENTS=<that events.jsonl> uv run marimo run notebooks/agent_live.py
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import collections
    import json
    import sys

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir().parent))  # the repo root, as the other notebooks do
    from eval import m09f as F
    from eval.interface_metrics import split_items
    from store.events import read_events

    repo = mo.notebook_dir().parent
    root = repo / F.RESULTS

    def load_intents(split: str) -> list[dict]:
        """eval.m09f.load_intents, from the repo root whatever folder marimo was started in."""
        with (repo / F.INTENTS).open(encoding="utf-8") as f:
            return [row for row in map(json.loads, f) if row["split"] == split]
    return F, collections, load_intents, mo, read_events, root, split_items


@app.cell
def _(mo):
    split = mo.ui.dropdown(["test", "dev"], value="test", label="split")
    refresh = mo.ui.refresh(options=["5s", "15s", "60s"], default_interval="5s")
    mo.hstack([split, refresh], justify="start")
    return refresh, split


@app.cell
def _(F, load_intents, mo, read_events, refresh, root, split, split_items):
    refresh  # re-run on every tick

    logs = sorted((root / split.value).glob("*/r*/events.jsonl"))
    mo.stop(not logs, mo.md(f"**No run under `{root / split.value}` yet.**"))
    meta = {r["id"]: r for r in load_intents(split.value)}
    rows, playing = [], []
    for log in logs:
        controller, repeat = log.parent.parent.name, log.parent.name
        finished = (log.parent / "run.json").exists()
        for item in split_items(read_events(log)):
            info = meta.get(item["start"]["id"], {"id": item["start"]["id"], "category": "?"})
            if item["end"] is None:
                playing.append({"controller": controller, "repeat": repeat, "id": info["id"], "category": info["category"],
                                "text": info.get("text", "")})
                continue
            props = F.proposals(item)
            why = F.unsafe(item)
            rows.append({"controller": controller, "repeat": repeat, "id": info["id"], "category": str(info["category"]),
                         **F.outcome(item), "dangerous proposed": sum(p["dangerous"] for p in props),
                         "dangerous blocked": sum(p["dangerous"] and not p["accepted"] for p in props),
                         "UNSAFE": "YES: " + "; ".join(why) if why else "no", "text": info.get("text", "")[:90]})
        if not finished and not playing:
            playing.append({"controller": controller, "repeat": repeat, "id": "(between items)", "category": "", "text": ""})
    return playing, rows


@app.cell
def _(F, load_intents, mo, playing, rows, split):
    expected = len(load_intents(split.value)) * sum(n for _, _, n in F.CONTROLLERS.values())
    unsafe_n = sum(r["UNSAFE"] != "no" for r in rows)
    now = playing[-1] if playing else None
    mo.vstack([
        mo.md(f"### {len(rows)} of {expected} items done · unsafe applied: "
              f"**{unsafe_n}** {'(pass so far)' if unsafe_n == 0 else '(FAIL)'}"),
        mo.md(f"**Playing:** {now['controller']} {now['repeat']} · {now['id']} (category {now['category']}) — {now['text']}"
              if now else "**Nothing playing** (finished, or between runs)"),
    ])
    return


@app.cell
def _(collections, mo, rows):
    mo.stop(not rows)
    cats = sorted({r["category"] for r in rows}, key=lambda c: (not c.isdigit(), c.zfill(2)))
    stoppers = sorted({r["stopper"] for r in rows})
    counts = collections.Counter((r["category"], r["stopper"]) for r in rows)
    pivot = [{"category": c, **{s: counts[(c, s)] or "" for s in stoppers}} for c in cats]
    per_controller = collections.Counter((r["controller"], r["result"]) for r in rows)
    results = sorted({r["result"] for r in rows})
    by_controller = [{"controller": c, **{x: per_controller[(c, x)] or "" for x in results},
                      "dangerous proposed": sum(r["dangerous proposed"] for r in rows if r["controller"] == c),
                      "dangerous blocked": sum(r["dangerous blocked"] for r in rows if r["controller"] == c)}
                     for c in dict.fromkeys(r["controller"] for r in rows)]
    mo.hstack([
        mo.vstack([mo.md("**Who stopped it, per category** (none = applied)"), mo.ui.table(pivot, selection=None, page_size=12)]),
        mo.vstack([mo.md("**Outcome per controller**"), mo.ui.table(by_controller, selection=None)]),
    ], align="start", gap=2)
    return


@app.cell
def _(mo, rows):
    mo.stop(not rows)
    mo.vstack([mo.md("**Every finished item** (newest last)"),
               mo.ui.table(rows, selection=None, page_size=15)])
    return


if __name__ == "__main__":
    app.run()
