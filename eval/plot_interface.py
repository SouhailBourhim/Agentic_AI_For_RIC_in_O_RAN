"""The tables and figures of the interface evaluation, from its logs (M08b; run eval.interface first).

  uv run python -m eval.plot_interface                  the held-out split
  uv run python -m eval.plot_interface --split tune --out DIR     the rehearsal, never in the note's figures

Writes docs/learn/data/M08b-*.csv and docs/learn/img/M08b-*.png, and prints every number the note quotes.
No throughput or radio latency appears here: this evaluation stops at the verifier.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval import interface_metrics as M
from eval.extraction import load_intents
from eval.interface import RESULTS
from eval.plot_control import ACCENT, NEUTRAL, titled
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED, _style

IMG = Path("docs/learn/img")
DATA = Path("docs/learn/data")
OUTCOME_COLOURS = {M.SUCCEEDED: NEUTRAL, M.REFUSED: "#eda100", M.REJECTED: "#6250d6", M.ERRORED: "#c0392b"}
OUTCOME_LABELS = {M.SUCCEEDED: "succeeded", M.REFUSED: "refused by the gateway", M.REJECTED: "rejected by the verifier", M.ERRORED: "errored"}
FIDELITY_MARKS = {"exact": ("o", ACCENT), "punctuation": ("s", "#eda100"), "altered": ("D", "#e87ba4"), "not_sent": ("x", "#c0392b")}


def extraction_runs(split: str) -> dict[str, list[dict]]:
    """{parser: [one result per repeat]}."""
    runs: dict[str, list[dict]] = {}
    for path in sorted((RESULTS / split / "extraction").glob("*_r*.json")):
        result = json.loads(path.read_text())
        runs.setdefault(result["parser"], []).append(result)
    return runs


def repeats(split: str, path: str) -> list[Path]:
    return sorted(p.parent for p in (RESULTS / split / path).glob("r*/run.json"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extraction_rows(split: str, runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for parser, results in runs.items():
        for repeat, result in enumerate(results, 1):
            for i in result["items"]:
                fields = M.proposal_fields(i["expected"], i["got"]) or {}
                rows.append({"split": split, "parser": parser, "repeat": repeat, "id": i["id"], "lang": i["lang"], "ambiguous": i["ambiguous"],
                             "correct": i["correct"], "understood_ok": i["understood_ok"], "json_valid": i["json_valid"], **{f: fields.get(f, "") for f in ("slice", "metric", "value", "window")},
                             "text": i["text"], "got": json.dumps(i["got"], ensure_ascii=False)})
    return rows


def interface_rows(split: str) -> list[dict]:
    rows = []
    labels = {r["id"]: r["expected"] for r in load_intents(split)}
    for path in ("gateway", "hermes"):
        for run in repeats(split, path):
            for item in M.load_items(run):
                start, end = item["start"], item["end"] or {}
                forward, back = M.interface_latency(item) if path == "hermes" else (None, None)
                rows.append({"split": split, "path": path, "repeat": run.name, "id": start["id"], "kind": start["kind"], "ambiguous": start["ambiguous"],
                             "outcome": M.outcome_class(end), "check": (end.get("verifier") or {}).get("check", ""), "state_changed": M.state_changed(item),
                             "relaxed": "; ".join(M.relaxed(item, labels.get(start["id"]))),
                             "fidelity": M.fidelity(item)[0] if path == "hermes" else "", "forward_s": "" if forward is None else round(forward, 2),
                             "reply_s": "" if back is None else round(back, 2), "probe_answer": end.get("probe_answer", ""),
                             "hermes_tools": ";".join(end.get("hermes_tools") or []), "reason": str(end.get("reason") or "")[:160]})
    return rows


def plot_extraction(runs: dict[str, list[dict]]) -> Figure:
    """Share right per measure, for the model and the regex baseline. The bar is the interval on the intents."""
    fig = Figure(figsize=(10, 3.8), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    measures = ["fully correct", "slice", "metric", "target value", "time window", "ambiguous\ndetected"]
    for parser, results in runs.items():
        shift, colour = (0.12, ACCENT) if parser != "rules" else (-0.12, NEUTRAL)
        counted = [M.summarise_extraction(r["items"]) for r in results]
        first = counted[0]
        for x, (key, n) in enumerate((("correct", first["n"]), ("slice", first["clear"]), ("metric", first["clear"]), ("value", first["clear"]),
                                     ("window", first["clear"]), ("ambiguous_detected", first["ambiguous"]))):
            shares = [(c["correct"] / n if key == "correct" else c["ambiguous_detected"] / n if key == "ambiguous_detected" else c["fields"][key]) for c in counted]
            low, high = M.wilson(round(float(np.median(shares)) * n), n)
            ax.vlines(x + shift, low * 100, high * 100, color=colour, linewidth=1.6)
            ax.scatter([x + shift] * len(shares), [s * 100 for s in shares], s=34, color=colour, edgecolors=SURFACE, linewidths=0.8, zorder=3,
                       label=parser if x == 0 else None)
    ax.set_xticks(range(len(measures)), measures, fontsize=9)
    ax.set_ylim(-3, 104)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    titled(ax, "Extraction per measure: a dot per repeat; the line is the 95 % interval from the small number of intents")
    ax.set_ylabel("% right", color=TEXT_MUTED)
    return fig


def plot_grid(runs: dict[str, list[dict]]) -> Figure:
    """Each intent against each repeat: which ones go wrong, and whether always."""
    columns = [(parser, i, r) for parser, results in runs.items() for i, r in enumerate(results, 1)]
    ids = [i["id"] for i in columns[0][2]["items"]]
    fig = Figure(figsize=(7.6, 0.27 * len(ids) + 1.5), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    for x, (_, _, result) in enumerate(columns):
        for item in result["items"]:
            y = ids.index(item["id"])
            ax.scatter(x, y, s=48, marker="o" if item["correct"] else "x", color=NEUTRAL if item["correct"] else "#c0392b", linewidths=1.8 if not item["correct"] else 0.6)
    info = {i["id"]: i for i in columns[0][2]["items"]}
    ax.set_yticks(range(len(ids)), [f"{i} ({info[i]['lang']}{', ambiguous' if info[i]['ambiguous'] else ''})" for i in ids], fontsize=8)
    ax.set_xticks(range(len(columns)), [f"{p.split(':')[0]}\nr{i}" for p, i, _ in columns], fontsize=7.5)
    ax.invert_yaxis()
    _style(ax, "")
    ax.grid(False)
    ax.set_title("Each intent, each repeat: dot right, cross wrong", loc="left", color=TEXT, fontsize=10)
    return fig


def plot_tools(split: str) -> Figure:
    """Calls per tool and outcome, over all repeats, one panel per path. Counts are labelled, not only drawn."""
    paths = [p for p in ("gateway", "hermes") if repeats(split, p)]
    fig = Figure(figsize=(11, 2.9), facecolor=SURFACE, layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(paths)))
    for ax, path in zip(axes, paths):
        total = {t: Counter() for t in M.GATEWAY_TOOLS}
        for run in repeats(split, path):
            for tool, counts in M.tool_call_table(M.load_items(run)).items():
                total[tool].update(counts)
        for y, tool in enumerate(M.GATEWAY_TOOLS):
            left = 0
            for outcome, colour in OUTCOME_COLOURS.items():
                n = total[tool][outcome]
                if not n:
                    continue
                ax.barh(y, n, left=left, color=colour, edgecolor=SURFACE, linewidth=1.5, height=0.55, label=OUTCOME_LABELS[outcome])
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8, color="#ffffff" if colour != "#eda100" else TEXT)
                left += n
        ax.set_yticks(range(3), M.GATEWAY_TOOLS, fontsize=8.5)
        ax.invert_yaxis()
        titled(ax, f"Through {'the MCP client' if path == 'gateway' else 'Hermes'}: calls over all repeats")
        ax.grid(False)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), ncols=4)
    return fig


def plot_hermes(rows: list[dict]) -> Figure:
    """Per intent: the two legs of interface latency, and what became of the text (the marker's shape)."""
    mine = [r for r in rows if r["path"] == "hermes" and r["kind"] == "intent"]
    ids = sorted({r["id"] for r in mine})
    fig = Figure(figsize=(11, 0.27 * max(len(ids), 1) + 1.6), facecolor=SURFACE, layout="constrained")
    forward, back = fig.subplots(1, 2, sharey=True)
    seen = set()
    for r in mine:
        y = ids.index(r["id"])
        marker, colour = FIDELITY_MARKS[r["fidelity"]]
        label = r["fidelity"].replace("_", " ") if r["fidelity"] not in seen else None
        seen.add(r["fidelity"])
        if r["forward_s"] != "":
            forward.scatter(r["forward_s"], y, marker=marker, color=colour, s=30, label=label, zorder=3)
            back.scatter(r["reply_s"], y, marker=marker, color=colour, s=30, zorder=3)
        else:
            forward.scatter(0, y, marker=marker, color=colour, s=44, label=label, zorder=3, clip_on=False)
    forward.set_yticks(range(len(ids)), ids, fontsize=8)
    forward.invert_yaxis()
    forward.set_xlim(left=0)
    back.set_xlim(left=0)
    forward.legend(frameon=False, fontsize=8, loc="lower right")
    titled(forward, "Typed -> at the gateway, s (a cross on 0: never sent)")
    titled(back, "Gateway's reply -> displayed, s")
    return fig


def plot_scope(rows: list[dict]) -> Figure:
    """Requests that must change nothing. One row per request; a mark for each run, red if the RIC changed."""
    mine = [r for r in rows if r["kind"] in ("probe", "lockdown") or str(r["ambiguous"]) == "True"]
    labels = list(dict.fromkeys((r["kind"], r["id"]) for r in mine))
    fig = Figure(figsize=(7.5, 0.24 * max(len(labels), 1) + 1.4), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    columns = sorted({(r["path"], r["repeat"]) for r in mine})
    for r in mine:
        y, x = labels.index((r["kind"], r["id"])), columns.index((r["path"], r["repeat"]))
        changed = r["state_changed"]
        ax.scatter(x, y, s=36, marker="x" if changed else "o", color="#c0392b" if changed else NEUTRAL, linewidths=1.8 if changed else 0.6)
    ax.set_yticks(range(len(labels)), [f"{k}: {i}" for k, i in labels], fontsize=7.5)
    ax.set_xticks(range(len(columns)), [f"{p}\n{r}" for p, r in columns], fontsize=7.5)
    ax.invert_yaxis()
    _style(ax, "")
    ax.grid(color=GRID, linewidth=0.4)
    ax.set_title("Must change nothing: dot no change, red cross the RIC changed", loc="left", color=TEXT, fontsize=10)
    return fig


def share(k: int, n: int) -> str:
    low, high = M.wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, interval {low:.0%}-{high:.0%})" if n else "-"


def print_summary(split: str, runs: dict[str, list[dict]], rows: list[dict]) -> None:
    print(f"\nExtraction, {split}")
    for parser, results in runs.items():
        for repeat, result in enumerate(results, 1):
            c = M.summarise_extraction(result["items"])
            wrong = [i["id"] for i in result["items"] if not i["correct"]]
            print(f"  {parser} r{repeat}: correct {share(c['correct'], c['n'])}; ambiguous detected {c['ambiguous_detected']}/{c['ambiguous']}; "
                  f"clear ones refused {c['clear_refused']}/{c['clear']}; fields {c['fields']}; per-field (annotation) {result['per_field']}; "
                  f"latency median {result['latency_median_s']:.2f} s, p95 {result['latency_p95_s']:.2f} s; wrong {wrong}")
    for path in ("gateway", "hermes"):
        for run in repeats(split, path):
            print(f"\n{path} {run.name}: {json.loads((run / 'run.json').read_text())['wall_s']} s; tools {M.tool_call_table(M.load_items(run))}")
        mine = [r for r in rows if r["path"] == path]
        if not mine:
            continue
        must_not = [r for r in mine if r["kind"] in ("probe", "lockdown") or str(r["ambiguous"]) == "True"]
        print(f"  outcomes of intents: {dict(Counter((r['outcome'], r['check']) for r in mine if r['kind'] == 'intent'))}")
        print(f"  must change nothing: {len(must_not)} requests, changed {[(r['repeat'], r['id']) for r in must_not if r['state_changed']]}")
        print(f"  not accepted: {sorted({(r['id'], r['outcome'], r['check']) for r in mine if r['outcome'] not in ('accepted', 'no_intent')})}")
        print(f"  accepted with a stated value changed: {sorted((r['repeat'], r['id'], r['relaxed']) for r in mine if r['relaxed'])}")
    hermes = [r for r in rows if r["path"] == "hermes" and r["kind"] == "intent"]
    if hermes:
        # Per repeat, on its intents: repeats of one text are not independent samples.
        print(f"\nFidelity: {dict(Counter(r['fidelity'] for r in hermes))} of {len(hermes)} turns")
        for run in sorted({r["repeat"] for r in hermes}):
            mine = [r for r in hermes if r["repeat"] == run]
            print(f"  {run}: exact {share(sum(r['fidelity'] == 'exact' for r in mine), len(mine))}, not sent {sum(r['fidelity'] == 'not_sent' for r in mine)}")
        print(f"  not sent: {sorted({(r['repeat'], r['id']) for r in hermes if r['fidelity'] == 'not_sent'})}")
        for key in ("forward_s", "reply_s"):
            values = [r[key] for r in hermes if r[key] != ""]
            if values:
                first = {min(r["id"] for r in hermes)}  # the turn right after a container start
                warm = [r[key] for r in hermes if r[key] != "" and r["id"] not in first]
                print(f"  {key}: median {np.median(values):.1f} s, p95 {np.percentile(values, 95):.1f} s, n {len(values)}; "
                      f"without each repeat's first turn median {np.median(warm):.1f} s, p95 {np.percentile(warm, 95):.1f} s, n {len(warm)}")
        tools = {r["hermes_tools"] for r in rows if r["path"] == "hermes"}
        print(f"  tools Hermes used: {sorted(tools)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["tune", "heldout"], default="heldout")
    parser.add_argument("--out", type=Path, default=None, help="where figures and tables go instead of docs/learn (rehearsal)")
    args = parser.parse_args()
    runs = extraction_runs(args.split)
    rows = interface_rows(args.split)
    data, img = (args.out, args.out) if args.out else (DATA, IMG)
    write_csv(data / "M08b-extraction.csv", extraction_rows(args.split, runs))
    write_csv(data / "M08b-interface.csv", rows)
    img.mkdir(parents=True, exist_ok=True)
    figures = {"M08b-extraction": plot_extraction(runs), "M08b-grid": plot_grid(runs), "M08b-tools": plot_tools(args.split), "M08b-scope": plot_scope(rows)}
    if any(r["path"] == "hermes" for r in rows):
        figures["M08b-hermes"] = plot_hermes(rows)
    for name, fig in figures.items():
        fig.savefig(img / f"{name}.png", dpi=120)
    print_summary(args.split, runs, rows)
    print(f"\nwrote {sorted(figures)} to {img}")


if __name__ == "__main__":
    main()
