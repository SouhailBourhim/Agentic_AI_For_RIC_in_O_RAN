"""The figures of M10b's checks (run `eval.m10b --check` for cell revisions 1 and 2 first).

  uv run python -m eval.plot_m10b

Reads only committed tables: docs/learn/data/M10b-runs-r{1,2}.csv, M10b-result-r{1,2}.json, and M09b-runs.csv for
the calibrated cell's reference. Writes into docs/learn/img/:
- M10b-executors.png: mean intent met (accepted intents) per executor, all families and each family, cell revision 2.
  Executors by position; the M10c comparator (formula_est) in the project's accent colour, everything else grey.
- M10b-checks.png: (a) the difficulty check per family: the share of intents the oracle meets in >= 90 % of their
  window, on the m09 cell, m10 revision 1 and revision 2, against the 80 % rule; (b) the reported decomposition, with
  95 % cluster-bootstrap intervals: foresight (oracle - formula_told) and estimation cost (formula_told - formula_est).
"""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval.generator import FAMILIES
from eval.harness import EXECUTORS
from eval.m10b import DIFFICULTY_MET, DIFFICULTY_SHARE
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED

DATA = Path("docs/learn/data")
REVISIONS = (1, 2)
FAMILY_LABELS = {"all": "all families", "f1": "F1 load surge", "f2": "F2 new goal", "f3": "F3 fast dynamics",
                 "f4": "F4 conflicts", "f5": "F5 easy"}
EXECUTOR_LABELS = {"static": "static", "v7": "v7", "rev4": "rev 4", "formula_est": "formula_est", "mpc_est": "mpc_est",
                   "formula_told": "formula\ntold", "oracle": "oracle"}
CELL_SHADES = {"m09": "#b9b7b0", 1: "#8d8b85", 2: TEXT}  # the calibrated cell lightest, the last revision darkest


def accepted_rows(revision: int) -> list[dict]:
    with (DATA / f"M10b-runs-r{revision}.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["accepted"] == "True"]


def oracle_share(rows: list[dict], family: str | None) -> float:
    mine = [float(r["intent_met"]) >= DIFFICULTY_MET for r in rows if family is None or r["family"] == family]
    return float(np.mean(mine))


def m09_oracle_rows() -> list[dict]:
    """The same odd dev instances on the calibrated cell (M09b's oracle), F4b left out as the verifier refuses it."""
    with (DATA / "M09b-runs.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["controller"] == "oracle" and int(r["instance"][-2:]) % 2 == 1
                and not (r["family"] == "f4" and r["variant"] == "b")]


def plot_executors(revision: int = 2) -> Figure:
    rows = accepted_rows(revision)
    fig = Figure(figsize=(13, 6.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(2, 3, sharey=True)
    x = np.arange(len(EXECUTORS))
    for ax, family in zip(axes.flat, ["all", *FAMILIES], strict=True):
        means = [np.mean([float(r["intent_met"]) for r in rows if r["executor"] == e
                          and (family == "all" or r["family"] == family)]) for e in EXECUTORS]
        colours = [ACCENT if e == "formula_est" else NEUTRAL for e in EXECUTORS]
        ax.vlines(x, 0, means, color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(x, means, s=46, color=colours, zorder=3)
        for k, m in enumerate(means):
            ax.annotate(f"{m:.2f}", (k, m), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5,
                        color=TEXT)
        ax.set_xticks(x, [EXECUTOR_LABELS[e] for e in EXECUTORS], fontsize=7.5)
        ax.set_ylim(0, 1.1)
        n = len({(r["instance"], r["intent"]) for r in rows if family == "all" or r["family"] == family})
        titled(ax, f"{FAMILY_LABELS[family]} ({n} intents)")
    axes[0, 0].set_ylabel("mean intent met", color=TEXT_MUTED, fontsize=8.5)
    axes[1, 0].set_ylabel("mean intent met", color=TEXT_MUTED, fontsize=8.5)
    return fig


def plot_checks() -> Figure:
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    share_ax, parts_ax = fig.subplots(1, 2, width_ratios=[1.8, 1])
    cells = {"m09": m09_oracle_rows(), **{r: accepted_rows(r) for r in REVISIONS}}
    labels = {"m09": "m09 (calibrated, no channel)", 1: "m10 revision 1 (8-18 dB)", 2: "m10 revision 2 (10-20 dB)"}
    groups = ["all", *FAMILIES]
    x = np.arange(len(groups))
    for k, (cell, rows) in enumerate(cells.items()):
        if cell != "m09":
            rows = [r for r in rows if r["executor"] == "oracle"]
        shares = [oracle_share(rows, None if g == "all" else g) for g in groups]
        share_ax.bar(x + (k - 1) * 0.26, shares, width=0.24, color=CELL_SHADES[cell], label=labels[cell])
    share_ax.axhline(DIFFICULTY_SHARE, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)), label="rule: 80 %")
    share_ax.set_xticks(x, [FAMILY_LABELS[g] for g in groups], fontsize=8)
    share_ax.set_ylim(0, 1.0)
    share_ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncols=4)
    titled(share_ax, "(a) difficulty: share of intents the oracle meets in >= 90 % of their window")
    share_ax.title.set_position((0.0, 1.13))

    measures = (("foresight", "foresight\noracle - formula_told"), ("estimation", "estimation cost\nformula_told - formula_est"))
    for k, revision in enumerate(REVISIONS):
        result = json.loads((DATA / f"M10b-result-r{revision}.json").read_text())
        for j, (key, _) in enumerate(measures):
            mean, low, high = result[key]
            y = j + (k - 0.5) * 0.25
            parts_ax.hlines(y, low, high, color=CELL_SHADES[revision], linewidth=2)
            parts_ax.scatter([mean], [y], s=36, color=CELL_SHADES[revision], zorder=3,
                             label=f"revision {revision}" if j == 0 else None)
    parts_ax.axvline(0, color=TEXT_MUTED, linewidth=0.8)
    parts_ax.set_yticks(range(len(measures)), [label for _, label in measures], fontsize=8)
    parts_ax.set_xlabel("difference in mean intent met (95 % interval)", color=TEXT_MUTED, fontsize=8.5)
    parts_ax.set_ylim(-0.6, 1.6)
    parts_ax.legend(frameon=False, fontsize=8, loc="upper right")
    titled(parts_ax, "(b) where the oracle's lead comes from")
    return fig


def main(out_dir: Path = IMG) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in (("M10b-executors.png", plot_executors()), ("M10b-checks.png", plot_checks())):
        fig.savefig(out_dir / name, dpi=150, facecolor=SURFACE)
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
