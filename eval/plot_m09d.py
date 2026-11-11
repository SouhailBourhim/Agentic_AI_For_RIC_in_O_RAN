"""The figures of M09d's test-seed result (run `eval.m09d --score --final` first).

  uv run python -m eval.plot_m09d

Writes docs/learn/img/M09d-hypotheses.png (P1-P5: LLM + RL minus the other controller, mean paired difference with
its 95 % interval, thick, and 99 % interval, thin; the pass line of each rule drawn) and M09d-verifier.png (P6).
Reads only the tables M09d wrote (docs/learn/data/M09d-hypotheses.csv, M09d-p6.csv). Same conventions as M09b:
purple for a claim that passes, grey for one that doesn't, and the verdict written next to it, not left to colour.
"""

import csv
from pathlib import Path

from matplotlib.figure import Figure

from eval.m09d import HYPOTHESES_CSV, MARGIN, P6_CSV, P6_PRECISION
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import SURFACE, TEXT, TEXT_MUTED

LABELS = {"static": "static", "rl_only": "RL only", "llm_only_v2": "LLM only v2", "formula": "LLM + formula"}
RULE = {"superior": "pass if the interval is above 0", "noninferior": f"pass if the interval is above -{MARGIN:g}",
        "faster": "pass if the interval is below 0"}


def read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def passed(row: dict, level: str = "95") -> bool:
    return row[f"pass_{level}"] in ("True", True)


def forest(ax, rows: list[dict], scale: float, unit: str) -> None:
    """One line per hypothesis, top to bottom in P order."""
    for y, row in enumerate(reversed(rows)):
        mid, low95, high95, low99, high99 = (float(row[k]) * scale for k in ("mean_diff", "low_95", "high_95", "low_99", "high_99"))
        colour = ACCENT if passed(row) else NEUTRAL
        ax.hlines(y, low99, high99, color=colour, linewidth=1)
        ax.hlines(y, low95, high95, color=colour, linewidth=3.5)
        ax.scatter(mid, y, s=36, color=colour, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        verdict = "PASS" if passed(row) else "FAIL"
        extra = "" if passed(row) == passed(row, "99") else " (99 %: " + ("PASS" if passed(row, "99") else "FAIL") + ")"
        ax.annotate(f"{verdict}{extra}  {mid:+.2f}{unit}", (high99, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks(range(len(rows)), [f"{r['hypothesis']} {r['family'].upper()}: vs {LABELS[r['other']]}\n({RULE[r['kind']]})"
                                     for r in reversed(rows)], fontsize=8.5)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax.margins(x=0.35)


def plot_hypotheses(rows: list[dict]) -> Figure:
    met = [r for r in rows if r["measure"] == "intent_met"]
    reaction = [r for r in rows if r["measure"] == "reaction"]
    fig = Figure(figsize=(13, 4.8), facecolor=SURFACE, layout="constrained")
    left, right = fig.subplots(1, 2, width_ratios=[2.2, 1])  # two scales (points, seconds): two axes, never one
    forest(left, met, 100, " pts")
    left.axvline(-MARGIN * 100, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
    titled(left, "Intent met, LLM + RL minus the other, points (dashed: the -5 margin of P4 and P5)")
    left.set_xlabel("points of the window met", color=TEXT_MUTED)
    forest(right, reaction, 1, " s")
    titled(right, "P3b reaction time, seconds")
    right.set_xlabel("s, LLM + RL minus LLM only v2\n(never reached = whole window)", color=TEXT_MUTED)
    fig.suptitle("M09d, test seeds, policy v7: mean paired difference, thick = 95 % interval (decides), thin = 99 %",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    return fig


def plot_verifier(p6: dict) -> Figure:
    """P6a precision and P6b recall under both feasibility labels, and P6c's counts in the title."""
    fig = Figure(figsize=(9, 3.2), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    bars = [("P6a: accepted that were feasible\n(oracle or static, decides)", "oracle_or_static_precision", True),
            ("P6a: accepted that were feasible\n(oracle only, reported)", "oracle_only_precision", False),
            ("P6b: feasible that were accepted\n(oracle or static)", "oracle_or_static_recall", False)]
    for y, (_label, key, decides) in enumerate(reversed(bars)):
        value = float(p6[key]) * 100
        ax.barh(y, value, height=0.55, color=ACCENT if decides else NEUTRAL)
        text = f"{value:.1f} %"
        if f"{key}_low" in p6:
            low, high = float(p6[f"{key}_low"]) * 100, float(p6[f"{key}_high"]) * 100
            ax.hlines(y, low, high, color=TEXT, linewidth=1.2)
            text += f"  [{low:.1f}, {high:.1f}]"
        ax.annotate(text, (100, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT)
    ax.axvline(P6_PRECISION * 100, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(bars)), [b[0] for b in reversed(bars)], fontsize=8.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 95, 100])
    titled(ax, f"LLM + RL, {p6['intents']} intents, {p6['accepted']} accepted (dashed: P6a's 95 % line)")
    unverified = sum(int(p6[k]) for k in p6 if k.endswith("_unverified_applies"))
    over = sum(int(p6[k]) for k in p6 if k.endswith("_over_range_applied"))
    fig.suptitle(f"M09d, the verifier. P6c, all agent controllers: {unverified} applies without an accepted verification, "
                 f"{over} over-range requests applied", color=TEXT, fontsize=10.5, x=0.01, ha="left")
    return fig


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    figures = {"M09d-hypotheses": plot_hypotheses(read(HYPOTHESES_CSV)), "M09d-verifier": plot_verifier(read(P6_CSV)[0])}
    for name, fig in figures.items():
        fig.savefig(IMG / f"{name}.png", dpi=120)
        print(f"wrote {IMG / name}.png")


if __name__ == "__main__":
    main()
