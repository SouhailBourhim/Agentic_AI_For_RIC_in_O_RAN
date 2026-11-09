"""M09g's diagnostics on the dev benchmark: why does LLM + formula beat LLM + RL (M09b)? No training, dev seeds only.

  uv run python -m eval.diag_m09g --d1      after `eval.benchmark --dev --controllers formula_a10` and `--table`
  uv run python -m eval.diag_m09g --d2      reads the M09b runs already on disk

- D1, actuator fairness: the formula acts at once, LLM + RL is served behind alpha 0.10 (~10 s to reach a new split).
  formula_a10 is the same formula behind the same filter. The paired drop formula -> formula_a10, intent by intent,
  says how much of the formula's lead is its unfiltered actuator rather than what it asks for.
- D2, edge-aiming: reward v1's shortfall hinge is at exactly min(target, load), while intent met is a step there. A
  policy that learns to sit on the hinge lands just short half the time. Per second of each intent's window and per
  slice with a target: margin = throughput - min(target, load), in Mb/s, per family and controller.
The reading rules are in docs/DECISIONS.md (2026-11-09), written before any of these numbers was computed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from eval import benchmark as B
from eval.generator import FAMILIES, dev_instances
from eval.metrics import in_window, load_run
from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import SURFACE, TEXT, TEXT_MUTED

D1_CSV = Path("docs/learn/data/M09g-d1.csv")
D2_CSV = Path("docs/learn/data/M09g-margin.csv")
D1_CONTROLLERS = ("proposed", "formula", "formula_a10")
D2_CONTROLLERS = ("proposed", "formula", "llm_only_v2", "formula_a10")
LABELS = {"proposed": "LLM + RL (v7)", "formula": "LLM + formula", "formula_a10": "LLM + formula, alpha 0.10",
          "llm_only_v2": "LLM only v2"}
# D1's rule (DECISIONS): the filter explains the F3 gap if formula_a10's F3 mean falls to within 0.10 of v8_nooverserve's
# 0.763 (M09c-selection.csv), i.e. to 0.863 or below, and the paired drop's 95 % interval is above 0.
D1_FAMILY, D1_REFERENCE, D1_WITHIN = "f3", 0.763, 0.10
# D2's rule (DECISIONS): A2 is supported if, on F2, LLM + RL's median margin is within +-1 Mb/s, at least 5 % of its
# seconds are just short (-1 <= margin < 0), and the formula's just-short share is under half of LLM + RL's.
D2_FAMILY, D2_EDGE_MBPS, D2_MIN_JUST_SHORT = "f2", 1.0, 0.05
SHORT_TOLERANCE = 1e-9  # as eval.metrics.THROUGHPUT_TOLERANCE: below this, the second is not met


# ---------------------------------------------------------------- D1


def d1(rows: list[dict]) -> list[dict]:
    """Per family (and all): mean intent met of the three controllers, and the paired drop formula - formula_a10."""
    met = {c: {(r["instance"], r["intent"]): (r["family"], float(r["intent_met"])) for r in rows if r["controller"] == c}
           for c in D1_CONTROLLERS}
    keys = sorted(set.intersection(*(set(m) for m in met.values())))
    if not keys:
        raise SystemExit("no intent has all three controllers: run the formula_a10 dev runs and `eval.benchmark --table`")
    out = []
    for family in (*FAMILIES, "all"):
        mine = [k for k in keys if family == "all" or met["formula"][k][0] == family]
        row: dict = {"family": family, "n": len(mine)}
        for c in D1_CONTROLLERS:
            row[f"{c}_mean"] = round(float(np.mean([met[c][k][1] for k in mine])), 3)
        drop = np.array([met["formula"][k][1] - met["formula_a10"][k][1] for k in mine])
        low, high = B.bootstrap_ci(drop, stat=np.mean)
        row |= {"drop": round(float(drop.mean()), 3), "drop_low": round(low, 3), "drop_high": round(high, 3)}
        gap = row["formula_mean"] - row["proposed_mean"]
        row["share_of_gap"] = round(float(drop.mean()) / gap, 2) if gap > 1e-9 else ""
        out.append(row)
    return out


def d1_verdict(rows: list[dict]) -> str:
    f3 = next(r for r in rows if r["family"] == D1_FAMILY)
    holds = f3["formula_a10_mean"] <= D1_REFERENCE + D1_WITHIN and f3["drop_low"] > 0
    return (f"F3: formula {f3['formula_mean']:.3f} -> formula_a10 {f3['formula_a10_mean']:.3f} "
            f"(drop {f3['drop']:+.3f} [{f3['drop_low']:+.3f}, {f3['drop_high']:+.3f}]); rule: <= {D1_REFERENCE + D1_WITHIN:.3f} "
            f"and interval above 0 -> {'HOLDS: the filter explains the F3 gap, A1 enters v9' if holds else 'does not hold: A1 stays out of v9'}")


def plot_d1(rows: list[dict]) -> Figure:
    fig = Figure(figsize=(9, 3.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    families = [r for r in rows if r["family"] != "all"]
    width = 0.26
    colours = {"proposed": ACCENT, "formula": NEUTRAL, "formula_a10": "none"}
    for j, c in enumerate(D1_CONTROLLERS):
        xs = np.arange(len(families)) + (j - 1) * width
        ax.bar(xs, [r[f"{c}_mean"] * 100 for r in families], width * 0.92, color=colours[c],
               edgecolor=NEUTRAL if c == "formula_a10" else colours[c], hatch="///" if c == "formula_a10" else None,
               linewidth=0.8, label=LABELS[c])
    ax.set_xticks(range(len(families)), [r["family"].upper() for r in families])
    ax.set_ylim(0, 105)
    ax.set_ylabel("mean seconds of the window met, %", color=TEXT_MUTED)
    titled(ax, "D1: the formula behind LLM + RL's filter (alpha 0.10), dev seeds")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncols=3)
    return fig


# ---------------------------------------------------------------- D2


def margins(work: Path, intents) -> np.ndarray:
    """throughput - min(target, load), Mb/s, every second of every intent's window, for every slice with a target."""
    kpis, _ = load_run(work)
    out = []
    for intent in intents:
        for s, target in intent.targets.items():
            k = kpis[s]
            keep = in_window(k["t"], *intent.window) & ~np.isnan(k["thr"]) & ~np.isnan(k["load"])
            out.append(k["thr"][keep] - np.minimum(target, k["load"][keep]))
    return np.concatenate(out) if out else np.array([])


def collect_margins() -> dict[tuple[str, str], np.ndarray]:
    skip = B.dropped()
    found: dict[tuple[str, str], list[np.ndarray]] = {}
    for inst in dev_instances():
        if inst.name in skip:
            continue
        for c in D2_CONTROLLERS:
            work = B.work_of(inst, c)
            if (work / "run.json").exists():
                found.setdefault((inst.family, c), []).append(margins(work, inst.intents))
    return {k: np.concatenate(v) for k, v in found.items()}


def d2(found: dict[tuple[str, str], np.ndarray]) -> list[dict]:
    out = []
    for family in (*FAMILIES, "all"):
        for c in D2_CONTROLLERS:
            parts = [v for (f, cc), v in found.items() if cc == c and (family == "all" or f == family)]
            if not parts:
                continue
            m = np.concatenate(parts)
            out.append({"family": family, "controller": c, "seconds": len(m),
                        "median_mbps": round(float(np.median(m)), 2), "p10_mbps": round(float(np.percentile(m, 10)), 2),
                        "p90_mbps": round(float(np.percentile(m, 90)), 2),
                        "just_short": round(float(np.mean((m >= -D2_EDGE_MBPS) & (m < -SHORT_TOLERANCE))), 4),
                        "short_more": round(float(np.mean(m < -D2_EDGE_MBPS)), 4),
                        "met": round(float(np.mean(m >= -SHORT_TOLERANCE)), 4)})
    return out


def d2_verdict(rows: list[dict]) -> str:
    get = {(r["family"], r["controller"]): r for r in rows}
    rl, formula = get[(D2_FAMILY, "proposed")], get[(D2_FAMILY, "formula")]
    holds = (abs(rl["median_mbps"]) <= D2_EDGE_MBPS and rl["just_short"] >= D2_MIN_JUST_SHORT
             and formula["just_short"] < rl["just_short"] / 2)
    return (f"F2: LLM + RL median {rl['median_mbps']:+.2f} Mb/s, just short {rl['just_short']:.1%}; formula median "
            f"{formula['median_mbps']:+.2f}, just short {formula['just_short']:.1%} -> "
            f"{'SUPPORTS A2: reward v4 enters v9' if holds else 'does not support A2: reward v4 stays out of v9'}")


def plot_d2(found: dict[tuple[str, str], np.ndarray]) -> Figure:
    """Empirical CDF of the margin around 0, per controller: on F2 (the rule) and over all families."""
    fig = Figure(figsize=(10, 3.8), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    styles = {"proposed": (ACCENT, "-"), "formula": (NEUTRAL, "-"), "llm_only_v2": (NEUTRAL, ":"), "formula_a10": (NEUTRAL, "--")}
    for ax, family in zip(axes, (D2_FAMILY, "all")):
        for c in D2_CONTROLLERS:
            parts = [v for (f, cc), v in found.items() if cc == c and (family == "all" or f == family)]
            if not parts:
                continue
            m = np.sort(np.concatenate(parts))
            colour, line = styles[c]
            ax.step(m, np.arange(1, len(m) + 1) / len(m) * 100, where="post", color=colour, linestyle=line,
                    linewidth=1.6, label=LABELS[c])
        ax.axvspan(-D2_EDGE_MBPS, 0, color=ACCENT, alpha=0.08, linewidth=0)
        ax.axvline(0, color=TEXT, linewidth=0.8)
        ax.set_xlim(-6, 15)
        ax.set_xlabel("throughput - min(target, load), Mb/s", color=TEXT_MUTED)
        titled(ax, "F2 new goal" if family == D2_FAMILY else "all families")
    axes[0].set_ylabel("seconds in the window, cumulative %", color=TEXT_MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("D2: where each controller lands against the target (shaded: just short, -1 to 0 Mb/s), dev seeds",
                 color=TEXT, fontsize=10.5, x=0.01, ha="left")
    return fig


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--d1", action="store_true", help="actuator fairness (needs the formula_a10 runs and --table)")
    group.add_argument("--d2", action="store_true", help="margin to the target, existing runs")
    args = parser.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    if args.d1:
        with B.RUNS.open() as f:
            rows = d1(list(csv.DictReader(f)))
        write(rows, D1_CSV)
        for r in rows:
            print(" ".join(f"{k} {v}" for k, v in r.items()))
        plot_d1(rows).savefig(IMG / "M09g-d1.png", dpi=150)
        print(d1_verdict(rows))
    else:
        found = collect_margins()
        rows = d2(found)
        write(rows, D2_CSV)
        for r in rows:
            print(" ".join(f"{k} {v}" for k, v in r.items()))
        plot_d2(found).savefig(IMG / "M09g-margin.png", dpi=150)
        print(d2_verdict(rows))


if __name__ == "__main__":
    main()
