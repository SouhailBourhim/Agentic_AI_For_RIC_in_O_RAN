"""Figure for M13b, the load-aware verifier, on F7's dev split (scored once).

  uv run python -m eval.plot_m13b

docs/learn/img/M13b-dev.png: one marker per instance and arm, against URLLC's traffic during the request. Left: the
guarantee already in force (eMBB + BE), met. Right: the URLLC request, delivered against what was stated. Reads the
committed docs/learn/data/M13b-dev-runs.csv. Colours: grey = no offer (as before M13), dark grey = M13's offer,
accent = M13b's; each with its own marker.
"""

import csv
import json
from pathlib import Path

from matplotlib.figure import Figure

from eval.plot_control import ACCENT, IMG, NEUTRAL, titled
from sim.plot import GRID, SURFACE, TEXT_MUTED

DATA = Path("docs/learn/data")
ARMS = {"reject": ("rejected, no offer", "#b4b2ac", "D"), "blind_offer": ("M13's offer (load-blind)", NEUTRAL, "s"),
        "aware_offer": ("M13b's offer (load-aware)", ACCENT, "o")}


def plot_dev() -> Figure:
    rows = list(csv.DictReader((DATA / "M13b-dev-runs.csv").open()))
    verdict = json.loads((DATA / "M13b-dev.json").read_text())["verdict"]
    offers = [json.loads(r["offer"])["urllc"] for r in rows if r["label"] == "blind_offer"]
    fig = Figure(figsize=(12.5, 4.2), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, 2, sharex=True)
    for ax, key, title in ((axes[0], "in_force_met", f"guarantee in force (eMBB + BE), met: aware − reject "
                                                      f"{verdict['H_in_force_change']:+.3f} (rule ≥ −0.02)"),
                           (axes[1], "delivered", f"URLLC request, delivered: where M13b offers, aware − reject "
                                                  f"{verdict['G_gain_where_offered']:+.3f} (rule ≥ +0.05)")):
        ax.axvspan(min(offers), max(offers), color=GRID, alpha=0.5, zorder=0,
                   label=f"the offers ({min(offers):g}-{max(offers):g} Mb/s)")
        for label, (name, colour, marker) in ARMS.items():
            mine = [r for r in rows if r["label"] == label]
            ax.scatter([float(r["traffic"]) for r in mine], [float(r[key]) for r in mine], s=30, color=colour,
                       marker=marker, label=name, edgecolors=SURFACE, linewidths=1.0, zorder=3 if label == "aware_offer" else 2)
        ax.set_ylim(-0.03, 1.05)
        titled(ax, title)
        ax.set_xlabel("URLLC traffic during the request (Mb/s; its target is 30-31)", color=TEXT_MUTED, fontsize=8.5)
    axes[0].set_ylabel("share of the window", color=TEXT_MUTED, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    return fig


def main() -> None:
    fig = plot_dev()
    fig.savefig(IMG / "M13b-dev.png", dpi=150, facecolor=SURFACE)
    print(IMG / "M13b-dev.png")


if __name__ == "__main__":
    main()
