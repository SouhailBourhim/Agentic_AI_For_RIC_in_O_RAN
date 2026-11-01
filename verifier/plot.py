"""Feasibility map and decision table of the verifier (M03).

Run with:  uv run python -m verifier.plot
- writes docs/learn/img/M03-feasibility.png (embedded in docs/learn/M03-verifier.md): for a grid
  of (URLLC target, eMBB target) pairs, which check decides, in three situations;
- prints a decision table of named example cases, one or more per check.

This module is a viewer, not part of the verifier's decisions: it may use numpy and
matplotlib, and no decision module imports it (tests/test_verifier_purity.py).
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from sim.plot import GRID, SURFACE, TEXT, TEXT_MUTED
from verifier.merge import ActiveIntent
from verifier.objective import RLObjective
from verifier.verify import VerifierConfig, VerifierState, verify

DEFAULT_OUT_DIR = Path("docs/learn/img")
NOW = 100.0
CAPACITY = 75.0

# One colour per deciding check. Categorical slots 4-8 of the reference palette: slots 1-3
# mean the three slices in every figure of the project, so they are not reused here.
OUTCOMES = {
    "ok": ("accepted", "#008300"),
    "budget": ("budget: targets exceed the cell", "#e34948"),
    "floor": ("floor: a slice left below 5 %", "#eda100"),
    "training_range": ("training range: target > 30 Mb/s", "#6250d6"),
    "conflict": ("conflict with the active intent", "#e87ba4"),
}
SHORT_LABELS = {"ok": "accepted", "budget": "budget", "floor": "floor",
                "training_range": "training range", "conflict": "conflict"}

# The map is about feasibility, not about how far the change is from what runs now, so the
# step check is switched off here (limits far above anything on the grid).
MAP_CONFIG = VerifierConfig(max_weight_step=1.0, max_target_step_fraction=100.0)

BE_20 = ActiveIntent(
    intent_id="be20", issued_at=50.0,
    objective=RLObjective(weights={"embb": 0.3, "urllc": 0.3, "be": 0.4}, min_thr_mbps={"be": 20.0},
                          valid_from=50.0, valid_until=NOW + 600.0),
)
PANELS = [
    ("(a) no latency bound", None, ()),
    ("(b) URLLC bound 5 ms", 5.0, ()),
    ("(c) bound 5 ms + active intent BE ≥ 20 Mb/s", 5.0, (BE_20,)),
]


def decision_grid(bound_ms: float | None, active: tuple[ActiveIntent, ...], step_mbps: float = 0.5) -> np.ndarray:
    """For each (eMBB target, URLLC target) on the grid, the index in OUTCOMES of the deciding check.

    Rows are eMBB targets, columns URLLC targets, both from 0 to the cell capacity.
    """
    state = VerifierState(now_s=NOW, capacity_mbps=CAPACITY, active=active)
    targets = np.arange(0.0, CAPACITY + step_mbps / 2, step_mbps)
    names = list(OUTCOMES)
    grid = np.zeros((len(targets), len(targets)), dtype=int)
    for i, embb in enumerate(targets):
        for j, urllc in enumerate(targets):
            proposal = {
                "weights": {"embb": 0.3, "urllc": 0.5, "be": 0.2},
                "min_thr_mbps": {"embb": float(embb), "urllc": float(urllc)},
                "max_latency_ms": bound_ms,
                "valid_from": NOW,
                "valid_until": NOW + 600.0,
            }
            grid[i, j] = names.index(verify(proposal, state, MAP_CONFIG).check)
    return grid


def label_position(grid: np.ndarray, index: int, window: int = 25) -> tuple[int, int] | None:
    """A cell deep inside the region of one outcome, to put its direct label there.

    Counts, for every cell, how many cells of that outcome are in the window around it
    (integral image), and takes the best one. None if the region is too small for a label.
    """
    mask = (grid == index).astype(float)
    if mask.sum() < window * window / 2:
        return None
    integral = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h = window // 2
    n = grid.shape[0]
    best, best_pos = -1.0, None
    for i in range(h, n - h):
        for j in range(h, n - h):
            count = (integral[i + h + 1, j + h + 1] - integral[i - h, j + h + 1]
                     - integral[i + h + 1, j - h] + integral[i - h, j - h])
            if mask[i, j] and count > best:
                best, best_pos = count, (i, j)
    return best_pos


def plot_feasibility_map(step_mbps: float = 0.5) -> Figure:
    """Three panels: URLLC target on x, eMBB target on y, coloured by the deciding check."""
    fig = Figure(figsize=(13, 5.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, len(PANELS), sharey=True)
    cmap = ListedColormap([colour for _, colour in OUTCOMES.values()])
    extent = (-step_mbps / 2, CAPACITY + step_mbps / 2, -step_mbps / 2, CAPACITY + step_mbps / 2)
    for ax, (title, bound, active) in zip(axes, PANELS):
        grid = decision_grid(bound, active, step_mbps)
        ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=len(OUTCOMES) - 1,
                  interpolation="nearest", aspect="auto")
        for index, name in enumerate(OUTCOMES):
            pos = label_position(grid, index)
            if pos is not None:
                # Keep the label box inside the panel (a label is ~15 Mb/s wide).
                y, x = pos[0] * step_mbps, float(np.clip(pos[1] * step_mbps, 9.0, CAPACITY - 9.0))
                ax.text(x, y, SHORT_LABELS[name], ha="center", va="center", fontsize=8.5, color=TEXT,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("URLLC minimum throughput (Mb/s)", color=TEXT_MUTED, fontsize=9)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel("eMBB minimum throughput (Mb/s)", color=TEXT_MUTED, fontsize=9)
    fig.legend(
        handles=[Patch(facecolor=colour, label=label) for label, colour in OUTCOMES.values()],
        loc="outside lower center", ncols=len(OUTCOMES), frameon=False, fontsize=9, labelcolor=TEXT,
    )
    fig.suptitle(
        f"Verifier decision for each pair of targets (cell {CAPACITY:.0f} Mb/s, BE target 0, step check off)",
        color=TEXT, fontsize=11,
    )
    return fig


def example_cases() -> list[tuple[str, dict, VerifierState]]:
    """Named examples for the decision table: at least one per check."""
    base = {"weights": {"embb": 0.1, "urllc": 0.8, "be": 0.1}, "min_thr_mbps": {"urllc": 10.0},
            "max_latency_ms": 10.0, "valid_from": NOW, "valid_until": NOW + 600.0}
    state = VerifierState(now_s=NOW, capacity_mbps=CAPACITY)
    with_be20 = state.model_copy(update={"active": (BE_20,)})

    def case(**changes) -> dict:
        return {**base, **changes}

    return [
        ("protect URLLC 10 Mb/s / 10 ms / 10 min", case(), state),
        ("weights sum to 1.1", case(weights={"embb": 0.2, "urllc": 0.8, "be": 0.1}), state),
        ("no end time", case(valid_until=None), state),
        ("lasts 5 s", case(valid_until=NOW + 5.0), state),
        ("URLLC bound 1.5 ms", case(max_latency_ms=1.5), state),
        ("30 + 30 + 16 Mb/s", case(min_thr_mbps={"embb": 30.0, "urllc": 30.0, "be": 16.0}, max_latency_ms=None), state),
        ("URLLC 30 Mb/s within 3 ms", case(min_thr_mbps={"urllc": 30.0}, max_latency_ms=3.0), state),
        ("eMBB 30 + URLLC 26 within 5 ms", case(min_thr_mbps={"embb": 30.0, "urllc": 26.0}, max_latency_ms=5.0), state),
        ("URLLC bound 60 ms", case(max_latency_ms=60.0), state),
        ("eMBB 30 + URLLC 22, BE 20 active", case(min_thr_mbps={"embb": 30.0, "urllc": 22.0}), with_be20),
        ("eMBB 20, BE 20 active", case(min_thr_mbps={"embb": 20.0}), with_be20),
        ("URLLC weight 0.3 -> 0.81", case(weights={"embb": 0.09, "urllc": 0.81, "be": 0.1}), state),
    ]


def decision_table() -> str:
    """The example cases and their decisions, as aligned text columns."""
    rows = [("case", "decision", "check", "reason")]
    for name, proposal, state in example_cases():
        d = verify(proposal, state, new_intent_id="new")
        rows.append((name, "ACCEPT" if d.accepted else "reject", d.check, d.reason))
    widths = [max(len(r[k]) for r in rows) for k in range(3)]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(r[:3], widths)) + "  " + r[3] for r in rows
    )


def main(out_dir: Path = DEFAULT_OUT_DIR, step_mbps: float = 0.5) -> Path:
    """Write the feasibility map, print the decision table. Returns the figure's path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "M03-feasibility.png"
    plot_feasibility_map(step_mbps).savefig(path, dpi=120)
    print(f"wrote {path}\n")
    print(decision_table())
    return path


if __name__ == "__main__":
    main()
