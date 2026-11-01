"""Figures for the simulator (M01).

Run with:  uv run python -m sim.plot
Writes two PNGs into docs/learn/img/ (embedded in docs/learn/M01-simulator.md):
- M01-model.png: the model itself, i.e. latency vs utilisation and throughput vs share
- M01-spike.png: the URLLC spike scenario under two fixed PRB splits, side by side

We use matplotlib's object API (Figure, not pyplot): no global state, and the same
functions work in a script, in tests and in the marimo notebook.
"""

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sim.cell import CellConfig, slice_latency_ms, step_slice
from sim.env import SlicingEnv
from sim.scenarios import SPIKE_END_S, SPIKE_START_S, urllc_spike
from sim.traffic import TrafficPattern
from verifier.objective import SLICES, RLObjective

# Colour follows the slice everywhere, in every figure of the project (dataviz reference
# palette, categorical slots 1-3: the only three that stay distinguishable for colour-blind
# readers in every pairing).
SLICE_COLOURS = {"embb": "#2a78d6", "urllc": "#eb6834", "be": "#1baf7a"}
SLICE_LABELS = {"embb": "eMBB", "urllc": "URLLC", "be": "Best effort"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e2e1dc"
LINE_WIDTH = 1.6  # points, about 2 px

# Two fixed splits to compare (eMBB, URLLC, BE).
FIXED_SPLITS = {
    "Equal split (1/3 each)": np.array([1 / 3, 1 / 3, 1 / 3]),
    "URLLC-protecting split (0.40 / 0.42 / 0.18)": np.array([0.40, 0.42, 0.18]),
}

DEFAULT_OUT_DIR = Path("docs/learn/img")


Controller = Callable[[np.ndarray], np.ndarray]  # observation -> action (3 numbers)


def run_episode(
    controller: Controller,
    traffic: dict[str, TrafficPattern],
    objective: RLObjective,
    steps: int = 30,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run one episode where `controller` picks the action from each observation.

    Any controller fits: a fixed split, a heuristic, or the trained RL policy, so they all
    produce the same arrays and the same plots.
    Keys: "time", "reward", and "<quantity>_<slice>" for load, thr, lat, share.
    """
    env = SlicingEnv(episode_steps=steps)
    obs, _ = env.reset(seed=seed, options={"objective": objective, "traffic": traffic})
    rows: dict[str, list[float]] = {"time": [], "reward": []}
    for _ in range(steps):
        obs, reward, _, _, info = env.step(np.asarray(controller(obs), dtype=np.float32))
        rows["time"].append(info["time_s"])
        rows["reward"].append(reward)
        for s in SLICES:
            rows.setdefault(f"load_{s}", []).append(info["load_mbps"][s])
            rows.setdefault(f"thr_{s}", []).append(info["throughput_mbps"][s])
            rows.setdefault(f"lat_{s}", []).append(info["latency_ms"][s])
            rows.setdefault(f"share_{s}", []).append(info["shares"][s])
    return {k: np.array(v) for k, v in rows.items()}


def run_fixed_split(
    shares: np.ndarray,
    traffic: dict[str, TrafficPattern],
    objective: RLObjective,
    steps: int = 30,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run the env with the same PRB split at every step (a controller that ignores obs)."""
    return run_episode(lambda _obs: shares, traffic, objective, steps, seed)


def _style(ax: Axes, ylabel: str) -> None:
    """Recessive axes: light grid behind the data, muted ticks, no top/right frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)


def _label_line_ends(ax: Axes, x: float, end_values: dict[str, float], min_gap: float = 4.0) -> None:
    """Direct labels at the right end of each slice's line, pushed apart so they never overlap.

    Labels are in text ink; the line next to each label carries the colour. min_gap is in
    data units (Mb/s here).
    """
    placed: list[float] = []
    for s, y in sorted(end_values.items(), key=lambda item: item[1]):
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(
            SLICE_LABELS[s], (x, y), xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=8, color=TEXT_MUTED, annotation_clip=False,
        )


def _shade_spike(ax: Axes) -> None:
    ax.axvspan(SPIKE_START_S, SPIKE_END_S, color="#efeee9", zorder=0)


def plot_spike_comparison(
    runs: dict[str, dict[str, np.ndarray]],
    objective: RLObjective | Mapping[str, RLObjective],
    urllc: TrafficPattern,
    what: str = "under two fixed PRB splits",
) -> Figure:
    """One column per run, four rows: throughput vs load, latency, PRB shares, reward.

    `objective` is either one objective for all runs, or one per run name (then each column
    draws its own URLLC latency bound).
    """
    fig = Figure(figsize=(12, 10), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(4, len(runs), sharex=True, sharey="row", squeeze=False)

    for col, (name, run) in enumerate(runs.items()):
        t = run["time"]
        ax_thr, ax_lat, ax_share, ax_rew = axes[:, col]
        ax_thr.set_title(name, color=TEXT, fontsize=10, loc="left")

        for s in SLICES:
            colour = SLICE_COLOURS[s]
            ax_thr.plot(t, run[f"load_{s}"], color=colour, linewidth=1.0, linestyle="--", alpha=0.8)
            ax_thr.plot(t, run[f"thr_{s}"], color=colour, linewidth=LINE_WIDTH, label=SLICE_LABELS[s])
            ax_lat.plot(t, run[f"lat_{s}"], color=colour, linewidth=LINE_WIDTH)
        _label_line_ends(ax_thr, t[-1], {s: run[f"thr_{s}"][-1] for s in SLICES})

        obj = objective[name] if isinstance(objective, Mapping) else objective
        if obj.max_latency_ms is not None:
            ax_lat.axhline(obj.max_latency_ms, color=TEXT_MUTED, linewidth=1.0, linestyle=":")
            ax_lat.annotate(
                f"URLLC bound {obj.max_latency_ms:.0f} ms", (t[0], obj.max_latency_ms),
                xytext=(0, 3), textcoords="offset points", fontsize=8, color=TEXT_MUTED,
            )
        ax_lat.set_yscale("log")

        # Stacked shares, with a thin surface-coloured edge as the gap between bands.
        ax_share.stackplot(
            t, *[run[f"share_{s}"] for s in SLICES],
            colors=[SLICE_COLOURS[s] for s in SLICES], edgecolor=SURFACE, linewidth=1.5,
        )
        ax_share.set_ylim(0, 1)

        ax_rew.axhline(0.0, color=GRID, linewidth=1.0)
        ax_rew.plot(t, run["reward"], color=TEXT_MUTED, linewidth=LINE_WIDTH)
        ax_rew.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=9)

        for ax in (ax_thr, ax_lat, ax_share, ax_rew):
            _shade_spike(ax)

        if col == 0:
            _style(ax_thr, "Mb/s  (solid: throughput,\ndashed: offered load)")
            _style(ax_lat, "latency (ms, log scale)")
            _style(ax_share, "PRB share")
            _style(ax_rew, "reward")
        else:
            for ax in (ax_thr, ax_lat, ax_share, ax_rew):
                _style(ax, "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False, fontsize=9)
    fig.suptitle(
        f"URLLC spike (shaded, {urllc.base_mbps:.1f} → {urllc.peak_mbps:.1f} Mb/s) {what}",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    return fig


def plot_model_curves(cfg: CellConfig | None = None) -> Figure:
    """The two formulas of the cell model, drawn: latency(ρ) and throughput(share)."""
    cfg = cfg or CellConfig()
    fig = Figure(figsize=(11, 4), facecolor=SURFACE, layout="constrained")
    ax_lat, ax_thr = fig.subplots(1, 2)

    # Left: latency vs utilisation, empty queue. cap fixed at 50 Mb/s, load varies.
    cap = 50.0
    rho = np.linspace(0.0, 1.2, 241)
    for s in SLICES:
        lat = [slice_latency_ms(cap, r * cap, 0.0, cfg.base_latency_ms[s], cfg) for r in rho]
        ax_lat.plot(rho, lat, color=SLICE_COLOURS[s], linewidth=LINE_WIDTH, label=SLICE_LABELS[s])
    ax_lat.axvline(cfg.max_utilisation, color=TEXT_MUTED, linewidth=1.0, linestyle=":")
    ax_lat.annotate(
        f"ρ capped at {cfg.max_utilisation}", (cfg.max_utilisation, 0.04), xytext=(-4, 0),
        xycoords=("data", "axes fraction"), textcoords="offset points", ha="right",
        fontsize=8, color=TEXT_MUTED,
    )
    ax_lat.set_yscale("log")
    ax_lat.set_title("Latency = base / (1 − ρ), empty queue", color=TEXT, fontsize=10, loc="left")
    _style(ax_lat, "latency (ms, log scale)")
    ax_lat.set_xlabel("utilisation ρ = load / slice capacity", color=TEXT_MUTED, fontsize=9)
    ax_lat.legend(frameon=False, fontsize=8)

    # Right: throughput vs PRB share for a slice offering 40 % of the cell, empty queue.
    load = 0.4 * cfg.capacity_mbps
    shares = np.linspace(0.0, 1.0, 101)
    thr = [step_slice(sh, load, 0.0, 2.0, cfg).throughput_mbps for sh in shares]
    ax_thr.plot(shares, thr, color=SLICE_COLOURS["embb"], linewidth=LINE_WIDTH)
    ax_thr.axhline(load, color=TEXT_MUTED, linewidth=1.0, linestyle=":")
    ax_thr.annotate(
        f"offered load {load:.0f} Mb/s", (1.0, load), xytext=(0, 3), textcoords="offset points",
        ha="right", fontsize=8, color=TEXT_MUTED,
    )
    ax_thr.set_title(
        f"Throughput = min(load, share × {cfg.capacity_mbps:.0f} Mb/s)", color=TEXT, fontsize=10, loc="left"
    )
    _style(ax_thr, "throughput (Mb/s)")
    ax_thr.set_xlabel("PRB share of the slice", color=TEXT_MUTED, fontsize=9)
    return fig


def main(out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    """Build both figures and save them as PNG. Returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traffic, objective = urllc_spike()
    runs = {name: run_fixed_split(shares, traffic, objective) for name, shares in FIXED_SPLITS.items()}
    figures = {
        "M01-model.png": plot_model_curves(),
        "M01-spike.png": plot_spike_comparison(runs, objective, traffic["urllc"]),
    }
    paths = []
    for filename, fig in figures.items():
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        paths.append(path)
        print(f"wrote {path}")
    return paths


if __name__ == "__main__":
    main()
