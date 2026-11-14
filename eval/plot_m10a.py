"""The figures of M10a: the radio channel per slice, and what it does to a controller sized on "target / C".

  uv run python -m eval.plot_m10a

Writes two PNGs into docs/learn/img/:
- M10a-channel.png: (a) the three slices' factor over one 360 s run (dev instance f1_00's channel); (b) the factor
  per second over 2,000 seeds; (c) each UE's mean factor over its run, over the same seeds. Mean 1 in (b) is by
  construction (sim/channel.py); (c) is the spread the told 79 Mb/s hides.
- M10a-formula.png: formula rev 4 (rl.baselines.FormulaPolicy, sized on the told capacity) on the even-indexed F1a dev
  instances, on m09 (no channel) and on m10 (same instance, same seed, plus the channel). (a) the instance whose URLLC
  UE has the lowest mean factor, URLLC throughput against what "met" asks; (b) the URLLC intent met per instance.
  Even indices only: the odd ones are kept unseen for M10b's checks (plan, DECISIONS).

An illustration, not a result: the formula is handed the intent's stated target and bound at the submission, with no
agent and no verifier (no implied fair-share targets, no LLM delay); the Runner applies its splits. M10b plays the
real path.
"""

import tempfile
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from backends.sim import DelayedBackend
from eval.generator import Instance, dev_instances
from eval.plot_control import IMG, NEUTRAL, titled
from eval.sim_profiles import KPI_DELAY_TICKS, channel_of, noise_std, testbed_cell
from rl.baselines import FormulaPolicy
from rl.runner import Runner
from sim import channel as ch
from sim.cell import CellConfig
from sim.env import OBJECTIVE_SLICE, objective_to_vector
from sim.plot import GRID, LINE_WIDTH, SLICE_COLOURS, SLICE_LABELS, SURFACE, TEXT, TEXT_MUTED
from store import db
from verifier.objective import DEFAULT_OBJECTIVE, SLICES, RLObjective, SliceValues

SEEDS = range(2000)
RUN_S = 360
F1A_EVEN = [i for i in dev_instances(["f1"], 20) if i.index % 2 == 0]  # F1a = indices 0-19 (eval.generator)
M09_COLOUR, M10_COLOUR = NEUTRAL, SLICE_COLOURS["urllc"]


class StatedObjective:
    """Formula rev 4 with the intent's stated objective inside its window, the default outside: the objective part of
    the runner's observation is replaced, nothing is written to a1_policy. The Runner still applies every split
    (hard invariant 4); no objective reaches the RL xApp, so nothing bypasses the verifier (invariant 3)."""

    def __init__(self, formula: FormulaPolicy, stated: RLObjective, window: tuple[float, float], backend, cfg) -> None:
        self.formula, self.window, self.backend = formula, window, backend
        self.stated = objective_to_vector(stated, cfg)

    def shares(self, obs: np.ndarray) -> np.ndarray:
        if self.window[0] <= self.backend.now() < self.window[1]:
            obs = obs.copy()
            obs[OBJECTIVE_SLICE] = self.stated
        return self.formula.shares(obs)


def replay_formula(inst: Instance, profile: str) -> dict[str, np.ndarray]:
    """Formula rev 4 on one instance, served by the Runner as in the benchmark (KPIs 2 ticks late, floor, no filter),
    with the intent's stated target and bound from its submission to the end of its window. URLLC's true values."""
    intent = inst.intents[0]
    cell = testbed_cell(profile, inst.seed)
    backend = DelayedBackend(cell, inst.traffic(noise_std()), inst.seed, KPI_DELAY_TICKS, channel=channel_of(profile, inst.seed))
    obs_cfg = CellConfig(n_prb=cell.n_prb, capacity_mbps=cell.told_capacity_mbps)  # what the runner builds from info()
    stated = RLObjective(weights=DEFAULT_OBJECTIVE.weights, min_thr_mbps=SliceValues(**intent.targets),
                         max_latency_ms=intent.max_latency_ms)
    policy = StatedObjective(FormulaPolicy(cell.told_urllc_base_latency_ms), stated, intent.window, backend, obs_cfg)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        runner = Runner(backend, policy, db.connect(Path(tmp) / "ric.db", timeout_s=0.0), cell.dt_s)
        for _ in range(RUN_S):
            result = runner.tick()
            last = backend._last["urllc"]  # the truth this second, as eval.metrics scores it
            rows.append((result.ts, last.throughput_mbps, backend._loads["urllc"], last.latency_ms))
        runner.conn.close()
    t, thr, load, lat = map(np.array, zip(*rows))
    target = intent.targets["urllc"]
    window = (t > intent.window[0]) & (t <= intent.window[1])
    met = (thr >= np.minimum(target, load) - 1e-9) & (lat <= intent.max_latency_ms)
    return {"t": t, "thr": thr, "load": load, "window": window, "missed": window & ~met, "met": float(met[window].mean())}


def plot_channel() -> Figure:
    fig = Figure(figsize=(12, 3.6), facecolor=SURFACE, layout="constrained")
    trace_ax, second_ax, mean_ax = fig.subplots(1, 3, width_ratios=[2.2, 1, 1])
    example = channel_of("m10", dev_instances(["f1"], 1)[0].seed)
    t = np.arange(RUN_S)
    label_y = 1.55  # direct labels stacked in the free space right of the run, top to bottom
    for i, s in enumerate(SLICES):
        trace_ax.plot(t, example.factors[:RUN_S, i], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH)
        trace_ax.plot([RUN_S + 8, RUN_S + 22], [label_y, label_y], color=SLICE_COLOURS[s], linewidth=LINE_WIDTH * 1.5)
        trace_ax.text(RUN_S + 28, label_y, f"{SLICE_LABELS[s]}\n{example.speed_kmh[i]:.0f} km/h, mean {example.mean_sinr_db[i]:.1f} dB",
                      va="center", fontsize=8, color=TEXT)
        label_y -= 0.4
    trace_ax.text(RUN_S + 8, 0.3, "dashed: 1.0, the rate\nof the told 79 Mb/s cell", va="center", fontsize=8,
                  color=TEXT_MUTED)
    trace_ax.set_xticks(range(0, RUN_S + 1, 60))
    trace_ax.axhline(1.0, color=TEXT_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    trace_ax.set_xlim(0, RUN_S * 1.42)
    trace_ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    titled(trace_ax, "(a) Mb/s per PRB / told, one run (instance f1_00)")

    traces = [ch.sample_channel(np.random.default_rng(seed), RUN_S) for seed in SEEDS]
    per_second = np.concatenate([tr.factors.ravel() for tr in traces])
    per_ue = np.concatenate([tr.factors.mean(axis=0) for tr in traces])
    for ax, values, title in ((second_ax, per_second, "(b) every second, 2,000 seeds"),
                              (mean_ax, per_ue, "(c) each UE's mean over its run")):
        ax.hist(values, bins=np.linspace(0, 1.7, 52), color=NEUTRAL, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(values.mean(), color=TEXT, linewidth=1.0)
        ax.annotate(f"mean {values.mean():.3f}\n5-95 %: {np.percentile(values, 5):.2f}-{np.percentile(values, 95):.2f}",
                    (0.03, 0.95), xycoords="axes fraction", va="top", fontsize=8, color=TEXT)
        ax.set_yticks([])
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
        ax.set_xlabel("factor on the told Mb/s per PRB", color=TEXT_MUTED, fontsize=8.5)
        titled(ax, title)
    return fig


def plot_formula() -> Figure:
    lowest = min(F1A_EVEN, key=lambda inst: channel_of("m10", inst.seed).factors[:RUN_S, SLICES.index("urllc")].mean())
    runs = {inst.name: {p: replay_formula(inst, p) for p in ("m09", "m10")} for inst in F1A_EVEN}
    fig = Figure(figsize=(12, 3.8), facecolor=SURFACE, layout="constrained")
    run_ax, met_ax = fig.subplots(1, 2, width_ratios=[2.2, 1])
    one = runs[lowest.name]
    intent = lowest.intents[0]
    run_ax.axvspan(*intent.window, color=GRID, alpha=0.6, linewidth=0,
                   label=f"intent: {intent.targets['urllc']:g} Mb/s, under {intent.max_latency_ms:g} ms")
    run_ax.plot(one["m09"]["t"], one["m09"]["load"], color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)),
                label="URLLC offered load")
    for k, (profile, colour) in enumerate((("m09", M09_COLOUR), ("m10", M10_COLOUR))):
        r = one[profile]
        run_ax.plot(r["t"], r["thr"], color=colour, linewidth=LINE_WIDTH,
                    label=f"{profile}: URLLC throughput (intent met {r['met']:.2f})")
        missed = r["t"][r["missed"]]  # a tick per second the intent was missed, under the curves
        run_ax.scatter(missed, np.full(len(missed), -4.0 - 3.0 * k), marker="|", s=30, color=colour, linewidths=1)
    run_ax.set_ylabel("Mb/s (ticks below 0: seconds missed)", color=TEXT_MUTED, fontsize=8.5)
    run_ax.set_xlabel("time (s)", color=TEXT_MUTED, fontsize=8.5)
    run_ax.legend(frameon=False, fontsize=8, loc="upper left")
    factor = channel_of("m10", lowest.seed).factors[:RUN_S, SLICES.index("urllc")].mean()
    titled(run_ax, f"(a) {lowest.name}, formula rev 4: URLLC's UE at {factor:.2f} of the told rate on average")

    names = [inst.name for inst in F1A_EVEN]
    x = np.arange(len(names))
    for offset, profile, colour in ((-0.13, "m09", M09_COLOUR), (0.13, "m10", M10_COLOUR)):
        met_ax.scatter(x + offset, [runs[n][profile]["met"] for n in names], s=40, color=colour, label=profile, zorder=3)
    for k, n in enumerate(names):
        met_ax.plot([k - 0.13, k + 0.13], [runs[n]["m09"]["met"], runs[n]["m10"]["met"]], color=GRID, linewidth=1, zorder=1)
    met_ax.set_xticks(x, [n.replace("f1_", "") for n in names], fontsize=8)
    met_ax.set_ylim(-0.03, 1.03)
    met_ax.set_xlabel("F1a dev instance (even indices)", color=TEXT_MUTED, fontsize=8.5)
    met_ax.legend(frameon=False, fontsize=8, loc="lower left")
    titled(met_ax, "(b) URLLC intent met, formula rev 4")
    return fig


def main(out_dir: Path = IMG) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in (("M10a-channel.png", plot_channel()), ("M10a-formula.png", plot_formula())):
        fig.savefig(out_dir / name, dpi=150, facecolor=SURFACE)
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
