"""Can the deployed policy be calmed without retraining? Simulator test for the M07b oscillation.

  uv run python testbed/sim_stability_m07b.py

The champion on the simulator's demo surge, with its KPIs 2 s late (what matches the testbed,
testbed/sim_delay_m07b.py), under the settings that need no retraining: the smoothing filter's
alpha and the decision tick. Each variant runs with the default objective and with an operator
objective (URLLC >= 20 Mb/s, applied at t=150 s), three seeds each. It reports how much the split
moves AND what that costs, because a calmer split that serves the slices worse is no fix.
A second table times the reaction, which that window cannot show: how long URLLC takes to reach
20 Mb/s when the surge starts, its peak delay, and how long it keeps its PRBs after the surge ends.
Simulator only, no held-out data. Writes testbed/results/m07b/sim_stability.json.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from backends.sim import DelayedBackend, scenario_traffic  # noqa: E402
from rl.policy import SlicePolicy  # noqa: E402
from rl.runner import Runner, load_policy  # noqa: E402
from sim.cell import CellConfig  # noqa: E402
from sim.scenarios import DEMO_SURGE_END_S, DEMO_SURGE_START_S  # noqa: E402
from store import a1_policy, db, kpi  # noqa: E402
from verifier.objective import RLObjective, SliceValues  # noqa: E402

SECONDS = 420
WINDOW = (180.0, 420.0)  # inside the surge (from 120 s), after the objective settled
TARGET_MBPS = 20.0
SEEDS = [0, 1, 2]
# (label, filter alpha, decision tick s, KPIs held back by this many decisions: 2 s late overall)
VARIANTS = [
    ("no delay (training)", 0.25, 1, 0),
    ("2 s late, as deployed", 0.25, 1, 2),
    ("2 s late, alpha 0.15", 0.15, 1, 2),
    ("2 s late, alpha 0.10", 0.10, 1, 2),
    ("2 s late, alpha 0.05", 0.05, 1, 2),
    ("2 s late, tick 2 s", 0.25, 2, 1),
    ("2 s late, tick 2 s, alpha 0.5", 0.5, 2, 1),
]


def run(policy_file: Path, alpha: float, tick_s: int, delay: int, with_intent: bool, seed: int, tmp: Path) -> dict:
    cfg = CellConfig()
    conn = db.connect(tmp / f"run_{alpha}_{tick_s}_{delay}_{with_intent}_{seed}.db", timeout_s=0.0)
    backend = DelayedBackend(cfg, scenario_traffic("demo", cfg), seed=seed, delay=delay)
    kpi.write_backend_info(conn, backend.info(), "stability test")
    if with_intent:
        objective = RLObjective(weights=SliceValues(embb=3 / 14, urllc=8 / 14, be=3 / 14),
                                min_thr_mbps=SliceValues(urllc=TARGET_MBPS), valid_from=150.0, valid_until=600.0)
        a1_policy.write_schedule(conn, [a1_policy.Segment(objective, ("test",), {"accepted": True}, "console",
                                                          150.0, 600.0)], "test", 0.0)
    runner = Runner(backend, SlicePolicy.load(policy_file, alpha=alpha), conn, tick_s)
    runner.run(SECONDS // tick_s)
    out: dict = {}
    for s in ("embb", "urllc", "be"):
        rows = conn.execute("SELECT prb_share_alloc, dl_thr_mbps, latency_ms FROM kpi WHERE slice = ? AND ts >= ? "
                            "AND ts <= ? ORDER BY ts", (s, *WINDOW)).fetchall()
        a = np.array([tuple(r) for r in rows], dtype=float)
        out[s] = {"thr": a[:, 1].mean(), "lat_median": float(np.median(a[:, 2])), "share": a[:, 0].mean(),
                  "share_p5": float(np.percentile(a[:, 0], 5)), "share_p95": float(np.percentile(a[:, 0], 95)),
                  "moved_per_s": float(np.abs(np.diff(a[:, 0])).mean())}
        if s == "urllc":
            out["urllc_met"] = float((a[:, 1] >= TARGET_MBPS).mean())
    conn.close()
    return out


def reaction(policy_file: Path, alpha: float, seed: int, tmp: Path) -> tuple[float, float, float]:
    """(s until URLLC holds 20 Mb/s for 5 s, peak delay in the surge's first 60 s, s until its share
    is back under 0.25 after the surge), KPIs 2 s late, default objective."""
    cfg = CellConfig()
    conn = db.connect(tmp / f"reaction_{alpha}_{seed}.db", timeout_s=0.0)
    backend = DelayedBackend(cfg, scenario_traffic("demo", cfg), seed=seed, delay=2)
    kpi.write_backend_info(conn, backend.info(), "stability test")
    Runner(backend, SlicePolicy.load(policy_file, alpha=alpha), conn, 1).run(700)
    t, thr, lat, share = np.array([tuple(r) for r in conn.execute(
        "SELECT ts, dl_thr_mbps, latency_ms, prb_share_alloc FROM kpi WHERE slice = 'urllc' ORDER BY ts")], dtype=float).T
    conn.close()
    start, end = DEMO_SURGE_START_S, DEMO_SURGE_END_S
    reached = next(t[i] - start for i in np.where(t >= start)[0] if (thr[i:i + 5] >= TARGET_MBPS).all())
    return float(reached), float(lat[(t >= start) & (t < start + 60)].max()), float(t[(t >= end) & (share < 0.25)][0] - end)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    import os

    os.environ["RIC_DB"], os.environ["RIC_EVENTS"] = str(tmp / "ric.db"), str(tmp / "events.jsonl")
    _, label = load_policy(None)
    policy_file = Path("var/policy_cache") / json.loads(Path("var/policy_cache/served.json").read_text())["file"]
    results = []
    print(f"{label}; window {WINDOW} s; mean of {len(SEEDS)} seeds")
    print(f"{'variant':32s} {'objective':8s} {'moved/s':>7s} {'eMBB p5-p95':>12s} {'URLLC Mb/s':>10s} {'>=20':>5s} "
          f"{'URLLC lat':>10s} {'eMBB Mb/s':>9s} {'eMBB lat':>9s} {'BE Mb/s':>7s}")
    for name, alpha, tick_s, delay in VARIANTS:
        for with_intent in (False, True):
            runs = [run(policy_file, alpha, tick_s, delay, with_intent, seed, tmp) for seed in SEEDS]
            mean = lambda f: float(np.mean([f(r) for r in runs]))  # noqa: B023 (used within this iteration)
            row = {"variant": name, "alpha": alpha, "tick_s": tick_s, "delay_decisions": delay, "intent": with_intent,
                   "moved_per_s": mean(lambda r: r["embb"]["moved_per_s"]) / tick_s,
                   "embb_p5": mean(lambda r: r["embb"]["share_p5"]), "embb_p95": mean(lambda r: r["embb"]["share_p95"]),
                   "urllc_mbps": mean(lambda r: r["urllc"]["thr"]), "urllc_met": mean(lambda r: r["urllc_met"]),
                   "urllc_lat_ms": mean(lambda r: r["urllc"]["lat_median"]), "embb_mbps": mean(lambda r: r["embb"]["thr"]),
                   "embb_lat_ms": mean(lambda r: r["embb"]["lat_median"]), "be_mbps": mean(lambda r: r["be"]["thr"]),
                   "be_lat_ms": mean(lambda r: r["be"]["lat_median"])}
            results.append(row)
            print(f"{name:32s} {'intent' if with_intent else 'default':8s} {row['moved_per_s']:7.3f} "
                  f"{row['embb_p5']:5.2f}-{row['embb_p95']:<5.2f} {row['urllc_mbps']:10.1f} {row['urllc_met']:5.2f} "
                  f"{row['urllc_lat_ms']:8.0f}ms {row['embb_mbps']:9.1f} {row['embb_lat_ms']:7.0f}ms {row['be_mbps']:7.1f}")
    print(f"\n{'alpha':>6s} {'s to 20 Mb/s':>13s} {'peak delay ms':>14s} {'s to release':>13s}")
    reactions = []
    for alpha in (0.25, 0.15, 0.10, 0.05):
        reached, peak, release = np.mean([reaction(policy_file, alpha, seed, tmp) for seed in SEEDS], axis=0)
        reactions.append({"alpha": alpha, "seconds_to_target": reached, "peak_delay_ms": peak, "seconds_to_release": release})
        print(f"{alpha:6.2f} {reached:13.1f} {peak:14.0f} {release:13.1f}")
    Path("testbed/results/m07b/sim_stability.json").write_text(json.dumps(
        {"policy": label, "window_s": WINDOW, "seeds": SEEDS, "rows": results, "reaction": reactions}, indent=1))


if __name__ == "__main__":
    main()
