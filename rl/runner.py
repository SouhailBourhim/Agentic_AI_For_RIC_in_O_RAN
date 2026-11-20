"""The RL runner: the "RL xApp" process of figure 2, the fast control loop.

Every tick (1 s of backend time):
1. read the a1_policy segment in force now (store/a1_policy.py); default objective if none;
2. build the observation from the last KPI records + that objective, exactly as in training;
3. ask the policy for a PRB split (or, in direct_share mode, take the row's shares);
4. enforce the per-slice minimum share (5 %, sim.cell.apply_min_share), the same rule the policy
   was trained with, so the floor holds exactly whatever the policy outputs;
5. apply it through the Backend: THE ONLY CALL TO apply_prb_split IN THE PROJECT (invariant 4);
6. let one tick pass, then write the KPI records, tagged with the policy_id in force.

It never waits on anything else (hard invariant 5): no LLM, no agent, no HTTP, no MLflow in the
loop. The database is opened with a zero busy-timeout. A read that fails keeps the last
objective, and KPI records that cannot be written are kept and written at the next tick.

Safe state (M11a). If the KPIs are stale (no report for SAFE_KPI_AGE_S), the actuator is down (the
slice-control xApp died) or the backend or the policy raises, the runner stops asking the policy and
applies SAFE_SPLIT, the static controller's equal split, at once. It logs safe_mode_enter, writes the
runner_state row that get_status and Prometheus show, and goes back to the policy after
RESUME_AFTER_TICKS clean ticks in a row. On the simulator the backend is always healthy, so this
never happens there and every earlier run replays unchanged.

Runtime monitor (M11b, rl/monitor.py). With a FallbackMonitor (on in the served runner, off in eval.harness unless a
run asks for it), the formula executor runs in shadow; if a target in force is missed long enough while the formula
would meet it (its Trigger), the formula's split is applied until the a1_policy segment ends. Logged (executor_fallback,
executor_resume) and shown as runner_state mode 'fallback'. The safe state still wins over it.

Run (the simulated cell lives inside this process):
  uv run python -m rl.runner --scenario demo --speed 1          live demo, one tick per second
  uv run python -m rl.runner --scenario urllc_spike --speed 0 --ticks 60
On the testbed (M07b), inside the RL xApp container (xapps/rl/), with var/ric-testbed.db:
  python -m rl.runner --backend flexric --policy-file /app/model/<file>.zip --model-label "..."
"""

import argparse
import dataclasses
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backends.base import Backend, BackendHealth, KpiRecord
from backends.flexric import FlexRICBackend
from backends.sim import SimBackend, scenario_traffic
from rl.baselines import EqualSplitPolicy, FormulaEstPolicy, FormulaPolicy
from rl.monitor import FallbackMonitor
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from sim.cell import CellConfig, apply_min_share
from sim.env import EfficiencyTracker, LoadHistory, applied_features, build_observation
from store import a1_policy, db, kpi
from store.events import log_event
from verifier.objective import DEFAULT_OBJECTIVE, MIN_SLICE_SHARE, SLICES, RLObjective
from xapps.kpm_monitor import DEFAULT_TESTBED_DB, cell_info

POLICY_CACHE = Path("var/policy_cache")
# The filter on the served split, stronger than the one the policy was trained and first evaluated with
# (SMOOTHING_ALPHA). The testbed's late KPIs make the split swing at alpha 0.25 (M07b), and the policy
# jitters on the simulator too (M08a: 5.6-8.5 % of the cell per s, 1.8-3.2 % at this value). No retraining.
SERVED_SMOOTHING_ALPHA = 0.10
MAX_PENDING_TICKS = 600  # KPI ticks kept in memory while the database is busy (10 min at 1 s)
# M11a, the safe state. The equal split is the static controller's: it needs no KPI and no objective, and every
# slice keeps a third of the cell, far above the 5 % floor. Not "the last good split": the policy chose it on
# data that may already have been going bad. Not the formula: it needs fresh KPIs too (that fallback is M11b).
SAFE_SPLIT = np.full(len(SLICES), 1.0 / len(SLICES))
SAFE_KPI_AGE_S = 5.0  # five missed 1 s KPM reports, FlexRIC's own kpm_stale threshold (backends/flexric.STALE_S)
RESUME_AFTER_TICKS = 3  # clean ticks in a row before the policy decides again: one fresh report is not a recovery


def load_policy(policy_file: Path | None = None, alpha: float = SERVED_SMOOTHING_ALPHA) -> tuple[SlicePolicy, str]:
    """The policy to serve, loaded ONCE at startup, and a label of its version.

    Order: an explicit file (tests, CI); else the MLflow registry's slice-policy@champion,
    copied to var/policy_cache/; else, if MLflow is unavailable, the last cached champion.
    The deployed smoothing filter is on, on either backend.
    """
    if policy_file is not None:
        return SlicePolicy.load(policy_file, alpha), f"file {policy_file.name}"
    served = POLICY_CACHE / "served.json"
    try:
        from mlops import registry  # imported here: only needed at startup, and MLflow is heavy

        found = registry.champion()
        version = registry.alias_version(registry.CHAMPION)
        if found is None or version is None:
            raise LookupError("no champion in the MLflow registry")
        name, path = found
        POLICY_CACHE.mkdir(parents=True, exist_ok=True)
        cached = POLICY_CACHE / path.name
        shutil.copy(path, cached)
        label = f"{registry.MODEL_NAME} v{version.version} ({name})"
        served.write_text(json.dumps({"label": label, "file": cached.name}))
    except Exception as err:  # MLflow missing, broken or empty: fall back to the cache
        if not served.exists():
            raise SystemExit(f"no policy: MLflow failed ({err}) and {served} does not exist; use --policy-file") from err
        cache = json.loads(served.read_text())
        label, cached = cache["label"] + " [cached]", POLICY_CACHE / cache["file"]
    return SlicePolicy.load(cached, alpha), label


# M10d: where the demand the controller sees comes from. Every load it reads goes through seen_loads: the
# observation, the load history and so the formula too, which reads the observation.
DEMANDS = ("offered", "measured", "throughput")


def seen_loads(records: list[KpiRecord], demand: str = "offered") -> dict[str, float]:
    """The load per slice as the controller sees it; the throughput when the chosen source has no value.

    offered: offered_load_mbps, as before M10d (the simulator's true arrivals; on the testbed the iperf3 rate);
    measured: measured_load_mbps, what the gNB measured (KPM DRB.PdcpSduVolumeDL, M10d);
    throughput: the throughput only, M07b's "blind" run: a slice asks at least what it got.
    """
    if demand not in DEMANDS:
        raise ValueError(f"demand is one of {DEMANDS}, got {demand!r}")
    field = {"offered": "offered_load_mbps", "measured": "measured_load_mbps", "throughput": "dl_thr_mbps"}[demand]
    by_slice = {r.slice: r for r in records}
    loads = {}
    for s in SLICES:
        value = getattr(by_slice[s], field)
        loads[s] = value if value is not None else by_slice[s].dl_thr_mbps
    return loads


def observation_from_records(records: list[KpiRecord], objective: RLObjective, cfg: CellConfig,
                             history: LoadHistory | None = None, applied: np.ndarray | None = None,
                             efficiency: EfficiencyTracker | None = None, demand: str = "offered") -> np.ndarray:
    """The policy input built from one tick of KPI records (same function as in training).

    Fields a backend cannot measure are replaced conservatively: an unknown offered load by
    the throughput (the slice asks at least what it got), an unknown latency by 0. Both are
    known in the simulator; the testbed mapping is in docs/PLAN.md section 3.
    With a LoadHistory (M09c), its 3 features follow the objective, as in SlicingEnv(history="on");
    a policy trained without them reads only the first 22 values (rl.policy.SlicePolicy).
    With the split applied at the last tick (M09h), its 3 values follow the history, as in
    SlicingEnv(applied_split=True): the runner knows it now, while the KPIs' own prb_share is as late as they are.
    With an EfficiencyTracker (M10a), its 6 features follow the applied split, as in
    SlicingEnv(efficiency_features=True).
    """
    by_slice = {r.slice: r for r in records}
    obs = build_observation(
        loads_mbps=seen_loads(records, demand),
        throughput_mbps={s: by_slice[s].dl_thr_mbps for s in SLICES},
        latency_ms={s: by_slice[s].latency_ms or 0.0 for s in SLICES},
        shares={s: by_slice[s].prb_share_alloc for s in SLICES},
        prb_used={s: by_slice[s].prb_used for s in SLICES},
        objective=objective,
        cfg=cfg,
    )
    if history is None:
        return obs
    parts = [obs, history.features(cfg.capacity_mbps)]
    if applied is not None:
        parts.append(applied_features(applied))
    if efficiency is not None:
        if applied is None:
            raise ValueError("the efficiency features go after the applied split: pass `applied` too")
        parts.append(efficiency.features())
    return np.concatenate(parts).astype(np.float32)


def health_problem(health: BackendHealth) -> str | None:
    """Why the runner should not trust this tick, or None. A dead actuator first: nothing reaches the radio."""
    if not health.actuator_ok:
        return "actuator_down"
    if health.kpi_age_s is None or health.kpi_age_s > SAFE_KPI_AGE_S:
        return "kpi_stale"
    return None


def update_efficiency(tracker: EfficiencyTracker, records: list[KpiRecord], capacity_mbps: float) -> None:
    """Feed the tracker one tick of KPI records: the same three fields SlicingEnv feeds it from its late view."""
    by_slice = {r.slice: r for r in records}
    tracker.update({s: by_slice[s].dl_thr_mbps for s in SLICES}, {s: by_slice[s].prb_share_alloc for s in SLICES},
                   {s: by_slice[s].prb_used for s in SLICES}, capacity_mbps)


@dataclass(frozen=True)
class TickResult:
    ts: float  # backend clock at the end of the tick
    policy_id: int | None  # a1_policy row in force (None = default objective)
    shares: tuple[float, ...]
    db_ok: bool  # False if the policy read or the KPI write had to be skipped this tick
    mode: str  # 'normal', 'safe' (M11a) or 'fallback' (M11b: the formula executes the segment)


class Runner:
    """The control loop, separate from the CLI so tests can drive it tick by tick."""

    def __init__(self, backend: Backend, policy: SlicePolicy, conn: sqlite3.Connection, dt_s: float = 1.0,
                 on_error: str = "safe", demand: str = "offered", monitor: FallbackMonitor | None = None) -> None:
        """on_error: 'safe' (served runner) turns an exception into the safe state; 'raise' (evaluation runs,
        eval.harness) lets it propagate, so a bug ends the run instead of quietly playing the equal split.
        demand: where the controller's loads come from (seen_loads, M10d).
        monitor: the runtime monitor with formula fallback (M11b); None = the policy alone, as before M11b."""
        if on_error not in ("safe", "raise"):
            raise ValueError(f"on_error is 'safe' or 'raise', got {on_error!r}")
        if demand not in DEMANDS:
            raise ValueError(f"demand is one of {DEMANDS}, got {demand!r}")
        self.demand = demand
        self.backend = backend
        self.policy = policy
        self.conn = conn
        self.dt_s = dt_s
        info = backend.info()
        self.cfg = CellConfig(n_prb=info.n_prb, capacity_mbps=info.capacity_mbps, dt_s=dt_s)
        self.history = LoadHistory(dt_s=dt_s)  # the policy's load history (M09c), fed every tick
        # M10a: measured Mb/s per PRB per slice, fed every tick like the history. Always in the observation (34 values):
        # every policy reads the prefix it was trained on, so earlier ones see exactly what they saw before.
        self.efficiency = EfficiencyTracker()
        self.applied = np.full(len(SLICES), 1.0 / len(SLICES))  # the split applied last tick (M09h), as at env reset
        self.row: a1_policy.PolicyRow | None = None  # last segment read successfully
        self.last_policy_id: int | None = None  # policy in force at the previous tick
        self.pending: list[tuple[list[KpiRecord], int | None]] = []  # KPI ticks not written yet
        self.on_error = on_error
        self.policy_invalid_logged = False  # M11c: a corrupt a1_policy row is logged once, not every tick
        self.mode = "normal"  # M11a: 'normal' (the policy decides) or 'safe' (SAFE_SPLIT)
        self.reason: str | None = None  # why safe
        self.since = backend.now()  # backend clock when the current mode started
        self.clean_ticks = 0  # clean ticks in a row while safe
        self.apply_error: Exception | None = None  # the last apply_prb_split failed: next tick is an error
        self.pending_state: tuple[str, str | None, float] | None = ("normal", None, self.since)  # written at flush
        self.monitor = monitor
        self.executor = "rl"  # M11b: 'rl' (the policy decides) or 'formula' (the monitor's fallback, this segment)
        self.fallback_reason: str | None = None
        self.fallback_since = self.since

    def read_policy(self, now: float) -> bool:
        """Refresh the segment in force. On any database error keep the previous one.

        M11c: a row that can't be parsed (its objective fails RLObjective's schema, or its JSON is broken) is
        treated the same way, so a corrupt a1_policy row can't end the runner. pydantic's ValidationError and
        json's JSONDecodeError are both ValueErrors. Logged once until a read succeeds again."""
        try:
            self.row = a1_policy.in_force(self.conn, now)
        except sqlite3.Error:
            return False
        except ValueError as err:
            if not self.policy_invalid_logged:
                log_event("runner", "a1_policy_invalid", ts_backend=now,
                          data={"error": f"{type(err).__name__}: {str(err).splitlines()[0]}",
                                "kept_policy_id": self.row.policy_id if self.row is not None else None})
                self.policy_invalid_logged = True
            return False
        self.policy_invalid_logged = False
        return True

    def tick(self) -> TickResult:
        now = self.backend.now()
        read_ok = self.read_policy(now)
        row = self.row
        if row is not None and row.expires_at is not None and now >= row.expires_at:
            row = None  # the database was unreadable past the segment's end: back to the default
        objective = row.objective if row is not None else DEFAULT_OBJECTIVE
        policy_id = row.policy_id if row is not None else None
        if self.monitor is not None and self.monitor.segment(policy_id):
            self.end_fallback(now, policy_id)  # a new segment: the policy decides again

        shares = SAFE_SPLIT
        try:
            records = self.backend.read_kpis()
            self.history.update(seen_loads(records, self.demand))  # every tick, whoever sets the split
            update_efficiency(self.efficiency, records, self.cfg.capacity_mbps)
            problem = "error" if self.apply_error is not None else health_problem(self.backend.health())
            if self.update_mode(problem, now, self.apply_error) == "normal":
                if row is not None and row.mode == "direct_share" and row.shares is not None:
                    shares = np.asarray(row.shares, dtype=np.float64)
                    if self.monitor is not None:
                        self.monitor.hold()  # LLM only's own split: not the policy's, not monitored
                else:
                    obs = observation_from_records(records, objective, self.cfg, self.history, self.applied,
                                                   self.efficiency, self.demand)
                    shares = self.policy.shares(obs)
                    if self.monitor is not None:
                        formula = self.monitor.step(obs, objective, records, seen_loads(records, self.demand), shares)
                        if formula is not None:
                            if self.monitor.fired is not None:
                                self.start_fallback(now, row)
                            shares = formula
            elif self.monitor is not None:
                self.monitor.hold()  # the safe split: the policy isn't deciding
        except Exception as err:  # the policy or the backend failed: the safe split, never a dead loop
            if self.on_error == "raise":
                raise
            self.update_mode("error", now, err)
            shares = SAFE_SPLIT
        self.apply_error = None
        shares = apply_min_share(shares, MIN_SLICE_SHARE)  # always the last step before applying
        try:
            self.backend.apply_prb_split([float(x) for x in shares])
        except Exception as err:
            if self.on_error == "raise":
                raise
            self.apply_error = err  # the next tick enters (or stays in) the safe state with reason 'error'
        self.applied = np.asarray(shares, dtype=np.float64)
        self.backend.sleep(self.dt_s)

        if policy_id != self.last_policy_id:  # every change is logged, including expiries
            log_event("runner", "policy_switch", ts_backend=now,
                      data={"from": self.last_policy_id, "to": policy_id,
                            "intent_ids": list(row.intent_ids) if row is not None else []})
            self.last_policy_id = policy_id
        try:
            self.pending.append((self.backend.read_kpis(), policy_id))
        except Exception:
            if self.on_error == "raise":
                raise
            # Nothing to record for this tick. The next tick's read meets the same failure and enters the safe state.
        write_ok = self.flush()
        return TickResult(self.backend.now(), policy_id, tuple(float(x) for x in shares), read_ok and write_ok,
                          self.shown_mode())

    def update_mode(self, problem: str | None, now: float, error: Exception | None = None) -> str:
        """Enter the safe state at once on a problem; leave it after RESUME_AFTER_TICKS clean ticks in a row.

        Every change (and a change of reason while safe) is logged and queued for the runner_state row."""
        if problem is not None:
            self.clean_ticks = 0
            if self.mode == "safe" and problem == self.reason:
                return self.mode
            data: dict = {"reason": problem, "previous_reason": self.reason}
            if error is not None:
                data["error"] = f"{type(error).__name__}: {error}"
            log_event("runner", "safe_mode_enter", ts_backend=now, data=data)
            if self.mode == "normal":
                self.since = now
            self.mode, self.reason = "safe", problem
            self.queue_state()
            return self.mode
        if self.mode == "normal":
            return self.mode
        self.clean_ticks += 1
        if self.clean_ticks < RESUME_AFTER_TICKS:
            return self.mode
        log_event("runner", "safe_mode_exit", ts_backend=now,
                  data={"reason": self.reason, "safe_s": round(now - self.since, 3)})
        self.mode, self.reason, self.since, self.clean_ticks = "normal", None, now, 0
        self.queue_state()
        if isinstance(self.policy, SlicePolicy):
            # The filter restarts from the split in force, not from its last output before the outage:
            # otherwise the first resumed split jumps straight back to a decision made on old data.
            self.policy.restart_from(self.applied)
        return self.mode

    def shown_mode(self) -> str:
        """What get_status and Prometheus show: the safe state first, then the formula's fallback, else normal."""
        if self.mode == "safe":
            return "safe"
        return "fallback" if self.executor == "formula" else "normal"

    def queue_state(self) -> None:
        """Queue the runner_state row for the next flush, from the shown mode."""
        if self.mode == "safe":
            self.pending_state = ("safe", self.reason, self.since)
        elif self.executor == "formula":
            self.pending_state = ("fallback", self.fallback_reason, self.fallback_since)
        else:
            self.pending_state = ("normal", None, self.since)

    def start_fallback(self, now: float, row: a1_policy.PolicyRow | None) -> None:
        """M11b: the monitor has just handed this segment to the formula. Logged with why, and shown."""
        fired = self.monitor.fired
        self.executor, self.fallback_since = "formula", now
        self.fallback_reason = f"target_missed:{fired.slice}"
        log_event("runner", "executor_fallback", ts_backend=now,
                  data={"policy_id": row.policy_id if row is not None else None,
                        "intent_ids": list(row.intent_ids) if row is not None else [],
                        "slice": fired.slice, "reason": fired.reason, "missed_s": fired.missed_s,
                        "rl_share": round(fired.rl_share, 4), "formula_share": round(fired.formula_share, 4)})
        self.queue_state()

    def end_fallback(self, now: float, policy_id: int | None) -> None:
        """M11b: the segment the formula executed has ended; the policy decides again. Its filter restarts from the
        split in force, as after the safe state (M11a), so its first split doesn't jump back to an old decision."""
        log_event("runner", "executor_resume", ts_backend=now,
                  data={"policy_id": policy_id, "fallback_s": round(now - self.fallback_since, 3)})
        self.executor, self.fallback_reason = "rl", None
        if self.mode != "safe":
            self.since = now
        if isinstance(self.policy, SlicePolicy):
            self.policy.restart_from(self.applied)
        self.queue_state()

    def flush(self) -> bool:
        """Write every pending KPI tick, and the runner state if it changed, in one transaction;
        keep them if the database is busy."""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            for records, policy_id in self.pending:
                kpi.write_records(self.conn, records, policy_id)
            if self.pending_state is not None:
                kpi.write_runner_state(self.conn, *self.pending_state)
            self.conn.execute("COMMIT")
        except sqlite3.Error:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            del self.pending[:-MAX_PENDING_TICKS]  # bounded memory: drop the oldest if it lasts
            return False
        self.pending.clear()
        self.pending_state = None
        return True

    def run(self, ticks: int | None = None) -> None:
        """Run `ticks` ticks, or forever if None."""
        done = 0
        while ticks is None or done < ticks:
            self.tick()
            done += 1


def clear_run_state(conn: sqlite3.Connection) -> None:
    """Empty the tables at the start of a simulated run: its clock restarts at 0, so KPI rows,
    timetables and intents of a previous run would refer to the wrong times. The JSONL event
    log is kept (it is the record of every run)."""
    conn.execute("BEGIN IMMEDIATE")
    for table in ("kpi", "a1_policy", "a1_schedule", "intents", "backend_info", "runner_state"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute("DELETE FROM sqlite_sequence")  # policy ids restart at 1
    conn.execute("COMMIT")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL xApp: the PRB-split control loop")
    parser.add_argument("--backend", choices=["sim", "flexric"], default="sim")
    parser.add_argument("--scenario", default="demo", help="sim: steady | urllc_spike | demo")
    parser.add_argument("--speed", type=float, default=1.0, help="sim: 1 = real time, 0 = as fast as possible")
    parser.add_argument("--model-label", default=None, help="version label of --policy-file (the RL xApp image's)")
    parser.add_argument("--smoothing-alpha", type=float, default=SERVED_SMOOTHING_ALPHA,
                        help=f"filter on the split; {SMOOTHING_ALPHA} is what training used")
    parser.add_argument("--baseline", choices=["equal", "formula"], default=None,
                        help="no RL: 'equal', an equal split unless a direct_share segment is in force (M08 baselines); "
                             "'formula', LLM + formula's executor (rl.baselines.FormulaPolicy, M09b; on the testbed in M09e)")
    parser.add_argument("--urllc-base-latency-ms", type=float, default=None,
                        help="flexric: the URLLC base latency told to the verifier and the formula, instead of "
                             "cell_info's (M09e tells the M09a calibrated value, as the M09 simulator profile does)")
    parser.add_argument("--demand", choices=DEMANDS, default="offered",
                        help="the load the controller sees (M10d): offered (default; on the testbed the iperf3 rate), "
                             "measured (the gNB's PDCP volume), throughput (none: M07b's blind run)")
    parser.add_argument("--ticks", type=int, default=None, help="stop after N ticks (default: run forever)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-file", type=Path, default=None, help="serve this SB3 file instead of the champion")
    parser.add_argument("--keep-db", action="store_true", help="do not empty var/ric.db at startup")
    parser.add_argument("--no-monitor", action="store_true",
                        help="no runtime monitor (M11b): the policy's split always stands, even on a target it keeps missing")
    args = parser.parse_args()

    alpha = args.smoothing_alpha
    cfg = CellConfig()
    if args.backend == "flexric":
        if "RIC_DB" not in os.environ:  # never the simulator's var/ric.db: the clocks differ
            os.environ["RIC_DB"] = str(DEFAULT_TESTBED_DB)
        info = cell_info()
        if args.urllc_base_latency_ms is not None:
            info = dataclasses.replace(info, urllc_base_latency_ms=args.urllc_base_latency_ms)
        backend = FlexRICBackend(db.connect(timeout_s=0.0), info)
        args.scenario, args.speed = "testbed", 1.0
    else:
        backend = SimBackend(cfg, scenario_traffic(args.scenario, cfg), seed=args.seed, speed=args.speed)
    if args.baseline == "equal":
        policy, model_version = EqualSplitPolicy(), "baseline: equal split, no RL"
    elif args.baseline == "formula":  # the same base latency the verifier is told, as eval.harness does
        policy, model_version = FormulaPolicy(backend.info().urllc_base_latency_ms), "baseline: LLM + formula, no RL"
    else:
        policy, model_version = load_policy(args.policy_file, alpha)
    if args.model_label is not None and args.baseline is None:  # a baseline serves no model
        model_version = args.model_label
    monitor = None
    if args.baseline is None and not args.no_monitor:  # the same base latency the verifier is told, as for --baseline
        monitor = FallbackMonitor(FormulaEstPolicy(backend.info().urllc_base_latency_ms), backend.info().n_prb, cfg.dt_s)
    setup = db.connect()  # startup may wait for other processes; the loop below never does
    if not args.keep_db:
        clear_run_state(setup)
    kpi.write_backend_info(setup, backend.info(), model_version)
    setup.close()
    conn = db.connect(timeout_s=0.0)
    log_event("runner", "run_start", ts_backend=backend.now(),
              data={"model_version": model_version, "scenario": args.scenario, "speed": args.speed, "alpha": alpha,
                    "demand": args.demand, "monitor": monitor.trigger.name if monitor is not None else None})
    print(f"RL runner: {model_version}, scenario {args.scenario}, speed {args.speed}, db {db.db_path()}")
    try:
        Runner(backend, policy, conn, cfg.dt_s, demand=args.demand, monitor=monitor).run(args.ticks)
    except KeyboardInterrupt:
        pass
    if isinstance(backend, FlexRICBackend):
        backend.close()
    log_event("runner", "run_stop", ts_backend=backend.now())


if __name__ == "__main__":
    main()
