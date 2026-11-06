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

Run (the simulated cell lives inside this process):
  uv run python -m rl.runner --scenario demo --speed 1          live demo, one tick per second
  uv run python -m rl.runner --scenario urllc_spike --speed 0 --ticks 60
On the testbed (M07b), inside the RL xApp container (xapps/rl/), with var/ric-testbed.db:
  python -m rl.runner --backend flexric --policy-file /app/model/<file>.zip --model-label "..."
"""

import argparse
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backends.base import Backend, KpiRecord
from backends.flexric import FlexRICBackend
from backends.sim import SimBackend, scenario_traffic
from rl.policy import SMOOTHING_ALPHA, SlicePolicy
from sim.cell import CellConfig, apply_min_share
from sim.env import build_observation
from store import a1_policy, db, kpi
from store.events import log_event
from verifier.objective import DEFAULT_OBJECTIVE, MIN_SLICE_SHARE, SLICES, RLObjective
from xapps.kpm_monitor import DEFAULT_TESTBED_DB, cell_info

POLICY_CACHE = Path("var/policy_cache")
# The testbed shows the policy its KPIs about 2 s later than training did, and at alpha 0.25 the split
# swings (docs/learn/M07b-slice-control.md). No retraining: a stronger filter on the testbed only.
TESTBED_SMOOTHING_ALPHA = 0.10
MAX_PENDING_TICKS = 600  # KPI ticks kept in memory while the database is busy (10 min at 1 s)


def load_policy(policy_file: Path | None = None, alpha: float = SMOOTHING_ALPHA) -> tuple[SlicePolicy, str]:
    """The policy to serve, loaded ONCE at startup, and a label of its version.

    Order: an explicit file (tests, CI); else the MLflow registry's slice-policy@champion,
    copied to var/policy_cache/; else, if MLflow is unavailable, the last cached champion.
    The deployed smoothing filter is on: alpha 0.25 on the simulator, as in evaluation, and
    TESTBED_SMOOTHING_ALPHA on the testbed (see main).
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
            raise SystemExit(f"no policy: MLflow failed ({err}) and {served} does not exist; use --policy-file")
        cache = json.loads(served.read_text())
        label, cached = cache["label"] + " [cached]", POLICY_CACHE / cache["file"]
    return SlicePolicy.load(cached, alpha), label


def observation_from_records(records: list[KpiRecord], objective: RLObjective, cfg: CellConfig) -> np.ndarray:
    """The policy input built from one tick of KPI records (same function as in training).

    Fields a backend cannot measure are replaced conservatively: an unknown offered load by
    the throughput (the slice asks at least what it got), an unknown latency by 0. Both are
    known in the simulator; the testbed mapping is in docs/PLAN.md section 3.
    """
    by_slice = {r.slice: r for r in records}
    return build_observation(
        loads_mbps={s: by_slice[s].offered_load_mbps if by_slice[s].offered_load_mbps is not None
                    else by_slice[s].dl_thr_mbps for s in SLICES},
        throughput_mbps={s: by_slice[s].dl_thr_mbps for s in SLICES},
        latency_ms={s: by_slice[s].latency_ms or 0.0 for s in SLICES},
        shares={s: by_slice[s].prb_share_alloc for s in SLICES},
        prb_used={s: by_slice[s].prb_used for s in SLICES},
        objective=objective,
        cfg=cfg,
    )


@dataclass(frozen=True)
class TickResult:
    ts: float  # backend clock at the end of the tick
    policy_id: int | None  # a1_policy row in force (None = default objective)
    shares: tuple[float, ...]
    db_ok: bool  # False if the policy read or the KPI write had to be skipped this tick


class Runner:
    """The control loop, separate from the CLI so tests can drive it tick by tick."""

    def __init__(self, backend: Backend, policy: SlicePolicy, conn: sqlite3.Connection, dt_s: float = 1.0) -> None:
        self.backend = backend
        self.policy = policy
        self.conn = conn
        self.dt_s = dt_s
        info = backend.info()
        self.cfg = CellConfig(n_prb=info.n_prb, capacity_mbps=info.capacity_mbps, dt_s=dt_s)
        self.row: a1_policy.PolicyRow | None = None  # last segment read successfully
        self.last_policy_id: int | None = None  # policy in force at the previous tick
        self.pending: list[tuple[list[KpiRecord], int | None]] = []  # KPI ticks not written yet

    def read_policy(self, now: float) -> bool:
        """Refresh the segment in force. On any database error keep the previous one."""
        try:
            self.row = a1_policy.in_force(self.conn, now)
            return True
        except sqlite3.Error:
            return False

    def tick(self) -> TickResult:
        now = self.backend.now()
        read_ok = self.read_policy(now)
        row = self.row
        if row is not None and row.expires_at is not None and now >= row.expires_at:
            row = None  # the database was unreadable past the segment's end: back to the default
        objective = row.objective if row is not None else DEFAULT_OBJECTIVE

        if row is not None and row.mode == "direct_share" and row.shares is not None:
            shares = np.asarray(row.shares, dtype=np.float64)
        else:
            obs = observation_from_records(self.backend.read_kpis(), objective, self.cfg)
            shares = self.policy.shares(obs)
        shares = apply_min_share(shares, MIN_SLICE_SHARE)  # always the last step before applying
        self.backend.apply_prb_split([float(x) for x in shares])
        self.backend.sleep(self.dt_s)

        policy_id = row.policy_id if row is not None else None
        if policy_id != self.last_policy_id:  # every change is logged, including expiries
            log_event("runner", "policy_switch", ts_backend=now,
                      data={"from": self.last_policy_id, "to": policy_id,
                            "intent_ids": list(row.intent_ids) if row is not None else []})
            self.last_policy_id = policy_id
        self.pending.append((self.backend.read_kpis(), policy_id))
        write_ok = self.flush()
        return TickResult(self.backend.now(), policy_id, tuple(float(x) for x in shares), read_ok and write_ok)

    def flush(self) -> bool:
        """Write every pending KPI tick in one transaction; keep them if the database is busy."""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            for records, policy_id in self.pending:
                kpi.write_records(self.conn, records, policy_id)
            self.conn.execute("COMMIT")
        except sqlite3.Error:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            del self.pending[:-MAX_PENDING_TICKS]  # bounded memory: drop the oldest if it lasts
            return False
        self.pending.clear()
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
    for table in ("kpi", "a1_policy", "a1_schedule", "intents", "backend_info"):
        conn.execute(f"DELETE FROM {table}")
    conn.execute("DELETE FROM sqlite_sequence")  # policy ids restart at 1
    conn.execute("COMMIT")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL xApp: the PRB-split control loop")
    parser.add_argument("--backend", choices=["sim", "flexric"], default="sim")
    parser.add_argument("--scenario", default="demo", help="sim: steady | urllc_spike | demo")
    parser.add_argument("--speed", type=float, default=1.0, help="sim: 1 = real time, 0 = as fast as possible")
    parser.add_argument("--model-label", default=None, help="version label of --policy-file (the RL xApp image's)")
    parser.add_argument("--smoothing-alpha", type=float, default=None,
                        help=f"filter on the split; default {SMOOTHING_ALPHA} (sim), {TESTBED_SMOOTHING_ALPHA} (flexric)")
    parser.add_argument("--ticks", type=int, default=None, help="stop after N ticks (default: run forever)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-file", type=Path, default=None, help="serve this SB3 file instead of the champion")
    parser.add_argument("--keep-db", action="store_true", help="do not empty var/ric.db at startup")
    args = parser.parse_args()

    alpha = args.smoothing_alpha
    if alpha is None:
        alpha = TESTBED_SMOOTHING_ALPHA if args.backend == "flexric" else SMOOTHING_ALPHA
    policy, model_version = load_policy(args.policy_file, alpha)
    if args.model_label is not None:
        model_version = args.model_label
    cfg = CellConfig()
    if args.backend == "flexric":
        if "RIC_DB" not in os.environ:  # never the simulator's var/ric.db: the clocks differ
            os.environ["RIC_DB"] = str(DEFAULT_TESTBED_DB)
        backend = FlexRICBackend(db.connect(timeout_s=0.0), cell_info())
        args.scenario, args.speed = "testbed", 1.0
    else:
        backend = SimBackend(cfg, scenario_traffic(args.scenario, cfg), seed=args.seed, speed=args.speed)
    setup = db.connect()  # startup may wait for other processes; the loop below never does
    if not args.keep_db:
        clear_run_state(setup)
    kpi.write_backend_info(setup, backend.info(), model_version)
    setup.close()
    conn = db.connect(timeout_s=0.0)
    log_event("runner", "run_start", ts_backend=backend.now(),
              data={"model_version": model_version, "scenario": args.scenario, "speed": args.speed, "alpha": alpha})
    print(f"RL runner: {model_version}, scenario {args.scenario}, speed {args.speed}, db {db.db_path()}")
    try:
        Runner(backend, policy, conn, cfg.dt_s).run(args.ticks)
    except KeyboardInterrupt:
        pass
    if isinstance(backend, FlexRICBackend):
        backend.close()
    log_event("runner", "run_stop", ts_backend=backend.now())


if __name__ == "__main__":
    main()
