#!/usr/bin/env bash
# One closed-loop run on the testbed (M07b): store cleared, KPM writer + RL xApp started, the
# surge scenario played, everything stopped. The testbed must be up on the slice gNB.
#
#   testbed/run_loop_m07b.sh loop1 rl          RL xApp, offered rates known (traffic_target)
#   testbed/run_loop_m07b.sh loop_blind blind  RL xApp, offered rates NOT given to the policy
#   testbed/run_loop_m07b.sh loop_stock stock  no RL xApp: the gNB's stock scheduler
#
# Leaves var/ric-testbed.db (kpi rows of the run) and appends to var/events.jsonl; copies both
# for the run into testbed/results/m07b/<run>_kpi.csv and <run>_events.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1 MODE=$2
COMPOSE="docker compose -f testbed/rfsim-flexric/docker-compose.yml --profile control"
export RIC_DB=var/ric-testbed.db
START=$(date +%s.%N)

uv run python - <<'PY'   # a fresh run: the same tables the runner clears on the simulator
from rl.runner import clear_run_state
from store import db
conn = db.connect()
clear_run_state(conn)
conn.execute("DELETE FROM kpm_latest")
conn.execute("DELETE FROM traffic_target")
PY

if [ "$MODE" = stock ]; then   # nothing controls the split: the M07a writer, straight into kpi
    uv run python -m xapps.kpm_monitor > "var/build/kpm_monitor_$RUN.out" 2>&1 &
else
    uv run python -m xapps.kpm_monitor --for-runner > "var/build/kpm_monitor_$RUN.out" 2>&1 &
fi
KPM_PID=$!
trap 'kill $KPM_PID 2>/dev/null || true; $COMPOSE stop rl-xapp >/dev/null 2>&1 || true' EXIT
sleep 3

if [ "$MODE" != stock ]; then
    $COMPOSE up -d rl-xapp
    sleep 5   # xApp connects to the RIC, runner starts ticking
    docker logs rl-xapp 2>&1 | tail -3
fi
SCALE=${SCALE:-1}   # SCALE=0.1: a ~40 s smoke run
if [ "$MODE" = blind ]; then
    uv run python testbed/scenario_m07b.py --run "$RUN" --no-target --scale "$SCALE"
else
    uv run python testbed/scenario_m07b.py --run "$RUN" --scale "$SCALE"
fi
if [ "$MODE" != stock ]; then
    $COMPOSE stop rl-xapp
    docker logs rl-xapp > "var/build/rl_xapp_$RUN.out" 2>&1
    $COMPOSE rm -f rl-xapp > /dev/null
fi

uv run python - "$RUN" "$START" <<'PY'
import csv, json, sys
from store import db
from store.events import read_events
run, start = sys.argv[1], float(sys.argv[2])
conn = db.connect()
rows = conn.execute("SELECT * FROM kpi ORDER BY ts, slice").fetchall()
with open(f"testbed/results/m07b/{run}_kpi.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(rows[0].keys() if rows else [])
    w.writerows(tuple(r) for r in rows)
events = [e for e in read_events() if (e.get("wall") or e.get("ts_wall") or 0) >= start]
with open(f"testbed/results/m07b/{run}_events.jsonl", "w") as f:
    f.writelines(json.dumps(e) + "\n" for e in events)
print(f"{run}: {len(rows)} kpi rows, {len(events)} events")
PY
