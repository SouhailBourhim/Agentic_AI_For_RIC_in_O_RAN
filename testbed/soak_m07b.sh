#!/usr/bin/env bash
# M07b soak: the RL xApp controlling the real gNB for 30 minutes under steady load, then one
# restart of the RL xApp, to look for E2 errors, crashes and memory growth. The testbed must be up
# on the slice gNB (testbed/up.sh). Takes ~33 minutes.
#
#   testbed/soak_m07b.sh
#
# Writes testbed/results/m07b/soak_summary.txt and soak/..._docker_stats.csv (memory of gNB, RIC,
# xApps every ~2 s). The raw iperf3 and ping files, kpi rows and events go to var/build/m07b-raw/
# (git-ignored: ~6 MB per run).
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE="docker compose -f testbed/rfsim-flexric/docker-compose.yml --profile control"
export RIC_DB=var/ric-testbed.db
SOAK_S=${SOAK_S:-1800}
START=$(date +%s.%N)
RAW=var/build/m07b-raw
mkdir -p "$RAW/soak" testbed/results/m07b/soak
GNB_RESTARTS_BEFORE=$(docker inspect -f '{{.RestartCount}}' rfsim5g-oai-gnb)

uv run python - <<'PY'
from rl.runner import clear_run_state
from store import db
conn = db.connect()
clear_run_state(conn)
conn.execute("DELETE FROM kpm_latest")
conn.execute("DELETE FROM traffic_target")
PY
uv run python -m xapps.kpm_monitor --for-runner > var/build/kpm_monitor_soak.out 2>&1 &
KPM_PID=$!
trap 'kill $KPM_PID 2>/dev/null || true' EXIT
uv run python - <<'PY'   # steady offered rates, known to the policy as in the loop runs
from store import db, testbed
testbed.write_traffic_target(db.connect(), {"embb": 30.0, "urllc": 15.0, "be": 20.0})
PY
$COMPOSE up -d rl-xapp
sleep 5

echo "soak: ${SOAK_S} s under 30 / 15 / 20 Mb/s (eMBB / URLLC / BE)"
uv run python testbed/traffic.py together 30M 15M 20M --seconds "$SOAK_S" --results "$RAW/soak"
cp "$RAW/soak/together_30M_15M_20M_docker_stats.csv" testbed/results/m07b/soak/

echo "restart of the RL xApp, then 60 s more"
$COMPOSE restart rl-xapp
sleep 60
$COMPOSE stop rl-xapp
docker logs rl-xapp > var/build/rl_xapp_soak.out 2>&1
$COMPOSE rm -f rl-xapp > /dev/null

uv run python - "$START" "$GNB_RESTARTS_BEFORE" <<'PY' | tee testbed/results/m07b/soak_summary.txt
import csv, json, subprocess, sys
from store import db
from store.events import read_events
start, restarts_before = float(sys.argv[1]), int(sys.argv[2])
events = [e for e in read_events() if e["ts_wall"] >= start]
with open("var/build/m07b-raw/soak_events.jsonl", "w") as f:   # raw, git-ignored
    f.writelines(json.dumps(e) + "\n" for e in events)
rows = db.connect().execute("SELECT * FROM kpi ORDER BY ts").fetchall()
with open("var/build/m07b-raw/soak_kpi.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(rows[0].keys()); w.writerows(tuple(r) for r in rows)
count = lambda name: sum(e["event"] == name for e in events)
rtts = sorted(e["data"]["rtt_ms"] for e in events if e["event"] == "xapp_ack")
gnb = subprocess.run(["docker", "logs", "--since", str(int(start)), "rfsim5g-oai-gnb"], capture_output=True, text=True)
log = gnb.stdout + gnb.stderr
restarts = int(subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}}", "rfsim5g-oai-gnb"],
                              capture_output=True, text=True).stdout)
print(f"controls acknowledged: {len(rtts)}; ack RTT median {rtts[len(rtts)//2]:.2f} ms, max {rtts[-1]:.2f} ms" if rtts else "no acks")
print(f"runner starts: {count('run_start')}; xApp exits: {count('xapp_exited')}; kpm stale: {count('kpm_stale')}")
print(f"gNB 'SLICE CONTROL OK': {log.count('SLICE CONTROL OK')}, 'REFUSED': {log.count('SLICE CONTROL REFUSED')}, "
      f"'Assertion': {log.count('Assertion')}, restarts during the soak: {restarts - restarts_before}")
retx = [int(x) for x in __import__('re').findall(r"retx waits since last control: (\d+)", log)]
print(f"HARQ retransmissions that waited for room in their slice: {sum(retx)}")
print(f"kpi rows: {len(rows)}")
PY
