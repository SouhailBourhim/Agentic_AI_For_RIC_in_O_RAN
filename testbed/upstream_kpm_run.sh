#!/usr/bin/env bash
# One rfsim run for the upstream KPM PR: start the testbed with the given gNB image, record the
# KPM xApp's JSON lines while iperf3 sends 30 / 10 / 5 Mb/s to the three slices, then stop.
# Usage: testbed/upstream_kpm_run.sh <gnb image> <label>      -> testbed/results/upstream_kpm/<label>/
set -uo pipefail
IMAGE=$1; LABEL=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=testbed/results/upstream_kpm/$LABEL
cd "$REPO"; mkdir -p "$OUT"
GNB_IMAGE=$IMAGE testbed/up.sh; up=$?
echo "up.sh exit=$up"
if [ $up -eq 0 ]; then
  sleep 8
  docker logs -f --since 0s xapp-kpm-slices 2>/dev/null | grep --line-buffered '^{"slice"' > "$OUT/kpm_raw.jsonl" &
  uv run python testbed/traffic.py together 30M 10M 5M --seconds 30 --results "$OUT"
  sleep 3
  pkill -f "docker logs -f --since 0s xapp-kpm-slices" || true
fi
docker image inspect "$IMAGE" --format '{{.Id}} created {{.Created}}' > "$OUT/gnb_image.txt"
docker logs rfsim5g-oai-gnb > "$OUT/gnb.log" 2>&1 || true
docker logs xapp-kpm-slices 2>&1 | tail -40 > "$OUT/xapp_tail.log" || true
testbed/down.sh > /dev/null 2>&1
echo "lines: $(wc -l < "$OUT/kpm_raw.jsonl" 2>/dev/null)"
