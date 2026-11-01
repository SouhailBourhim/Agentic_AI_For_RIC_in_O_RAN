#!/usr/bin/env bash
# Measure the rfsim downlink throughput with iperf3 (UDP), for the M06 spike.
#
# The ext-dn container (behind the UPF) sends UDP at a fixed offered rate to each UE; the UE
# side records what actually arrives, once per second. We run the iperf3 client from a
# throwaway netshoot container that shares the UE container's network namespace
# (--net container:...), so it sees the UE's tunnel interface oaitun_ue1 without changing
# the UE image. `-R` reverses the direction: the server (ext-dn) sends, the client receives.
#
# Usage: testbed/measure_throughput.sh [offered_rate] [seconds]   e.g. 200M 20
# Output: testbed/results/spike/<run>.json (raw iperf3 JSON, one file per UE and run)

set -euo pipefail

RATE="${1:-200M}"
DURATION="${2:-20}"
OUT_DIR="testbed/results/spike"
EXT_DN_IP="192.168.70.135"
mkdir -p "$OUT_DIR"

# container  iperf3 port on ext-dn (one server per concurrent client)
UES=(
  "rfsim5g-oai-nr-ue 5201"
  "rfsim5g-oai-nr-ue2 5202"
  "rfsim5g-oai-nr-ue3 5203"
)

ue_ip() {  # the SMF assigns UE addresses at attach time, so read it from the tunnel interface
  docker exec "$1" ip -4 -o addr show oaitun_ue1 | awk '{print $4}' | cut -d/ -f1
}

# ext-dn already runs `iperf3 -s` on 5201; start one more server per extra port. If one is
# already listening on that port, the new daemon just fails to bind and exits, which is fine.
# (Do not guard this with `pgrep -f` inside `sh -c`: pgrep matches the sh command line itself.)
for port in 5202 5203; do
  docker exec oai-ext-dn iperf3 -s -D -p "$port" || true
done
sleep 1

run_client() {  # container ue_ip port output_name
  docker run --rm --net "container:$1" nicolaka/netshoot \
    iperf3 -c "$EXT_DN_IP" -p "$3" -B "$2" -u -b "$RATE" -R -t "$DURATION" -J > "$OUT_DIR/$4.json"
}

echo "1) each UE alone, offered ${RATE} for ${DURATION}s"
for entry in "${UES[@]}"; do
  read -r container port <<< "$entry"
  run_client "$container" "$(ue_ip "$container")" "$port" "alone_${container##*-}"
  sleep 3
done

echo "2) all three UEs together"
pids=()
for entry in "${UES[@]}"; do
  read -r container port <<< "$entry"
  run_client "$container" "$(ue_ip "$container")" "$port" "together_${container##*-}" &
  pids+=($!)
done
wait "${pids[@]}"

echo "done: $(ls "$OUT_DIR")"
