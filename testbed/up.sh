#!/usr/bin/env bash
# Start the testbed in the order that works (M06-spike lesson: core -> RIC -> gNB -> UEs; E2
# state is fragile if this order is broken), wait for each step, then print one line per slice.
#
# Usage: testbed/up.sh          (from anywhere; never runs sudo)
# Stop:  testbed/down.sh
# The KPM monitor xApp is not started here (M07a replaces it):
#   docker compose -f testbed/rfsim-flexric/docker-compose.yml up -d xapp-kpm-moni

set -euo pipefail
cd "$(dirname "$0")"

CORE=oai-cn5g/docker-compose.yaml
RAN=rfsim-flexric/docker-compose.yml
EXT_DN_IP=192.168.70.135
# slice  container                  expected UE subnet
SLICES=(
  "embb  rfsim5g-oai-nr-ue-embb  10.0.1."
  "urllc rfsim5g-oai-nr-ue-urllc 10.0.2."
  "be    rfsim5g-oai-nr-ue-be    10.0.3."
)

# N2 (gNB-AMF) and E2 (gNB-RIC) run over SCTP. The module is lost after `wsl --shutdown`.
if ! grep -q '^sctp ' /proc/modules; then
  echo "The SCTP kernel module is not loaded. Load it with:  sudo modprobe sctp   then run this again." >&2
  exit 1
fi
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not reachable from WSL: start Docker Desktop (WSL integration on)." >&2
  exit 1
fi

wait_healthy() {  # compose_file: wait until no service with a healthcheck is anything but healthy
  local file=$1
  for _ in $(seq 1 60); do
    local waiting
    waiting=$(docker compose -f "$file" ps --format '{{.Name}} {{.Health}}' | awk '$2 != "" && $2 != "healthy"')
    if [ -z "$waiting" ]; then
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for: $waiting" >&2
  return 1
}

ue_ip() {  # container -> IPv4 of its tunnel interface, empty if not up yet
  docker exec "$1" ip -4 -o addr show oaitun_ue1 2> /dev/null | awk '{print $4}' | cut -d/ -f1
}

echo "1/4 core (mysql, NRF, UDR, UDM, AUSF, AMF, SMF, UPF, ext-dn)"
docker compose -f "$CORE" up -d
wait_healthy "$CORE"

echo "2/4 near-RT RIC"
docker compose -f "$RAN" up -d nearRT-RIC
wait_healthy "$RAN"

echo "3/4 gNB (rfsim, E2 agent)"
docker compose -f "$RAN" up -d oai-gnb
wait_healthy "$RAN"

echo "4/4 UEs, one per slice"
docker compose -f "$RAN" up -d oai-nr-ue-embb oai-nr-ue-urllc oai-nr-ue-be
for entry in "${SLICES[@]}"; do
  read -r _ container _ <<< "$entry"
  for _ in $(seq 1 60); do
    [ -n "$(ue_ip "$container")" ] && break
    sleep 2
  done
done

echo
printf '%-6s %-25s %-12s %s\n' slice container "UE IP" "ping ext-dn"
status=0
for entry in "${SLICES[@]}"; do
  read -r name container subnet <<< "$entry"
  ip=$(ue_ip "$container")
  if [ -z "$ip" ]; then
    result="NO IP (docker logs $container)"; status=1
  elif [[ "$ip" != "$subnet"* ]]; then
    result="WRONG SUBNET (expected ${subnet}x)"; status=1
  elif docker exec "$container" ping -c 2 -W 2 -I oaitun_ue1 "$EXT_DN_IP" > /dev/null 2>&1; then
    result="ok"
  else
    result="FAILED"; status=1
  fi
  printf '%-6s %-25s %-12s %s\n' "$name" "$container" "${ip:--}" "$result"
done
exit $status
