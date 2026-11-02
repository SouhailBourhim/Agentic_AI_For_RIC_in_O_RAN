# testbed/

The real RAN for the project: the "RAN et cœur 5G open source" box of figure 2.
OAI 5G core (CN5G), OAI gNB and 3 OAI nrUE in RF-simulator mode, and the FlexRIC near-RT RIC,
all from the official Docker images. RAN configs come from the OAI repository at tag **v3.0.0**
(`github.com/duranta-project/openairinterface5g`); the multi-slice core config follows
oai-cn5g-fed **v2.2.1** (the version of the core images). Each file names its source and any
change we made. How it works and what was measured: `docs/learn/M06-testbed.md` (M06) and
`docs/learn/M06-spike.md` (the spike).

| Folder | What | Network |
|---|---|---|
| `oai-cn5g/` | mysql, NRF, UDR, UDM, AUSF, AMF, SMF, UPF, ext-dn (traffic endpoint with `iperf3 -s`) | creates `oai-cn5g-public-net` 192.168.70.128/26 |
| `rfsim-flexric/` | nearRT-RIC, KPM + MAC/RLC/PDCP/GTP monitor xApps, gNB (E2 agent), one nrUE per slice | joins `oai-cn5g-public-net` |

## Slices

| Slice | S-NSSAI (SST, SD) | DNN | UE subnet | 5QI | UE container (IMSI) |
|---|---|---|---|---|---|
| eMBB | 1, 000001 | `embb` | 10.0.1.0/24 | 9 | `rfsim5g-oai-nr-ue-embb` (001010000000002) |
| URLLC | 2, 000001 | `urllc` | 10.0.2.0/24 | 7 | `rfsim5g-oai-nr-ue-urllc` (001010000000003) |
| Best effort | 1, 000002 | `be` | 10.0.3.0/24 | 9 | `rfsim5g-oai-nr-ue-be` (001010000000004) |

One AMF, SMF and UPF serve all three. The slices are configured in `oai-cn5g/conf/config.yaml`,
`oai-cn5g/database/oai_db.sql`, `rfsim-flexric/conf/gnb.conf` and
`rfsim-flexric/conf/nrue-<slice>.conf`; `tests/test_testbed_config.py` checks they agree.
**The stock OAI v3.0.0 scheduler ignores slices and 5QI** (equal shares in overload): per-slice
PRB control is M07b.

Addresses: AMF .132, SMF .133, UPF .134, ext-dn .135, nearRT-RIC .150, gNB .151,
UEs .152 (eMBB) / .153 (URLLC) / .158 (BE). UE IPs come from the SMF at attach time.

## Prerequisites

- Docker with Compose ≥ 2.36 (Docker Desktop with WSL integration works).
- The SCTP kernel module, needed by N2 (gNB ↔ AMF) and E2 (gNB ↔ RIC). It is lost after a WSL
  restart: `sudo modprobe sctp`. `up.sh` checks it and stops if it is missing.

## Start, use, stop

```bash
testbed/up.sh            # core -> RIC -> gNB -> UEs, waits for each, prints slice / UE IP / ping
uv run python testbed/traffic.py idle                     # RTT per slice
uv run python testbed/traffic.py alone --rate 200M        # capacity, one slice at a time
uv run python testbed/traffic.py together 40M 40M 40M     # eMBB URLLC BE at once (+ RTT, docker stats)
uv run python testbed/plot_m06.py                         # figures in docs/learn/img/
docker compose -f testbed/rfsim-flexric/docker-compose.yml up -d xapp-kpm-moni   # stock KPM xApp
testbed/down.sh          # reverse order
```

The stock `xapp-kpm-moni` gets only cell-level reports with these slices (its per-UE
subscription filters on SST 1 without SD); M07a's own xApp subscribes per slice.

Packet captures go to `testbed/captures/` (not in git) and measurement outputs to `testbed/results/`.
