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
| `rfsim-flexric/` | nearRT-RIC, our KPM xApp `xapp-kpm-slices` (+ the stock monitor xApps, not started), gNB (E2 agent), one nrUE per slice | joins `oai-cn5g-public-net` |
| `gnb/` | `kpm-per-ue.patch` (OAI v3.0.0 KPM fix, M07a) and `build.sh` (our gNB image with OAI's own Dockerfiles) | |

## Slices

| Slice | S-NSSAI (SST, SD) | DNN | UE subnet | 5QI | UE container (IMSI) |
|---|---|---|---|---|---|
| eMBB | 1, 000001 | `embb` | 10.0.1.0/24 | 9 | `rfsim5g-oai-nr-ue-embb` (001010000000002) |
| URLLC | 2, 000001 | `urllc` | 10.0.2.0/24 | 7 | `rfsim5g-oai-nr-ue-urllc` (001010000000003) |
| Best effort | 1, 000002 | `be` | 10.0.3.0/24 | 9 | `rfsim5g-oai-nr-ue-be` (001010000000004) |

One AMF, SMF and UPF serve all three. The slices are configured in `oai-cn5g/conf/config.yaml`,
`oai-cn5g/database/oai_db.sql`, `rfsim-flexric/conf/gnb.conf` and
`rfsim-flexric/conf/nrue-<slice>.conf`; `tests/test_testbed_config.py` checks they agree.
**The stock OAI v3.0.0 scheduler ignores slices and 5QI** (equal shares in overload). Our gNB
image adds per-slice PRB ranges set over E2 (M07b, `gnb/slice-control.patch`); until the first
control message it schedules exactly like the stock one.

Addresses: AMF .132, SMF .133, UPF .134, ext-dn .135, nearRT-RIC .150, gNB .151,
UEs .152 (eMBB) / .153 (URLLC) / .158 (BE), KPM xApp .156, RL xApp .159 (M07b, started on demand).
UE IPs come from the SMF at attach time.

## Prerequisites

- Three images we build: `testbed/gnb/build.sh` (gNB with the KPM and slice-control patches,
  `agentic-ric/oai-gnb:v3.0.0-slice`; long the first time), `docker build -t
  agentic-ric/xapp-kpm-slices:v3.0.0 xapps/kpm`, and `xapps/rl/build.sh` (the RL xApp with the
  registry's champion, `agentic-ric/rl-xapp:v<version>` and `:latest`).
  `GNB_IMAGE=oaisoftwarealliance/oai-gnb:v3.0.0 testbed/up.sh` runs the stock gNB instead (its KPM
  values are wrong with one subscription per slice: `docs/learn/M07a-kpm.md`).

- Docker with Compose ≥ 2.36 (Docker Desktop with WSL integration works).
- The SCTP kernel module, needed by N2 (gNB ↔ AMF) and E2 (gNB ↔ RIC). It is lost after a WSL
  restart: `sudo modprobe sctp`. `up.sh` checks it and stops if it is missing.

## Start, use, stop

```bash
testbed/up.sh            # core -> RIC -> gNB -> UEs -> KPM xApp, waits for each, prints slice / UE IP / ping
uv run python -m xapps.kpm_monitor                         # KPM reports -> var/ric-testbed.db (M07a)
uv run python testbed/traffic.py idle                     # RTT per slice
uv run python testbed/traffic.py alone --rate 200M        # capacity, one slice at a time
uv run python testbed/traffic.py together 40M 40M 40M     # eMBB URLLC BE at once (+ RTT, docker stats)
uv run python testbed/plot_m06.py                         # figures in docs/learn/img/
testbed/down.sh          # reverse order
```

PRB-split control (M07b, `docs/learn/M07b-slice-control.md`):

```bash
testbed/run_loop_m07b.sh loop_rl rl                       # KPM writer + RL xApp + the URLLC surge (~8 min)
uv run python testbed/measure_m07b.py cap                 # bench: splits sent straight to the xApp
uv run python testbed/check_gateway_route.py              # invariant 2: no route from the gateway's side
testbed/soak_m07b.sh                                      # 30 min under the RL xApp, then a restart
docker logs rfsim5g-oai-gnb 2>&1 | grep "SLICE CONTROL"   # every command, OK or REFUSED with the reason
```

The stock `xapp-kpm-moni` gets only cell-level reports with these slices (its per-UE
subscription filters on SST 1 without SD, which OAI matches only against UEs with no SD).
Our `xapp-kpm-slices` subscribes once per slice with the full S-NSSAI (M07a).
`traffic.py --results testbed/results/m07a` writes elsewhere than the M06 folder.

Packet captures go to `testbed/captures/` (not in git) and measurement outputs to `testbed/results/`.
