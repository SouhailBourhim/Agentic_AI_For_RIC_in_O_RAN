# testbed/

The real RAN for the project: the "RAN et cœur 5G open source" box of figure 2.
OAI 5G core (CN5G), OAI gNB and 3 OAI nrUE in RF-simulator mode, and the FlexRIC near-RT RIC,
all from the official Docker images. Configs are copied from the OAI repository at tag
**v3.0.0** (`github.com/duranta-project/openairinterface5g`). Each file names its source and
any change we made.

| Folder | What | Network |
|---|---|---|
| `oai-cn5g/` | mysql, NRF, UDR, UDM, AUSF, AMF, SMF, UPF, ext-dn (traffic endpoint with `iperf3 -s`) | creates `oai-cn5g-public-net` 192.168.70.128/26 |
| `rfsim-flexric/` | nearRT-RIC, KPM + MAC/RLC/PDCP/GTP monitor xApps, gNB (E2 agent), 3 nrUE | joins `oai-cn5g-public-net` |

Addresses: AMF .132, UPF .134, ext-dn .135, nearRT-RIC .150, gNB .151, UEs .152/.153/.158.
UE IPs are given by the SMF at attach time, in attach order, from 10.0.0.0/24; the static IPs in the
subscriber database are not used. Read them with `ip -4 addr show oaitun_ue1`. PLMN 001/01, one slice SST=1.

## Prerequisites

- Docker with Compose ≥ 2.36 (Docker Desktop with WSL integration works).
- The SCTP kernel module, needed by N2 (gNB ↔ AMF) and E2 (gNB ↔ RIC):
  `lsmod | grep sctp`. If it's missing after a WSL restart, run `sudo modprobe sctp`.

## Bring up, in this order

```bash
cd testbed
docker compose -f oai-cn5g/docker-compose.yaml up -d          # core; wait until all are healthy
docker compose -f oai-cn5g/docker-compose.yaml ps
docker compose -f rfsim-flexric/docker-compose.yml up -d nearRT-RIC
docker compose -f rfsim-flexric/docker-compose.yml up -d oai-gnb
docker compose -f rfsim-flexric/docker-compose.yml up -d oai-nr-ue oai-nr-ue2 oai-nr-ue3
docker exec rfsim5g-oai-nr-ue ip -4 addr show oaitun_ue1      # a 10.0.0.x address
docker exec rfsim5g-oai-nr-ue ping -c 3 -I oaitun_ue1 192.168.70.135
docker compose -f rfsim-flexric/docker-compose.yml up -d xapp-kpm-moni   # prints KPM reports
docker logs -f xapp-kpm-moni
```

## Tear down

```bash
docker compose -f rfsim-flexric/docker-compose.yml down
docker compose -f oai-cn5g/docker-compose.yaml down
```

Packet captures go to `testbed/captures/` and measurement outputs to `testbed/results/`.
How they were made is in `docs/learn/M06-spike.md`.
