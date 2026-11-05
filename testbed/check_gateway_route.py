"""Hard invariant 2 on the testbed (M07b): the MCP gateway has no network route to FlexRIC or OAI.

  uv run python testbed/check_gateway_route.py      # the testbed must be up (testbed/up.sh)

The gateway runs in WSL (`python -m gateway`, and Hermes reaches it through a relay, M05b). This
script tries a TCP connection to each RAN address from WSL, i.e. from the gateway's own network
context, and then the same list from a throwaway container on the testbed network, so that a
failure on the first side means "no route", not "nothing there". (E2 and E42 are SCTP, so a TCP
attempt on them is refused by a live RIC: refused counts as alive.)

Writes testbed/results/m07b/gateway_route.json; exit code 0 if WSL reaches none of them and the
control reaches every one.
"""

import json
import socket
import subprocess
from pathlib import Path

OUT = Path("testbed/results/m07b/gateway_route.json")
TARGETS = [  # (what, address, TCP port)
    ("AMF, SBI", "192.168.70.132", 8080),
    ("SMF, SBI", "192.168.70.133", 8080),
    ("UPF, SBI", "192.168.70.134", 8080),
    ("ext-dn, iperf3", "192.168.70.135", 5201),
    ("near-RT RIC, E2", "192.168.70.150", 36421),
    ("near-RT RIC, E42", "192.168.70.150", 36422),
    ("gNB, rfsim", "192.168.70.151", 4043),
]
PROBE = """
import json, socket, sys
for host, port in json.load(sys.stdin):
    s = socket.socket(); s.settimeout(3)
    try:
        s.connect((host, port)); print("connected")
    except ConnectionRefusedError:
        print("refused")
    except Exception as e:
        print(type(e).__name__)
    finally:
        s.close()
"""


def from_here(host: str, port: int) -> str:
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect((host, port))
        return "connected"
    except ConnectionRefusedError:
        return "refused"
    except Exception as e:  # timeout, no route, unreachable
        return type(e).__name__
    finally:
        s.close()


def from_testbed_network() -> list[str]:
    pairs = json.dumps([[h, p] for _, h, p in TARGETS])
    out = subprocess.run(["docker", "run", "--rm", "-i", "--network", "oai-cn5g-public-net", "--entrypoint", "python3",
                          "oaisoftwarealliance/oai-flexric:v3.0.0", "-c", PROBE],
                         input=pairs, capture_output=True, text=True, check=True).stdout
    return out.split()


def main() -> None:
    control = from_testbed_network()
    rows = []
    for (what, host, port), ctl in zip(TARGETS, control, strict=True):
        wsl = from_here(host, port)
        rows.append({"target": what, "host": host, "port": port, "from_wsl": wsl, "from_testbed_network": ctl})
        print(f"{what:18s} {host}:{port:<6} WSL (gateway): {wsl:16s} testbed network: {ctl}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    alive = {"connected", "refused"}
    ok = all(r["from_wsl"] not in alive and r["from_testbed_network"] in alive for r in rows)
    print("PASS: no route from the gateway's side, every target alive" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
