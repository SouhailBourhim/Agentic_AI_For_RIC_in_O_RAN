"""What can the Hermes container reach? Tries each target from inside the container (M05b, H3).

  uv run python gateway/hermes/probe.py --out testbed/results/m05b/network_probe.json

Start the testbed first (testbed/up.sh), so that its addresses are alive and "unreachable" means
the network refuses, not that nothing is there. Writes one row per target: expected, result, detail.

  uv run python gateway/hermes/probe.py --control --out testbed/results/m05b/network_probe_control.json

--control runs the same code in the same image, attached to the testbed network instead: a refused
connection there shows the address is alive, where the Hermes container gets "Network is unreachable".
"""

import argparse
import json
import subprocess
from pathlib import Path

COMPOSE = ["docker", "compose", "-f", "gateway/hermes/docker-compose.yml"]
INSIDE_HERMES = [*COMPOSE, "exec", "-T", "hermes", "/opt/hermes/.venv/bin/python"]
IMAGE = "nousresearch/hermes-agent:v0.22.7@sha256:e6d81ae52ebf6a43efa529701405509f20054ba818f179ddd748bed4f8af584c"
ON_TESTBED_NET = ["docker", "run", "--rm", "-i", "--network", "oai-cn5g-public-net",
                  "--entrypoint", "/opt/hermes/.venv/bin/python", IMAGE]

# (label, host, port, should it connect?)
TARGETS = [
    ("MCP gateway, through its relay", "gateway-relay", 8100, True),
    ("Ollama, through its relay", "ollama.internal", 11434, True),
    ("gateway relay, agent API port", "gateway-relay", 8000, False),
    ("gateway relay, Ollama port", "gateway-relay", 11434, False),
    ("Ollama relay, gateway port", "ollama-relay", 8100, False),
    ("Ollama relay, agent API port", "ollama-relay", 8000, False),
    ("AMF (N2 is SCTP; SBI port)", "192.168.70.132", 8080, False),
    ("SMF (SBI port)", "192.168.70.133", 8080, False),
    ("ext-dn (iperf3)", "192.168.70.135", 5201, False),
    ("near-RT RIC (E2 port)", "192.168.70.150", 36421, False),
    ("near-RT RIC (E42 port)", "192.168.70.150", 36422, False),
    ("gNB (rfsim port)", "192.168.70.151", 4043, False),
    ("host.docker.internal, agent API", "host.docker.internal", 8000, False),
    ("host.docker.internal, Ollama", "host.docker.internal", 11434, False),
    ("Docker Desktop host address, agent API", "192.168.65.254", 8000, False),
    ("internet by name", "github.com", 443, False),
    ("internet by address", "1.1.1.1", 443, False),
]

# Run inside the container. TCP for all; a refused connection and a timeout are both "no".
INSIDE = """
import json, socket, sys
for host, port in json.load(sys.stdin):
    s = socket.socket()
    s.settimeout(4)
    try:
        s.connect((host, port))
        print(json.dumps([True, "connected"]))
    except Exception as e:
        print(json.dumps([False, type(e).__name__ + ": " + str(e)]))
    finally:
        s.close()
"""


def probe(python: list[str]) -> list[dict]:
    pairs = json.dumps([[host, port] for _, host, port, _ in TARGETS])
    out = subprocess.run([*python, "-c", INSIDE], input=pairs, capture_output=True, text=True, check=True).stdout
    rows = []
    for (label, host, port, expected), line in zip(TARGETS, out.splitlines(), strict=True):
        reached, detail = json.loads(line)
        rows.append({"label": label, "host": host, "port": port, "expected": expected, "reached": reached,
                     "detail": detail})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--control", action="store_true", help="probe from the testbed network instead")
    args = parser.parse_args()
    rows = probe(ON_TESTBED_NET if args.control else INSIDE_HERMES)
    for r in rows:
        verdict = "" if args.control else "as expected" if r["reached"] == r["expected"] else "UNEXPECTED"
        print(f"{'reached' if r['reached'] else 'no':8s} {verdict:12s} {r['label']:42s} {r['host']}:{r['port']}  {r['detail'][:60]}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    if args.control:  # no expectation to meet: the rows are compared with the Hermes run by eye and in the note
        return
    raise SystemExit(0 if all(r["reached"] == r["expected"] for r in rows) else 1)


if __name__ == "__main__":
    main()
