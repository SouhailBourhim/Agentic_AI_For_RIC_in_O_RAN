"""M07b measurements on the testbed: does the per-slice PRB cap set over E2 work, and what does it do?

Run from the repo root with the testbed up on the slice gNB (testbed/up.sh):

  uv run python testbed/measure_m07b.py cap         all three slices overloaded at several splits
  uv run python testbed/measure_m07b.py isolation   BE overloaded next to a light URLLC, stock vs capped

This is a bench test: the splits go straight to the slice-control xApp's stdin, in a throwaway
container of the RL xApp image, not through the RL runner. It measures the radio, not the policy.
Traffic and the throughput read-out are those of testbed/traffic.py (iperf3 UDP from ext-dn, the
receiver's steady-state throughput). The KPM xApp's lines for the whole run are saved too, so the
figure can compare iperf3 with what KPM reports.

Writes testbed/results/m07b/<run>_<slice>.json (raw iperf3), <run>_ping_<slice>.txt,
<mode>_controls.jsonl (each command with its wall time and the xApp's ack), <mode>_kpm.jsonl,
<mode>_gnb.txt (the gNB's SLICE CONTROL lines).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for traffic.py
import traffic  # noqa: E402

RESULTS = Path("testbed/results/m07b")
FLEXRIC_CONF = Path("testbed/rfsim-flexric/conf/flexric.conf").resolve()
XAPP = "/usr/local/flexric/xApp/c/slice/xapp_slice_ctrl"
OVERLOAD = "80M"  # each slice alone can take ~71 Mb/s, so 80 Mb/s overloads every split
SECONDS = 25
SETTLE_S = 2  # after a command, before traffic starts

# (run label, xApp line). "del" = stock scheduler. PRB counts sum to 106 (shares_to_prbs output).
CAP_SPLITS = [
    ("stock", "del"),
    ("equal", "36 35 35"),
    ("s60_30_10", "64 32 10"),
    ("s10_80_10", "11 84 11"),
    ("s05_05_90", "5 5 96"),
    ("s20_20_60", "21 21 64"),
    ("s80_15_05", "85 16 5"),
]
ISOLATION_RATES = {"urllc": "10M", "be": OVERLOAD}
ISOLATION_SPLITS = [("iso_stock", "del"), ("iso_capped", "53 27 26")]  # URLLC 25 % ~ 17 Mb/s > 10


class SliceXapp:
    """The slice-control xApp in a throwaway container on the testbed network, fed line by line."""

    def __init__(self, image: str, log: Path) -> None:
        self.log = log.open("w")
        self.proc = subprocess.Popen(
            ["docker", "run", "-i", "--rm", "--name", "slice-bench", "--network", "oai-cn5g-public-net",
             "-v", f"{FLEXRIC_CONF}:/usr/local/etc/flexric/flexric.conf", "--entrypoint", "stdbuf", image,
             "-o0", XAPP],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.read_until("ready")

    def read_until(self, event: str) -> dict:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.startswith("{"):
                reply = json.loads(line)
                if reply.get("event") == event:
                    return reply
        raise SystemExit(f"slice-control xApp exited before '{event}'")

    def send(self, label: str, line: str) -> None:
        assert self.proc.stdin is not None
        sent = time.time()
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        ack = self.read_until("ack")
        self.log.write(json.dumps({"run": label, "line": line, "sent_wall": sent, "ack": ack}) + "\n")
        self.log.flush()

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait(timeout=20)


def run_series(mode: str, splits: list[tuple[str, str]], rates: dict[str, str], image: str) -> None:
    traffic.RESULTS = RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    traffic.start_servers()
    start = time.time()
    xapp = SliceXapp(image, RESULTS / f"{mode}_controls.jsonl")
    try:
        for label, line in splits:
            xapp.send(label, line)
            time.sleep(SETTLE_S)
            traffic.run(label, rates, SECONDS, with_ping=True, stats_label=f"{mode}_{label}")
        xapp.send("end", "del")  # leave the gNB on its stock scheduler
    finally:
        xapp.close()
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start - 1))
    kpm = subprocess.run(["docker", "logs", "--since", since, "xapp-kpm-slices"], capture_output=True, text=True)
    (RESULTS / f"{mode}_kpm.jsonl").write_text(
        "".join(l + "\n" for l in kpm.stdout.splitlines() if l.startswith("{")))
    gnb = subprocess.run(["docker", "logs", "--since", since, "rfsim5g-oai-gnb"], capture_output=True, text=True)
    (RESULTS / f"{mode}_gnb.txt").write_text(
        "".join(l + "\n" for l in (gnb.stdout + gnb.stderr).splitlines() if "SLICE CONTROL" in l))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["cap", "isolation"])
    parser.add_argument("--image", default="agentic-ric/rl-xapp:latest")
    args = parser.parse_args()
    if args.mode == "cap":
        run_series("cap", CAP_SPLITS, {s: OVERLOAD for s in traffic.SLICES}, args.image)
    else:
        run_series("isolation", ISOLATION_SPLITS, ISOLATION_RATES, args.image)


if __name__ == "__main__":
    main()
