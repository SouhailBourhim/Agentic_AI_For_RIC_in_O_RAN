"""Per-slice downlink traffic on the testbed, and the M06 measurements (capacity, CPU/RAM, RTT).

Run from the repo root, with the testbed up (testbed/up.sh):

  uv run python testbed/traffic.py idle                          ping only: baseline RTT per slice
  uv run python testbed/traffic.py alone [--rate 200M]           each slice alone: the cell capacity
  uv run python testbed/traffic.py together 40M 40M 40M          the three slices at once (eMBB URLLC BE)

Method (same as the spike's measure_throughput.sh): ext-dn, behind the UPF, sends UDP at a fixed
rate to each UE; the iperf3 client runs in a throwaway netshoot container sharing the UE's network
namespace (so it sees oaitun_ue1) and records what arrives (-R: the server sends). One iperf3
server port per slice on ext-dn. During `idle` and `together`, each UE also pings ext-dn every
0.2 s: the echo reply queues in the gNB behind that UE's downlink data, so the RTT shows the
slice's downlink queueing. `docker stats` is sampled during every run.

Writes testbed/results/m06/<run>_<slice>.json (raw iperf3), <run>_ping_<slice>.txt,
<run>_docker_stats.csv, and prints a summary computed from those files.
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # testbed/, for plot_spike
from plot_spike import load_run  # noqa: E402  (receiver-side throughput from iperf3 JSON)

RESULTS = Path("testbed/results/m06")  # --results changes it (M07a writes to testbed/results/m07a)
EXT_DN = "oai-ext-dn"
EXT_DN_IP = "192.168.70.135"
# The UE <-> slice mapping of the testbed (configs: tests/test_testbed_config.py). M07a reuses it.
SLICES = {
    "embb": {"container": "rfsim5g-oai-nr-ue-embb", "port": 5201},
    "urllc": {"container": "rfsim5g-oai-nr-ue-urllc", "port": 5202},
    "be": {"container": "rfsim5g-oai-nr-ue-be", "port": 5203},
}
PING_INTERVAL_S = 0.2
SEGMENT_PORT_STEP = 10  # M08d: segment i of a run uses port + 10 i, as the previous iperf3 may still be draining
# A run starts with a transient: after idle, link adaptation restarts from a low MCS (6 -> 10 -> 19
# Mb/s in the first seconds), a queue builds, then drains. Measured in M06: gone after ~15 s.
STEADY_AFTER_S = 15.0


def docker(*args: str, check: bool = True) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout


def ue_ip(container: str) -> str:
    """The SMF assigns UE addresses at attach time, so read it from the tunnel interface."""
    out = docker("exec", container, "ip", "-4", "-o", "addr", "show", "oaitun_ue1")
    match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/", out)
    if match is None:
        raise SystemExit(f"{container} has no oaitun_ue1 address: is the testbed up (testbed/up.sh)?")
    return match.group(1)


def start_servers(segments: int = 1) -> None:
    """One iperf3 server per slice port and segment on ext-dn. ext-dn already runs one on 5201; a second
    daemon on a port in use just fails to bind and exits, which is fine."""
    for spec in SLICES.values():
        for segment in range(segments):
            docker("exec", EXT_DN, "iperf3", "-s", "-D", "-p", str(spec["port"] + SEGMENT_PORT_STEP * segment), check=False)
    time.sleep(1)


def iperf(name: str, rate: str, seconds: int, out: Path, segment: int = 0) -> subprocess.Popen:
    spec = SLICES[name]
    cmd = ["docker", "run", "--rm", "--net", f"container:{spec['container']}", "nicolaka/netshoot",
           "iperf3", "-c", EXT_DN_IP, "-p", str(spec["port"] + SEGMENT_PORT_STEP * segment), "-B", ue_ip(spec["container"]),
           "-u", "-b", rate, "-R", "-t", str(seconds), "-J"]
    return subprocess.Popen(cmd, stdout=out.open("w"), stderr=subprocess.PIPE, text=True)


def ping(name: str, seconds: int, out: Path) -> subprocess.Popen:
    count = int(seconds / PING_INTERVAL_S)
    cmd = ["docker", "exec", SLICES[name]["container"], "ping", "-i", str(PING_INTERVAL_S), "-c", str(count),
           "-I", "oaitun_ue1", EXT_DN_IP]
    return subprocess.Popen(cmd, stdout=out.open("w"), stderr=subprocess.STDOUT, text=True)


class DockerStats(threading.Thread):
    """Samples `docker stats` (CPU %, memory) of every container until stopped. One sample takes
    about 2 s (docker measures CPU over an interval), which sets the sampling period."""

    def __init__(self, out: Path) -> None:
        super().__init__(daemon=True)
        self.out, self.stop = out, threading.Event()

    def run(self) -> None:
        with self.out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_wall", "container", "cpu_percent", "mem_mib"])
            while not self.stop.is_set():
                now = time.time()
                for line in docker("stats", "--no-stream", "--format", "{{.Name}};{{.CPUPerc}};{{.MemUsage}}").splitlines():
                    name, cpu, mem = line.split(";")
                    writer.writerow([f"{now:.1f}", name, cpu.rstrip("%"), f"{mib(mem.split('/')[0]):.1f}"])
                f.flush()


def mib(text: str) -> float:
    """'1.2GiB' / '350.5MiB' / '900KiB' -> MiB."""
    value, unit = re.match(r"([\d.]+)\s*([KMG]i?B|B)", text.strip()).groups()
    return float(value) * {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit]


def rtt_quantiles(rtts: list[float]) -> str:
    rtts = sorted(rtts)
    if not rtts:
        return "no replies"
    q = lambda f: rtts[min(len(rtts) - 1, int(f * len(rtts)))]  # noqa: E731
    return f"median {q(0.5):.0f} ms, p95 {q(0.95):.0f} ms"


def ping_summary(path: Path) -> str:
    """RTT over the whole run and after the start-up transient (ping sent at seq x interval)."""
    replies = [(int(seq), float(rtt)) for seq, rtt in re.findall(r"icmp_seq=(\d+) .*time=([\d.]+) ms", path.read_text())]
    steady = [rtt for seq, rtt in replies if seq * PING_INTERVAL_S >= STEADY_AFTER_S]
    return (f"RTT {rtt_quantiles([rtt for _, rtt in replies])} (all {len(replies)}); "
            f"after {STEADY_AFTER_S:g} s: {rtt_quantiles(steady)}")


def run(label: str, rates: dict[str, str], seconds: int, with_ping: bool, stats_label: str | None = None) -> None:
    """One run: iperf3 at the given rates (all at once) and/or pings, with docker stats."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stats = DockerStats(RESULTS / f"{stats_label or label}_docker_stats.csv")
    stats.start()
    procs = [iperf(name, rate, seconds, RESULTS / f"{label}_{name}.json") for name, rate in rates.items()]
    pings = [ping(name, seconds, RESULTS / f"{label}_ping_{name}.txt") for name in SLICES] if with_ping else []
    for proc in procs + pings:
        proc.wait()
    stats.stop.set()
    stats.join()
    print(f"\n{label}:")
    for name in SLICES:
        if name not in rates and not with_ping:
            continue
        parts = []
        if name in rates:
            r = load_run(RESULTS / f"{label}_{name}.json")
            parts.append(f"offered {r['offered_mbps']:.1f} Mb/s, received {r['steady_mbps']:.1f} Mb/s "
                         f"(steady), delivered {100 * r['delivered']:.0f} %")
        if with_ping:
            parts.append(ping_summary(RESULTS / f"{label}_ping_{name}.txt"))
        print(f"  {name:6s} " + "; ".join(parts))


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser(description="Per-slice downlink traffic on the testbed (M06)")
    parser.add_argument("mode", choices=["idle", "alone", "together"])
    parser.add_argument("rates", nargs="*", help="together: eMBB URLLC BE rates, e.g. 40M 40M 40M")
    parser.add_argument("--rate", default="200M", help="alone: the offered rate (above capacity)")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--results", type=Path, default=RESULTS, help="output folder")
    args = parser.parse_args()
    RESULTS = args.results
    start_servers()
    if args.mode == "idle":
        run("idle", {}, args.seconds, with_ping=True)
    elif args.mode == "alone":
        for name in SLICES:  # one after the other, a pause between runs
            run(f"alone_{args.rate}", {name: args.rate}, args.seconds, with_ping=False,
                stats_label=f"alone_{args.rate}_{name}")
            time.sleep(3)
    else:
        if len(args.rates) != 3:
            parser.error("together needs three rates: eMBB URLLC BE")
        label = "together_" + "_".join(args.rates)
        run(label, dict(zip(SLICES, args.rates)), args.seconds, with_ping=True)


if __name__ == "__main__":
    main()
