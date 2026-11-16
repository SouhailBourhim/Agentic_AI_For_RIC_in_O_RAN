"""M12: the one-command demo keeps working. It runs as a user would run it (separate processes, the MCP gateway over
stdio, no LLM), with the formula executor so it needs no trained policy, at 30 cell seconds per wall second."""

import json
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_demo_applies_the_intent_and_urllc_meets_its_bound(tmp_path: Path) -> None:
    out = tmp_path / "shown.json"
    run = subprocess.run([sys.executable, "-m", "demo", "--executor", "formula", "--speed", "30",
                          "--port", str(free_port()), "--json", str(out)],
                         cwd=REPO, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "formula executor (M09b) stands in" in run.stdout and "confirmation: automatic" in run.stdout
    shown = json.loads(out.read_text())
    assert shown["state"] == "applied" and shown["verifier"]["accepted"]
    before, after = shown["before"], shown["after"]
    assert before["latency mean (ms)"] > 100  # the surge overloads URLLC under the default objective
    assert after["latency p95 (ms)"] < 10  # the stated bound, met once the executor has settled
    assert after["PRB share"] > before["PRB share"]
