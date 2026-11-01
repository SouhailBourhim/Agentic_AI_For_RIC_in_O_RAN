"""Hard invariants 5 and 4 for the RL runner (M04a).

5: the runner never waits on an LLM or the agent. Checked statically (it imports none of them)
   and at run time (a tick stays fast while another process holds the database write lock).
4 (groundwork): only the runner calls apply_prb_split.
Plus: the runner follows the a1_policy timetable, switching at expiry with no gap.
"""

import ast
import time
from pathlib import Path

from conftest import start_cell
from store import a1_policy, db
from store.events import read_events
from verifier.objective import RLObjective, SliceValues

REPO = Path(__file__).resolve().parent.parent
REALTIME_MODULES = ["rl/runner.py", "rl/policy.py", "backends/base.py", "backends/sim.py",
                    "store/a1_policy.py", "store/kpi.py", "store/db.py", "store/events.py"]
# Nothing on the real-time path may import these: the agent, the console, HTTP, LLM clients.
FORBIDDEN_PREFIXES = ("agent", "console", "gateway", "fastapi", "uvicorn", "httpx", "requests",
                      "urllib", "ollama", "langchain", "langgraph", "openai", "anthropic", "mcp")


def imports_of(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_realtime_path_imports_no_agent_http_or_llm() -> None:
    for module in REALTIME_MODULES:
        bad = {n for n in imports_of(REPO / module) if n.split(".")[0] in FORBIDDEN_PREFIXES}
        assert not bad, f"{module} imports {sorted(bad)}"


def test_only_the_runner_calls_apply_prb_split() -> None:
    """Invariant 4: the one call that changes the radio appears in rl/runner.py only."""
    callers = set()
    for path in REPO.rglob("*.py"):
        if any(part in (".venv", "tests") for part in path.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "apply_prb_split":
                callers.add(str(path.relative_to(REPO)))
    assert callers == {"rl/runner.py"}


def test_runner_works_with_no_agent_and_no_policy(ric_db, tiny_policy_file) -> None:
    """No agent process, empty a1_policy: the runner runs on the default objective."""
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(20)
    assert runner.backend.now() == 20.0
    assert tuple(ric_db.execute("SELECT COUNT(*), MAX(policy_id) FROM kpi").fetchone()) == (60, None)


def test_tick_never_waits_for_a_busy_database(ric_db, tiny_policy_file) -> None:
    """Another connection holds the write lock: the tick still ends at once, keeps its KPI
    records, and writes them as soon as the lock is released."""
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(3)
    blocker = db.connect()
    blocker.execute("BEGIN IMMEDIATE")  # the write lock, as a long agent transaction would hold it
    start = time.monotonic()
    result = runner.tick()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"tick took {elapsed * 1000:.0f} ms"
    assert not result.db_ok and len(runner.pending) == 1
    blocker.execute("ROLLBACK")
    assert runner.tick().db_ok
    assert ric_db.execute("SELECT COUNT(DISTINCT ts) FROM kpi").fetchone()[0] == 5  # nothing lost


def test_runner_follows_timetable_and_switches_at_expiry(ric_db, tiny_policy_file) -> None:
    """Two segments written ahead: the runner tags each tick with the segment in force, and
    moves on at the exact second the first one expires, then back to the default."""
    protect = RLObjective(weights=SliceValues(embb=0.2, urllc=0.6, be=0.2),
                          min_thr_mbps=SliceValues(urllc=10.0), valid_from=5.0, valid_until=15.0)
    segments = [
        a1_policy.Segment(protect, ("i0001",), {"accepted": True}, "console", 5.0, 15.0),
        a1_policy.Segment(protect.model_copy(update={"valid_from": 15.0, "valid_until": 25.0}),
                          ("i0002",), {"accepted": True}, "console", 15.0, 25.0),
    ]
    a1_policy.write_schedule(ric_db, segments, "test", 0.0)
    runner = start_cell(ric_db, tiny_policy_file)
    runner.run(30)
    tags = dict(ric_db.execute("SELECT ts, policy_id FROM kpi WHERE slice = 'urllc'").fetchall())
    # A tick from t to t+1 is recorded at ts = t+1, with the policy read at its start, t.
    assert [tags[t] for t in (5.0, 6.0, 15.0, 16.0, 25.0, 26.0)] == [None, 1, 1, 2, 2, None]
    switches = [(e["ts_backend"], e["data"]["to"]) for e in read_events() if e["event"] == "policy_switch"]
    assert switches == [(5.0, 1), (15.0, 2), (25.0, None)]


def test_direct_share_mode_applies_the_given_split(ric_db, tiny_policy_file) -> None:
    """The LLM-only baseline (M04b) writes shares; the runner applies them, and is still the caller."""
    objective = RLObjective(weights=SliceValues(embb=0.4, urllc=0.3, be=0.3), valid_until=100.0)
    segment = a1_policy.Segment(objective, ("i0001",), {"accepted": True}, "auto-eval", 0.0, 100.0,
                                mode="direct_share", shares=(0.5, 0.3, 0.2))
    a1_policy.write_schedule(ric_db, [segment], "test", 0.0)
    runner = start_cell(ric_db, tiny_policy_file)
    assert runner.tick().shares == (0.5, 0.3, 0.2)
