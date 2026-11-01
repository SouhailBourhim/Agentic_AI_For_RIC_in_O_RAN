"""Hard invariant 3: the verifier is plain deterministic Python — no LLM, no randomness, no network.

Three complementary checks:
1. static: the decision modules import only an allowlist of modules (read from the source, so
   an import hidden in a function body is caught too);
2. determinism: the same inputs give the same decision, every time;
3. no network at run time: verify() works with sockets disabled.

verifier/plot.py draws the feasibility map (matplotlib, numpy). It is a viewer, not on the
decision path: it is left out of the scan, and no decision module may import it.
"""

import ast
import socket
from pathlib import Path

import pytest

from verifier.merge import ActiveIntent
from verifier.objective import RLObjective
from verifier.verify import VerifierState, reverify_after_removal, verify

VERIFIER_DIR = Path(__file__).resolve().parent.parent / "verifier"
DECISION_MODULES = ["__init__.py", "objective.py", "merge.py", "verify.py"]
VIEWER_MODULES = ["plot.py"]

# Top-level module names the decision code may import. Everything else is refused, which
# covers random, numpy, time, datetime, os, subprocess, socket, urllib, http, requests,
# httpx, ollama, langchain, langgraph, mcp, sqlite3...
ALLOWED_IMPORTS = {"__future__", "collections", "dataclasses", "math", "pydantic", "typing", "verifier"}


def imported_modules(path: Path) -> set[str]:
    """Every module named in an import statement of the file (full dotted names)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_every_verifier_file_is_classified() -> None:
    """A new file in verifier/ must be added to one of the two lists, so it cannot skip the scan."""
    files = {p.name for p in VERIFIER_DIR.glob("*.py")}
    assert files == set(DECISION_MODULES) | set(VIEWER_MODULES)


@pytest.mark.parametrize("module", DECISION_MODULES)
def test_decision_module_imports_only_allowlist(module: str) -> None:
    imports = imported_modules(VERIFIER_DIR / module)
    refused = {name for name in imports if name.split(".")[0] not in ALLOWED_IMPORTS}
    assert not refused, f"verifier/{module} imports {sorted(refused)}"


@pytest.mark.parametrize("module", DECISION_MODULES)
def test_decision_module_does_not_import_the_viewer(module: str) -> None:
    assert "verifier.plot" not in imported_modules(VERIFIER_DIR / module)


def test_no_builtin_escape_hatches() -> None:
    """No dynamic import, eval/exec or file access that would bypass the import scan."""
    forbidden = {"__import__", "eval", "exec", "open", "compile"}
    for module in DECISION_MODULES:
        tree = ast.parse((VERIFIER_DIR / module).read_text())
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls & forbidden, f"verifier/{module} calls {sorted(calls & forbidden)}"


ACTIVE = ActiveIntent(
    intent_id="be20",
    issued_at=50.0,
    objective=RLObjective(
        weights={"embb": 0.3, "urllc": 0.3, "be": 0.4}, min_thr_mbps={"be": 20.0},
        valid_from=50.0, valid_until=700.0,
    ),
)
STATE = VerifierState(now_s=100.0, capacity_mbps=75.0, active=(ACTIVE,))
PROPOSALS = [
    {"weights": {"embb": 0.1, "urllc": 0.8, "be": 0.1}, "min_thr_mbps": {"urllc": 10.0},
     "max_latency_ms": 10.0, "valid_from": 100.0, "valid_until": 700.0},  # accepted
    {"weights": {"embb": 0.1, "urllc": 0.8, "be": 0.1}, "min_thr_mbps": {"embb": 30.0, "urllc": 22.0},
     "max_latency_ms": 10.0, "valid_from": 100.0, "valid_until": 700.0},  # conflict
    {"weights": {"embb": 0.5, "urllc": 0.5}},  # schema
]


@pytest.mark.parametrize("prop", PROPOSALS)
def test_same_inputs_same_decision(prop: dict) -> None:
    first = verify(prop, STATE)
    assert all(verify(prop, STATE) == first for _ in range(100))


def test_works_with_network_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """If anything in verify() tried to open a socket, this would raise instead of deciding."""

    def no_network(*args, **kwargs):
        raise AssertionError("the verifier tried to use the network")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for prop in PROPOSALS:
        verify(prop, STATE)
    reverify_after_removal(STATE)
