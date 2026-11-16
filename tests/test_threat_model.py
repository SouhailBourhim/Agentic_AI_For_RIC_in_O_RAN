"""M11c: docs/THREAT_MODEL.md stays true. Every test it cites exists, every hard invariant has a row, and every
residual risk says it has no test. A renamed or deleted test then fails here instead of leaving a dead line.

The tests are found with ast, not imported: this file checks names, not behaviour (the cited tests do that).
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "docs" / "THREAT_MODEL.md"
REFERENCE = re.compile(r"`(tests/test_\w+\.py)::(test_\w+)`")
N_INVARIANTS = 6  # CLAUDE.md, "Hard invariants"


def section(title_start: str) -> str:
    """The text of the `## ` section whose title starts with title_start, up to the next `## `."""
    text = PAGE.read_text()
    start = text.index(f"\n## {title_start}")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def table_rows(text: str, first_cell: str) -> list[str]:
    """The table rows whose first cell matches first_cell (a regex), e.g. 'T\\d+'."""
    return [line for line in text.splitlines() if re.match(rf"\|\s*{first_cell}\s*\|", line)]


def defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


REFERENCES = sorted(set(REFERENCE.findall(PAGE.read_text())))


def test_the_page_cites_tests() -> None:
    assert len(REFERENCES) >= 40  # 21 threats, most with several tests: a broken pattern would find none


@pytest.mark.parametrize("file, name", REFERENCES, ids=[f"{f}::{n}" for f, n in REFERENCES])
def test_every_cited_test_exists(file: str, name: str) -> None:
    path = REPO / file
    assert path.exists(), f"{file} is cited in docs/THREAT_MODEL.md but does not exist"
    assert name in defined_tests(path), f"{file}::{name} is cited in docs/THREAT_MODEL.md but not defined"


def test_every_threat_names_at_least_one_test() -> None:
    rows = table_rows(section("4."), r"T\d+")
    assert len(rows) >= 20
    for row in rows:
        assert REFERENCE.search(row), f"a threat with no test: {row[:80]}"


@pytest.mark.parametrize("invariant", range(1, N_INVARIANTS + 1))
def test_every_hard_invariant_has_a_row(invariant: int) -> None:
    rows = table_rows(section("4."), r"T\d+")
    assert any(f"Inv. {invariant})" in row or f"Inv. {invariant}," in row for row in rows), \
        f"hard invariant {invariant} has no row in docs/THREAT_MODEL.md"


def test_every_residual_risk_says_it_has_no_test() -> None:
    rows = table_rows(section("5."), r"R\d+")
    assert len(rows) >= 5
    for row in rows:
        assert "no test" in row, f"a residual risk must say 'no test': {row[:80]}"
        assert not REFERENCE.search(row), f"a residual risk cites a test; move it to section 4: {row[:80]}"
