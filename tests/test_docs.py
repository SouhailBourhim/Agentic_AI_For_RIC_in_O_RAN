"""M12: the reader's entry points stay true. Every relative link in the README, the claims table
and the notes index points at a file that exists, and every learning note is in the index exactly once."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENTRY_POINTS = ["README.md", "docs/CLAIMS.md", "docs/learn/README.md"]  # THREAT_MODEL: tests/test_threat_model.py
LINK = re.compile(r"\]\(([^)\s]+)\)")  # [text](target), markdown links and images


def relative_links(page: Path) -> list[str]:
    """Targets that are files in the repository: no web addresses, no in-page anchors."""
    targets = [t for t in LINK.findall(page.read_text()) if not re.match(r"[a-z]+:", t) and not t.startswith("#")]
    return [t.split("#")[0] for t in targets]


@pytest.mark.parametrize("page", ENTRY_POINTS)
def test_every_relative_link_resolves(page: str) -> None:
    path = REPO / page
    links = relative_links(path)
    assert links, f"{page}: no links found, the pattern is broken"
    missing = [t for t in links if not (path.parent / t).exists()]
    assert not missing, f"{page} links to files that don't exist: {missing}"


def test_every_note_is_in_the_index_once() -> None:
    """Under "All notes", one line per learning note: none missing, none twice."""
    index = (REPO / "docs" / "learn" / "README.md").read_text()
    lines = [line for line in index.split("## All notes", 1)[1].splitlines() if line.startswith("- [")]
    listed = [Path(LINK.search(line).group(1)).name for line in lines]
    notes = sorted(p.name for p in (REPO / "docs" / "learn").glob("M*.md"))
    assert sorted(listed) == notes, f"missing {set(notes) - set(listed)}, extra {set(listed) - set(notes)}"
    assert len(listed) == len(set(listed)), "a note is listed twice"
