"""The register of held-out sets already scored, kept in git (docs/learn/data/spent_sets.json).

Every held-out set (the goal test's `test` and `m09_test`, M08b's held-out intents) may be scored once.
The scripts that score them already refuse a second run when their result file exists, but those files
live in git-ignored folders (runs/, eval/results/): on a clean clone, or after `runs/` is deleted, the
old guard would let the set be scored again. This register is committed, so the guard survives.

- `refuse_if_spent(name)` is called before a final run, next to the existing file check.
- `mark_spent(name, ...)` is called right after a final run writes its result. It edits a tracked file,
  so the run leaves an uncommitted change that has to be committed with the result.
"""

import json
import subprocess
from datetime import date
from pathlib import Path

SPENT_FILE = Path("docs/learn/data/spent_sets.json")


def _load(path: Path | None) -> dict:
    path = path or SPENT_FILE  # read at call time, so a test can point it elsewhere
    return json.loads(path.read_text()) if path.exists() else {"sets": {}}


def spent(name: str, path: Path | None = None) -> dict | None:
    """The register's entry for this set, or None if it has never been scored."""
    return _load(path)["sets"].get(name)


def refuse_if_spent(name: str, path: Path | None = None) -> None:
    """Stop the program if the set is in the register."""
    entry = spent(name, path)
    if entry is not None:
        raise SystemExit(f"{name} was already scored on {entry['date']} (commit {entry['commit']}, "
                         f"{entry['result_file']}); it is spent, see {path or SPENT_FILE}")


def mark_spent(name: str, result_file: Path | str, note: str = "", path: Path | None = None) -> None:
    """Add the set to the register, with today's date and the commit the run used."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    register = _load(path)
    register["sets"][name] = {"date": date.today().isoformat(), "commit": commit or "unknown",
                              "result_file": str(result_file), "note": note}
    path = path or SPENT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, indent=1, ensure_ascii=False) + "\n")
