"""The evidence behind the results, fingerprinted in git and packed into one archive to copy off this machine.

The trained models, the MLflow registry (the champion), the testbed's KPI store and the raw runs of every
evaluation are git-ignored: too large for git, and some can't be regenerated (testbed runs, LLM runs, which
aren't deterministic). They exist only on this laptop. This script:
- `--manifest`: writes docs/learn/data/evidence_manifest.csv, one row per unit (a folder such as
  eval/results/m09b/m09b_dev, or a single file such as var/mlflow.db): file count, bytes and a tree SHA-256,
  the hash of the sorted "path<TAB>file sha256" lines. The CSV is committed: it says what the evidence was.
- `--verify`: recomputes every unit and lists the ones that differ, appeared or disappeared since.
- `--archive DEST.tar`: writes the manifest, then one tar of every unit plus the manifest. Copying it off
  the machine (another disk, a cloud drive) is Souhail's step.

  uv run python -m mlops.archive --manifest
  uv run python -m mlops.archive --verify
  uv run python -m mlops.archive --archive /mnt/d/backup/agentic-ric-evidence-2026-11-09.tar
"""

import argparse
import csv
import hashlib
import tarfile
from pathlib import Path

MANIFEST = Path("docs/learn/data/evidence_manifest.csv")
# (folder or file, depth): depth 0 = the path itself is one unit; 1 = each entry directly inside is a unit; ...
SOURCES: tuple[tuple[str, int], ...] = (
    ("models", 1),
    ("runs", 1),
    ("eval/results", 2),
    ("var/mlartifacts", 0),
    ("var/mlflow.db", 0),
    ("var/ric-testbed.db", 0),
)
FIELDS = ("unit", "files", "bytes", "sha256")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def units(root: Path = Path()) -> list[Path]:
    """Every unit, relative to root, in a stable order. A file found at a folder's depth is its own unit."""
    found = []
    for source, depth in SOURCES:
        level = [root / source] if (root / source).exists() else []
        for _ in range(depth):
            level = [child for p in level for child in (sorted(p.iterdir()) if p.is_dir() else [p])]
        found += [p.relative_to(root) for p in level]
    return sorted(set(found))


def fingerprint(unit: Path, root: Path = Path()) -> dict:
    """File count, bytes and tree hash of one unit (a file or a folder)."""
    path = root / unit
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    lines = "".join(f"{p.relative_to(root).as_posix()}\t{file_sha256(p)}\n" for p in files)
    return {"unit": unit.as_posix(), "files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "sha256": hashlib.sha256(lines.encode()).hexdigest()}


def build_manifest(root: Path = Path()) -> list[dict]:
    return [fingerprint(unit, root) for unit in units(root)]


def write_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def verify(root: Path = Path(), path: Path = MANIFEST) -> list[str]:
    """The units whose fingerprint differs from the manifest, as readable lines (empty: all match)."""
    with path.open() as f:
        recorded = {r["unit"]: r for r in csv.DictReader(f)}
    now = {r["unit"]: r for r in build_manifest(root)}
    problems = [f"missing: {u}" for u in recorded if u not in now]
    problems += [f"new: {u}" for u in now if u not in recorded]
    problems += [f"changed: {u} ({recorded[u]['files']} -> {now[u]['files']} files)" for u in now
                 if u in recorded and now[u]["sha256"] != recorded[u]["sha256"]]
    return problems


def archive(dest: Path, root: Path = Path()) -> list[dict]:
    """Write the manifest, then one tar with every unit and the manifest. Uncompressed: the bulk is
    already-compressed model zips and MLflow artifacts, so compressing would cost time for little."""
    rows = build_manifest(root)
    write_manifest(rows, root / MANIFEST)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w") as tar:
        for row in rows:
            tar.add(root / row["unit"], arcname=row["unit"])
        tar.add(root / MANIFEST, arcname=MANIFEST.as_posix())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", action="store_true", help=f"write {MANIFEST}")
    group.add_argument("--verify", action="store_true", help="compare the evidence with the manifest")
    group.add_argument("--archive", type=Path, metavar="DEST.tar", help="manifest + one tar of the evidence")
    args = parser.parse_args()
    if args.verify:
        problems = verify()
        print("\n".join(problems) if problems else f"all units match {MANIFEST}")
        raise SystemExit(1 if problems else 0)
    rows = archive(args.archive) if args.archive else build_manifest()
    if args.manifest:
        write_manifest(rows)
    total = sum(r["bytes"] for r in rows) / 1e9
    print(f"{len(rows)} units, {sum(r['files'] for r in rows)} files, {total:.2f} GB; manifest {MANIFEST}"
          + (f"; archive {args.archive}" if args.archive else ""))


if __name__ == "__main__":
    main()
