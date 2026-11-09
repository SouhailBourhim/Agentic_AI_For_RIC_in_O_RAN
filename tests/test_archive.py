"""mlops/archive.py: the evidence manifest is stable, detects a changed, missing or new unit, and the archive
holds every unit and the manifest."""

import tarfile

from mlops import archive


def _evidence(root):
    (root / "models/m09c/dev").mkdir(parents=True)
    (root / "models/m09c/dev/td3_seed30.zip").write_bytes(b"model")
    (root / "eval/results/m09b/m09b_dev/f1_00").mkdir(parents=True)
    (root / "eval/results/m09b/m09b_dev/f1_00/kpi.csv").write_text("ts\n1\n")
    (root / "var").mkdir()
    (root / "var/mlflow.db").write_bytes(b"db")


def test_units_follow_each_source_depth(tmp_path):
    _evidence(tmp_path)
    assert [u.as_posix() for u in archive.units(tmp_path)] == [
        "eval/results/m09b/m09b_dev", "models/m09c", "var/mlflow.db"]


def test_manifest_is_stable_and_verify_finds_changes(tmp_path):
    _evidence(tmp_path)
    rows = archive.build_manifest(tmp_path)
    assert rows == archive.build_manifest(tmp_path)  # same files, same fingerprints
    manifest = tmp_path / "manifest.csv"
    archive.write_manifest(rows, manifest)
    assert archive.verify(tmp_path, manifest) == []
    (tmp_path / "eval/results/m09b/m09b_dev/f1_00/kpi.csv").write_text("ts\n2\n")
    (tmp_path / "var/mlflow.db").unlink()
    (tmp_path / "runs/m10").mkdir(parents=True)
    (tmp_path / "runs/m10/test_result.json").write_text("{}")
    problems = archive.verify(tmp_path, manifest)
    assert "missing: var/mlflow.db" in problems and "new: runs/m10" in problems
    assert any(p.startswith("changed: eval/results/m09b/m09b_dev") for p in problems)


def test_archive_holds_every_unit_and_the_manifest(tmp_path):
    _evidence(tmp_path)
    dest = tmp_path / "out/evidence.tar"
    archive.archive(dest, tmp_path)
    with tarfile.open(dest) as tar:
        names = set(tar.getnames())
    assert {"models/m09c/dev/td3_seed30.zip", "eval/results/m09b/m09b_dev/f1_00/kpi.csv", "var/mlflow.db",
            archive.MANIFEST.as_posix()} <= names
