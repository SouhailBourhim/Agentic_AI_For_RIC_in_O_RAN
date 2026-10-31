# Decisions

One line per decision: date — decision — reason. Newest at the bottom.

- 2026-10-31 — Use `uv` for Python and dependencies; commit `uv.lock`. — One tool for venv + install + run; the lock file gives every machine identical versions.
- 2026-10-31 — Python 3.12 (`.python-version`), `requires-python >= 3.11`. — CLAUDE.md asks for 3.11+; 3.12 is current and supported by gymnasium, SB3 and LangGraph. uv uses its own managed 3.12.14, not Ubuntu's 3.12.3, so system updates cannot break the project.
- 2026-10-31 — Flat layout (packages at the repo root, `[tool.uv] package = false`, pytest `pythonpath = ["."]`) instead of a `src/` layout installed as a package. — Fewer concepts to explain; we never publish this code as a library. Rejected `src/` layout: it protects against import mistakes in published libraries, which is not our risk.
- 2026-10-31 — Add dependencies per checkpoint, not up front. — Each checkpoint only pulls what it needs (e.g. torch only in M02), so every dependency has a reason we can explain.
- 2026-10-31 — Smoke test `tests/test_skeleton.py` that imports every package. — pytest exits with code 5 when it collects zero tests; a real test makes `uv run pytest` exit 0 and checks the layout.
- 2026-10-31 — No linter, CI or pre-commit for now. — Not needed yet; less to explain. Can add `ruff` later.
