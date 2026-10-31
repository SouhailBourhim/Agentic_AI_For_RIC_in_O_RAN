# Decisions

One line per decision: date — decision — reason. Newest at the bottom.

- 2026-10-31 — Use `uv` for Python and dependencies; commit `uv.lock`. — One tool for venv + install + run; the lock file gives every machine identical versions.
- 2026-10-31 — Python 3.12 (`.python-version`), `requires-python >= 3.11`. — CLAUDE.md asks for 3.11+; 3.12 is current and supported by gymnasium, SB3 and LangGraph. uv uses its own managed 3.12.14, not Ubuntu's 3.12.3, so system updates cannot break the project.
- 2026-10-31 — Flat layout (packages at the repo root, `[tool.uv] package = false`, pytest `pythonpath = ["."]`) instead of a `src/` layout installed as a package. — Fewer concepts to explain; we never publish this code as a library. Rejected `src/` layout: it protects against import mistakes in published libraries, which is not our risk.
- 2026-10-31 — Add dependencies per checkpoint, not up front. — Each checkpoint only pulls what it needs (e.g. torch only in M02), so every dependency has a reason we can explain.
- 2026-10-31 — Smoke test `tests/test_skeleton.py` that imports every package. — pytest exits with code 5 when it collects zero tests; a real test makes `uv run pytest` exit 0 and checks the layout.
- 2026-10-31 — No linter, CI or pre-commit for now. — Not needed yet; less to explain. Can add `ruff` later.
- 2026-10-31 — Adopt `docs/PLAN.md` as the overall plan for M01–M08. — Fixes the order, the interfaces between the simulator and testbed paths, and how each hypothesis gets checked, before any code is written.
- 2026-10-31 — Split M04, M05, M07, M08 into sub-checkpoints (M04a/b, M05a/b, M07a/b, M08a/b/c) and add a timeboxed M06 spike after M03; PROMPTS.md updated to match. — The original checkpoints were each several days of unrelated work; the spike checks what the real gNB can report and control before M04a freezes the KPI record and Backend interface.
- 2026-10-31 — The pydantic `RLObjective` schema is built in M01, not M03. — The M01 observation contains the objective and M02 samples random objectives, so the schema is needed before the verifier.
- 2026-10-31 — KPI store and emulated `a1_policy` are two tables in one SQLite file (`var/ric.db`), built in M04a. — Several processes (KPM xApp, RL runner, agent) must share them; SQLite is in the standard library and needs no server. Rejected Redis: an extra service to run and explain.
- 2026-10-31 — The RL runner is the only caller of `Backend.apply_prb_split`; the LLM-only baseline goes through it with `a1_policy.mode = direct_share`. — Keeps invariant 4 true by construction: no LLM-side code ever touches the RAN.
- 2026-10-31 — Policy observations use throughput and targets as fractions of cell capacity. — The policy is trained before the testbed capacity is known; normalising lets us recalibrate the sim without changing the policy's input.
- 2026-10-31 — Conflicting intents: the verifier checks all active objectives together and rejects the newer one if they are infeasible. — Deterministic, easy to explain and test. Rejected "newest wins" (silently drops a commitment) and fixed slice priority (arbitrary).
- 2026-10-31 — Train DDPG on CPU by default. — Small MLPs are fast on 32 threads; avoids the RTX 5070 / CUDA ≥ 12.8 PyTorch requirement and leaves the GPU to Ollama.
- 2026-10-31 — Docker Desktop with WSL integration; Ollama runs on Windows (`localhost:11434` from WSL). — Already installed and working; SCTP loaded in WSL is visible inside Docker Desktop containers.
- 2026-10-31 — Testbed uses 3 nrUE, one per slice. — Same three slices as the simulator, so the policy and verifier need no special case.
- 2026-10-31 — Cuttable if late: Hermes (→ CLI client on the same 3 tools), cloud-model comparison, analyst agent (dropped now). No cloud API key yet; the LLM client stays swappable. — Testbed control and testbed evaluation are kept as priorities.
