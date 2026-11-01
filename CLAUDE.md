# Agentic AI for the RIC in O-RAN — project guide for Claude Code

Personal project by Souhail Bourhim.
The reference design is `docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf` (French). Read it before planning anything.

## The most important rule: I must understand everything

The code is written with AI help. I must still be able to explain every part of it without notes.
So:

- Prefer simple, readable code over clever code. No unnecessary abstractions, no frameworks the
  task does not need. Short functions, explicit names, type hints, docstrings that say *why*.
- Before writing code for a checkpoint, present a short plan in plain words and wait for approval.
- After each checkpoint, write `docs/learn/MXX-<name>.md` (format below). A checkpoint is not done
  until that note exists.
- When asked "explain <file or concept>", walk through it step by step for someone who knows
  Python, telecom basics (LTE/5G, PRB, slicing) and classic ML, but not this specific library.
- When you make a non-obvious choice (library, parameter, simplification), say so in the note,
  with the alternative you rejected and why.

### Format of a learning note (`docs/learn/MXX-<name>.md`)

1. What we built (3–5 sentences).
2. Why it exists in the architecture (which box of the proposal's figure 2 it is).
3. How it works, as a numbered walk through the main code path, with file:line references.
4. A small Mermaid diagram if there is a flow or a loop.
5. How to run it and how to check it works (exact commands, expected output), and at least
   one visualization of the result (figure embedded in the note, dashboard, notebook or
   packet capture; see `docs/PLAN.md` §6) with the command that produces it and how to read it.
6. Choices and simplifications we made, and what a real system would do differently.
7. Five questions to check understanding, each with a short correct answer.

## Architecture (do not change without asking)

```
Operator → Hermes Agent (chat UI, MCP client)
        → MCP gateway: ONLY submit_intent, get_kpis, get_status
        → LangGraph service (HTTP): parses the intent, proposes an RL objective
        → Deterministic verifier: bounds + safety checks (+ operator confirmation in demo mode)
        → RL xApp (DDPG): decides the PRB split per slice, fast loop
        → RAN: OAI gNB + nrUE (rfsim) + OAI CN5G, slices eMBB / URLLC / best effort
KPM monitoring xApp → KPI store → read by the RL xApp, the LangGraph service and get_kpis
```

The same LangGraph service, verifier and RL policy must run against two backends behind one
interface: the Python simulator (default) and the real testbed (FlexRIC). Swapping the backend must
not require changes in the upper layers.

## Hard invariants (never break these; add a test for each)

1. Hermes and the MCP gateway expose exactly three tools: `submit_intent`, `get_kpis`, `get_status`.
   Nothing else, ever. `submit_intent` only creates a request; it never applies anything.
2. Neither Hermes, the gateway nor any LLM can send E2 commands, control the radio or set the PRB
   split directly. The gateway has no network route or client to FlexRIC or OAI.
3. Every change to the RL objective goes through the verifier. The verifier is plain deterministic
   Python: no LLM, no randomness, no network calls. It returns accept/reject with a reason.
4. Only the RL xApp sets the PRB split in the proposed controller. The internal tool
   `set_slice_share` exists only for the "LLM only" baseline controller, and still goes through the
   verifier. It is never exposed to Hermes.
5. No LLM call is on the real-time path. The RL xApp loop never waits for an LLM.
6. In demo mode, an accepted objective waits for operator confirmation in a separate console (not in
   Hermes) and expires after a configurable timeout. Automated evaluation runs skip confirmation.

## Design detail that matters: the RL policy must be goal-conditioned

The LLM changes the RL *objective* at run time (`set_rl_objective(weights, constraints)`), so the
policy must take the objective as part of its input (weights per slice, minimum throughput targets,
latency bound). Train it on randomly sampled objectives so it generalises. Otherwise a new goal would
need retraining, which defeats the design.

## Note on "A1 policy"

FlexRIC is a Near-RT RIC; there is no real Non-RT RIC / A1 interface in this stack. We emulate the
A1 policy channel with a small, explicit mechanism (for example a JSON policy published to a local
key-value store or an HTTP endpoint the RL xApp polls). Name it `a1_policy` in code and state clearly
in the docs that it is an emulation.

## Repository layout (target)

```
sim/            gym environment of one cell with 3 slices
rl/             DDPG training + policy wrapper (goal-conditioned)
verifier/       objective schema (pydantic) + deterministic checks
agent/          LangGraph service (FastAPI), internal tools, intent parsing
gateway/        MCP server with the 3 tools; Hermes config example
console/        operator confirmation console (CLI)
backends/       Backend interface + SimBackend + FlexRICBackend
xapps/          KPM monitoring xApp, slice-control xApp (FlexRIC)
testbed/        docker-compose and configs for OAI CN5G, gNB, nrUE, FlexRIC; traffic scripts
eval/           scenarios, controller runners, interface metrics, plots
monitoring/     Prometheus exporter (reads var/ric.db) + Prometheus/Grafana compose and dashboards
notebooks/      marimo notebooks (plain .py) to explore each checkpoint interactively
mlops/          MLflow tracking, backfill, model registry and promotion gate (store in var/)
tests/          pytest; one test per hard invariant at least
docs/learn/     learning notes, one per checkpoint
```

## Tech stack

- Python 3.11+, managed with `uv`. pytest. pydantic for all schemas. FastAPI for HTTP services.
- gymnasium for the simulator, stable-baselines3 for DDPG.
- LangGraph for the agent. LLM via Ollama (local, ~8B model with tool calling, e.g. Qwen), behind a
  small client so a cloud API model can be swapped in for comparison.
- Official MCP Python SDK for the gateway.
- MLOps: local MLflow (tracking, model registry with a promotion gate), GitHub Actions for tests.
- Visualization: matplotlib (object API) for figures, TensorBoard for RL training curves,
  marimo for interactive notebooks, Prometheus + Grafana for live monitoring (from M04a),
  Wireshark on pcaps for the testbed interfaces.
- OAI and FlexRIC via their official docs and Docker images where possible.

## Environment

- Windows laptop, Ubuntu under WSL2 (Ryzen 9, 32 threads, 32 GB RAM, ~24 GB given to WSL,
  RTX 5070 8 GB VRAM, ~6.6 GB usable). SCTP kernel module loads in WSL2.
- Only ONE local LLM fits in VRAM: Hermes and the LangGraph service share the same Ollama model.
- OAI/FlexRIC docs target Ubuntu 22.04; if a native build fails on 24.04, say so and propose using a
  22.04 WSL distro or Docker instead of patching the build.

## How to work

- One checkpoint at a time, in the order of `PROMPTS.md`. Plan first, then build, then test, then
  write the learning note, then stop and summarise.
- Small commits with clear messages. Do not commit secrets, model weights or large logs.
- Never mention Claude as a co-author in any commit: no `Co-Authored-By: Claude ...` trailer and
  no other Claude attribution line in commit messages or PR descriptions.
- Ask before: any `sudo` command, installing system packages, changing WSL/Docker/network settings,
  deleting files outside the repo, or anything that changes the design above.
- When something fails, show the real error and your diagnosis. Do not hide failures, do not fake
  results, never hard-code numbers into evaluation outputs.
- Keep a running `docs/DECISIONS.md`: one line per decision (date, decision, reason).

## Open assumptions from the proposal (to confirm as we go)

H1 Hermes install on WSL2 and version not yet tested. H2 Hermes forwards the operator text
verbatim; extraction is done by LangGraph. H3 Hermes general tools (terminal, files, web) can be
disabled; run it isolated. H4 confirmation console separate from Hermes, with timeout. H5 the
gateway only talks to the LangGraph HTTP API. H6 `set_slice_share` only for the "LLM only"
baseline. H7 the 4-controller comparison runs without Hermes. H8 ~40 hand-annotated intents for the
extraction evaluation.
