# Agentic AI for the RIC in O-RAN

Personal project by Souhail Bourhim.

An operator states a goal in plain language. An LLM agent (LangGraph) turns it into an objective for a
goal-conditioned RL xApp (DDPG) that allocates PRBs between slices in the Near-RT RIC. A deterministic
verifier checks every objective before it is applied. Hermes Agent is only the operator chat interface,
behind an MCP gateway that exposes three tools. Testbed: OpenAirInterface (rfsim) + OAI 5G core + FlexRIC.

- Design: [`docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf`](docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf) (French)
- Rules for the coding agent: [`CLAUDE.md`](CLAUDE.md)
- Overall plan (checkpoints, order, interfaces, risks): [`docs/PLAN.md`](docs/PLAN.md)
- Build order, one prompt per checkpoint: [`PROMPTS.md`](PROMPTS.md)
- Learning notes, one per checkpoint: `docs/learn/` (written as the project progresses)

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (it installs the right Python by itself).

```bash
uv sync          # create .venv and install the locked dependencies
uv run pytest    # run the tests
```

## Layout

| Folder | What goes there |
|---|---|
| `sim/` | gymnasium environment: one cell, three slices |
| `rl/` | DDPG training and the goal-conditioned policy |
| `verifier/` | objective schema and deterministic checks (no LLM) |
| `agent/` | LangGraph service (FastAPI): intent → RL objective |
| `gateway/` | MCP server with exactly 3 tools: `submit_intent`, `get_kpis`, `get_status` |
| `console/` | operator confirmation console (demo mode) |
| `backends/` | one interface, two backends: simulator and FlexRIC |
| `xapps/` | FlexRIC xApps: KPM monitoring, slice control |
| `testbed/` | Docker/configs for OAI CN5G, gNB, nrUE (rfsim), FlexRIC |
| `eval/` | scenarios, controller comparison, interface metrics, plots |
| `tests/` | pytest |
| `docs/` | design, [`DECISIONS.md`](docs/DECISIONS.md), learning notes in `docs/learn/` |

Status: M00 (skeleton) and M01 (simulator + objective schema) done; overall plan in `docs/PLAN.md`. Next: M02 (DDPG policy).
