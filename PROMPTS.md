# Checkpoint prompts for Claude Code

Paste one prompt at a time, in order. Each one asks for a plan first. Approve or correct the plan,
let it build, then read the learning note it writes before moving on.

After each checkpoint, do the "Understand it" step yourself. That is the part that prepares the defence.

---

## M00 — Repository and tooling

```
Read CLAUDE.md and docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf.
Checkpoint M00: set up the repository skeleton from the "Repository layout" in CLAUDE.md,
with uv, pytest, a minimal README, docs/DECISIONS.md and docs/learn/.
First give me a short plan in plain words and wait for my approval.
Done when: `uv run pytest` runs (even with zero tests) and everything is committed.
Then write docs/learn/M00-setup.md.
```

Understand it: be able to say what `uv`, `pyproject.toml` and `pytest` each do in one sentence.

---

## M01 — Simulator of one cell with three slices

```
Checkpoint M01: build the gymnasium environment in sim/.
One cell, a fixed PRB budget, three slices (eMBB, URLLC, best effort), each with a traffic
generator whose load can change over time (steady, spike, ramp).
The action is the PRB share per slice (continuous, sums to 1).
The observation contains per-slice load, throughput, latency and PRB use, PLUS the current
objective (weights per slice, minimum throughput targets, latency bound), because the RL policy
must be goal-conditioned (see CLAUDE.md).
Keep the throughput and latency models simple and documented (for example throughput
proportional to PRBs, latency from a queue model). Explain every formula in the learning note.
Plan first and wait for approval. Add tests: shares sum to 1, more PRBs never lowers throughput,
an overloaded slice shows rising latency.
Then write docs/learn/M01-simulator.md.
```

Understand it: explain the observation, the action and the reward to Ghita without notes.

---

## M02 — Goal-conditioned DDPG policy

```
Checkpoint M02: train a DDPG policy with stable-baselines3 on the M01 simulator.
During training, sample a new random objective at each episode so the policy learns to follow
whatever objective it is given. Log training curves to a file and save the model outside git.
Add a small evaluation script that shows the same policy behaving differently under two
different objectives (for example "protect URLLC" vs "maximise eMBB").
Compare against a static equal split on the same traffic.
Plan first, including the hyperparameters you choose and why.
Then write docs/learn/M02-ddpg.md, including what actor, critic, replay buffer and target
networks are, in plain words.
```

Understand it: explain why DDPG and not DDQN, and why the objective is in the observation.

---

## M03 — Objective schema and deterministic verifier

```
Checkpoint M03: build verifier/.
A pydantic schema for an RL objective (weights, min throughput per slice, latency bound,
validity window). A pure deterministic function verify(objective, current_state) -> accept or
reject with a human-readable reason. Checks: schema valid, weights in range and summing to 1,
targets physically feasible given the PRB budget, no slice starved below a floor, no change
larger than a configured step, validity window sane.
No LLM, no randomness, no network calls in this package.
Write table-driven tests with accepted and rejected cases, including tricky edge cases.
Plan first. Then write docs/learn/M03-verifier.md.
```

Understand it: explain each check and why a verifier must not be an LLM.

---

## M04 — LangGraph service against the simulator

```
Checkpoint M04: build agent/ as a FastAPI service with a LangGraph agent, using the
SimBackend through a Backend interface in backends/ (so the FlexRIC backend can replace it
later).
Loop: parse intent to structured JSON -> plan -> call one internal tool -> verifier -> apply
(or wait for confirmation in demo mode) -> observe KPIs -> reflect -> loop or finish.
Internal tools: get_kpis, set_rl_objective, wait_and_observe; set_slice_share only enabled for
the "LLM only" baseline mode.
Intent states: received, interpreted, rejected (with reason), awaiting_confirmation, applied,
expired. Expose HTTP endpoints for submit_intent, get_status and get_kpis.
Build console/ as a small CLI that lists pending objectives and confirms or refuses them.
LLM through Ollama behind a small client class so a cloud model can be swapped in.
Plan first, including the prompt for intent parsing. Add tests with a fake LLM so they run
without a GPU. Then write docs/learn/M04-langgraph.md with a Mermaid diagram of the graph.
```

Understand it: trace one intent from HTTP request to applied objective, naming each node.

---

## M05 — MCP gateway and Hermes (minimal prototype)

```
Checkpoint M05: build gateway/ as an MCP server (official Python SDK) exposing exactly three
tools: submit_intent, get_kpis, get_status. It only calls the LangGraph HTTP API.
Add tests that fail if any other tool is exposed.
Give me a Hermes Agent config example that registers only these three tools (MCP `include`
whitelist) and the steps to disable Hermes general tools; check the current Hermes docs for
the exact keys and tell me what you could not confirm.
Then run the demo scenario from the proposal (section 5.4) against the simulator:
accepted intent with confirmation, rejected out-of-bounds intent, and the negative test
"set the PRB split to 90/5/5" producing no state change. Log tool calls with timestamps.
Plan first. Then write docs/learn/M05-gateway-hermes.md.
```

Understand it: explain why Hermes cannot touch the radio even if its LLM misbehaves.

---

## M06 — Testbed: OAI core, gNB and UE in rfsim (supervised)

```
Checkpoint M06: bring up OAI CN5G, the OAI gNB and 2 nrUE in rfsim mode in testbed/, using
the official OAI docs and Docker images. Configure slices if supported at this stage.
Goal: each UE gets an IP and can ping through the core, and iperf3 traffic flows.
Ask me before any sudo command or system change. Show real logs when something fails.
Plan first, telling me which OAI documentation version you follow.
Then write docs/learn/M06-testbed.md, including which container does what and which
3GPP interfaces (N2, N3, F1 if used) carry what.
```

Understand it: draw the testbed on paper with every container and interface.

---

## M07 — FlexRIC, KPM xApp and slice-control xApp

```
Checkpoint M07: build FlexRIC, connect it to the gNB, and write xapps/:
1) a KPM monitoring xApp that stores per-slice throughput, PRB use and latency in the KPI store;
2) a slice-control xApp that applies a PRB split and reads the emulated a1_policy (see
   CLAUDE.md) to get the current objective, then runs the M02 policy.
Implement FlexRICBackend so the M04 service runs unchanged on the real testbed.
Start from the official FlexRIC examples and tell me which ones.
Plan first. Then write docs/learn/M07-xapps.md, including what an E2 service model is
(KPM, slice control) in plain words.
```

Understand it: explain the difference between E2 reports and E2 control, with your own xApps as examples.

---

## M08 — Evaluation

```
Checkpoint M08: build eval/.
Part 1, control: run the 4 controllers (static equal split, RL only, LLM only, LLM + RL) on the
scripted scenarios (steady then spike, new intent mid-run, two conflicting intents), on the
simulator first, then on the testbed. Metrics: share of time each intent is met, throughput and
latency per slice, reaction time, unsafe actions proposed and blocked, LLM latency and cost.
No Hermes in this part; intents are injected directly.
Part 2, interface: on a set of hand-annotated intents (I will write them; give me a template),
measure Hermes/interface latency, transmission fidelity, extraction accuracy per field,
tool calls succeeded / rejected by gateway / rejected by verifier / errored, and out-of-scope
attempts. Keep the two parts in separate result files and separate plots.
Repeat runs and report spread, not single numbers. Never hard-code results.
Plan first. Then write docs/learn/M08-evaluation.md explaining how to read every plot.
```

Understand it: for each plot, say in one sentence what it shows and why it matters.

---

## Any time

```
Explain <file or function> step by step, as if I had to defend it tomorrow.
```

```
Quiz me with 5 questions on checkpoint MXX, one at a time, and correct my answers.
```

```
What did you simplify or assume in MXX that a real deployment would do differently?
```
