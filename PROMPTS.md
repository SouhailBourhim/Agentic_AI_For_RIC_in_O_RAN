# Checkpoint prompts for Claude Code

Paste one prompt at a time, in order. Each one asks for a plan first. Approve or correct the plan,
let it build, then read the learning note it writes before moving on.

After each checkpoint, do the "Understand it" step yourself. That is the part that makes sure you really understand it.

The order below follows [`docs/PLAN.md`](docs/PLAN.md), which also gives the interfaces
(Backend, KPI record, RL objective, `a1_policy`) that every checkpoint must respect.

Every checkpoint also delivers the visualization listed for it in `docs/PLAN.md` §6, shown in
section 5 of its learning note.

Working order: M00 → M01 → M06-spike → M02 → M03 → M04a → M04b → M05a → M06 → M07a → M05b →
M07b → M08a → M08b → M08c.

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
Checkpoint M01: build the gymnasium environment in sim/, and the pydantic RLObjective schema
in verifier/ (fields as in docs/PLAN.md section 3; the checks come in M03).
One cell, a fixed PRB budget, three slices (eMBB, URLLC, best effort), each with a traffic
generator whose load can change over time (steady, spike, ramp).
The action is the PRB share per slice (continuous, sums to 1).
The observation contains per-slice load, throughput, latency and PRB use, PLUS the current
objective (weights per slice, minimum throughput targets, latency bound), because the RL policy
must be goal-conditioned (see CLAUDE.md). Throughputs and targets are normalised by the cell
capacity, and the capacity is a parameter, so we can calibrate it to the testbed later.
Keep the throughput and latency models simple and documented (for example throughput
proportional to PRBs, latency from a queue model). Explain every formula in the learning note.
Plan first and wait for approval. Add tests: shares sum to 1, more PRBs never lowers throughput,
an overloaded slice shows rising latency, the schema rejects weights that do not sum to 1.
Then write docs/learn/M01-simulator.md.
```

Understand it: explain the observation, the action and the reward out loud, without notes.

---

## M06-spike — Testbed reconnaissance (timeboxed, 1–2 days)

```
Checkpoint M06-spike: a timeboxed reconnaissance of the testbed, right after M01 and before
M04a freezes the interfaces. OAI and FlexRIC now live on GitHub (duranta-project); follow the
official v3.0.0 docs and Docker images (Docker Desktop, WSL2, SCTP loaded):
1) bring up OAI CN5G + gNB (E2 agent) + FlexRIC + 3 nrUE in rfsim; each UE pings through the core;
2) measure the rfsim downlink capacity with iperf (one UE alone, then 3 together) and calibrate
   the M01 simulator to it;
3) find out which KPM measurements the OAI gNB reports (throughput, PRB use, delay; per UE or
   per slice), and which slice-control path exists.
Ask me before any sudo command or system change. Show real logs when something fails.
Plan first, telling me which OAI and FlexRIC versions you follow.
Output: docs/learn/M06-spike.md with findings, a throughput figure and the protocol message
sequence, and the changes to the KPI record, Backend interface and capacity in docs/PLAN.md.
Stop when the timebox ends, even if not everything works.
```

Understand it: say what the spike changed in the plan, and why it had to come before M04a.

---

## M02 — Goal-conditioned DDPG policy

```
Checkpoint M02: train a DDPG policy with stable-baselines3 on the M01 simulator, on CPU.
During training, sample a new random objective at each episode so the policy learns to follow
whatever objective it is given. Log training curves to TensorBoard (runs/) and save the model
outside git. Add matplotlib figures of the comparison, and a marimo notebook to pick an
objective and watch the policy's PRB split.
Add a small evaluation script that shows the same policy behaving differently under two
different objectives (for example "protect URLLC" vs "maximise eMBB").
Compare against a static equal split on the same traffic.
Plan first, including the hyperparameters you choose and why.
Then write docs/learn/M02-ddpg.md, including what actor, critic, replay buffer and target
networks are, in plain words.
```

Understand it: explain why DDPG and not DDQN, and why the objective is in the observation.

---

## M03 — Deterministic verifier

```
Checkpoint M03: build verifier/ around the RLObjective schema from M01.
A pure deterministic function verify(objective, current_state) -> accept or reject with a
human-readable reason. Checks: schema valid, weights in range and summing to 1, targets
physically feasible given the PRB budget, no slice starved below a floor, no change larger than
a configured step, validity window sane, and the conflict rule from docs/PLAN.md: all active
objectives together must stay feasible, otherwise the newer one is rejected.
No LLM, no randomness, no network calls in this package; add a test that checks this
(hard invariant 3).
Write table-driven tests with accepted and rejected cases, including tricky edge cases.
Add a feasibility map figure (two targets on the axes, coloured by decision and reason).
Plan first. Then write docs/learn/M03-verifier.md.
```

Understand it: explain each check and why a verifier must not be an LLM.

---

## M04a — Plumbing: backend, KPI store, a1_policy, RL runner, service, console

```
Checkpoint M04a: build the plumbing with no LLM, following the interfaces in docs/PLAN.md.
- backends/: the Backend interface and SimBackend (wraps the M01 simulator, clock can run
  faster than real time or at real time).
- KPI store and the emulated a1_policy as two tables in SQLite (var/ric.db). State clearly
  in code and docs that a1_policy is an emulation of A1.
- rl/runner: the "RL xApp" loop. Every tick: read a1_policy (never block; default objective if
  none), run the M02 policy, call Backend.apply_prb_split, write KPI records. In direct_share
  mode it applies the given shares instead. It is the only caller of apply_prb_split.
- agent/: FastAPI service with endpoints submit_intent, get_status, get_kpis, and the intent
  state machine (received, interpreted, rejected with reason, awaiting_confirmation, applied,
  expired). Use a simple rule-based parser for now; M04b replaces it with the LLM.
- console/: a small CLI that lists pending objectives and confirms or refuses them; pending
  objectives expire after a configurable timeout; evaluation mode skips confirmation.
- One JSONL event log format for agent and verifier decisions.
- monitoring/: a Prometheus exporter that reads var/ric.db and exposes /metrics, plus a
  docker-compose with Prometheus and Grafana and a provisioned dashboard (per-slice KPIs,
  PRB split, active a1_policy, intent states, verifier decisions).
Tests: hard invariant 5 (the runner never waits on an LLM or the agent), hard invariant 6
(expiry, eval mode skips confirmation), an intent that the verifier rejects never reaches
a1_policy.
Plan first. Then write docs/learn/M04a-plumbing.md with a Mermaid diagram of the processes.
```

Understand it: trace one intent from HTTP request to a changed PRB split, naming each process
and table it touches.

---

## M04b — LangGraph agent

```
Checkpoint M04b: replace the rule-based parser with a LangGraph agent inside agent/.
Loop: parse intent to structured JSON -> plan -> call one internal tool -> verifier -> apply
(or wait for confirmation in demo mode) -> observe KPIs -> reflect -> loop or finish.
Internal tools: get_kpis, set_rl_objective, wait_and_observe; set_slice_share only enabled for
the "LLM only" baseline mode (it writes a1_policy in direct_share mode, still behind the
verifier).
LLM through Ollama (running on Windows, localhost:11434) behind a small client class so a cloud
model can be swapped in later.
Give me a template for hand-annotated intents (French and English, including ambiguous ones);
part of them tunes the prompt, a held-out part is kept for M08b.
Plan first, including the prompt for intent parsing. Add tests with a fake LLM so they run
without a GPU, and a test for hard invariant 4 (set_slice_share only exists in LLM-only mode).
Then write docs/learn/M04b-langgraph.md with a Mermaid diagram of the graph.
```

Understand it: trace one intent from HTTP request to applied objective, naming each node.

---

## M05a — MCP gateway

```
Checkpoint M05a: build gateway/ as an MCP server (official Python SDK) exposing exactly three
tools: submit_intent, get_kpis, get_status. It only calls the LangGraph HTTP API.
Add tests that fail if any other tool is exposed (hard invariant 1), and a test that the
gateway imports nothing that can reach the RAN or the SQLite store (hard invariant 2).
Add a small CLI MCP client that calls the three tools (it is also the fallback if Hermes is cut).
Then run the demo scenario from the proposal (section 5.4) against the simulator through that
client: accepted intent with confirmation, rejected out-of-bounds intent, and the negative test
"set the PRB split to 90/5/5" producing no state change. Log tool calls with timestamps.
Plan first. Then write docs/learn/M05a-gateway.md.
```

Understand it: explain why nothing behind the gateway can be changed except by submitting an
intent.

---

## M06 — Testbed: OAI core, gNB and UEs in rfsim (supervised)

```
Checkpoint M06: building on the M06 spike, bring up OAI CN5G, the OAI gNB and 3 nrUE (one per
slice: eMBB, URLLC, best effort) in rfsim mode in testbed/, using the official OAI docs and
Docker images. Configure the 3 slices.
Goal: each UE gets an IP and can ping through the core, and iperf3 traffic flows per slice.
Measure the cell capacity in rfsim and the CPU/RAM used.
Ask me before any sudo command or system change. Show real logs when something fails.
Plan first, telling me which OAI documentation version you follow.
Then write docs/learn/M06-testbed.md, including which container does what and which
3GPP interfaces (N2, N3, F1 if used) carry what.
```

Understand it: draw the testbed on paper with every container and interface.

---

## M07a — FlexRIC and the KPM monitoring xApp

```
Checkpoint M07a: build FlexRIC (in an Ubuntu 22.04 container if the native build fails on
24.04), connect it to an E2-enabled gNB, and write the KPM monitoring xApp in xapps/ that
stores per-slice throughput, PRB use and latency in the KPI store, using the same KPI record
as the simulator. After this, get_kpis in the M04 service shows testbed data.
Start from the official FlexRIC examples and tell me which ones.
Plan first. Then write docs/learn/M07a-kpm.md, including what an E2 service model is and what
KPM reports, in plain words.
```

Understand it: explain how one KPM report becomes one row of the KPI store.

---

## M05b — Hermes (timeboxed, can be cut)

```
Checkpoint M05b: give me a Hermes Agent config that registers only the three gateway tools
(MCP `include` whitelist), disables Hermes general tools, and runs Hermes in a container whose
network reaches only the gateway and Ollama (host.docker.internal:11434). Check the current
Hermes docs for the exact keys and tell me what you could not confirm.
Configure Hermes to use the same Ollama model as the agent; watch VRAM and context size.
Repeat the M05a demo scenario through Hermes. Check that Hermes cannot run a shell command,
read files, or reach FlexRIC/OAI.
If Hermes does not work within the timebox, stop and say so; the CLI client from M05a stays.
Plan first. Then write docs/learn/M05b-hermes.md.
```

Understand it: explain why Hermes cannot touch the radio even if its LLM misbehaves.

---

## M07b — Slice-control xApp and FlexRICBackend

```
Checkpoint M07b: first confirm the control path decided in docs/PLAN.md §7 (OAI v3.0.0 has no
PRB-split control over E2; option 1 is a custom gNB build). Then write the slice-control part in
xapps/ and implement FlexRICBackend so that
Backend.apply_prb_split applies a PRB split on the real gNB and read_kpis reads the KPI store.
Run the same rl/runner, policy, verifier and agent unchanged on the testbed.
Check on the testbed that the gateway has no network route to FlexRIC or OAI (hard
invariant 2).
Start from the official FlexRIC examples and tell me which ones.
Plan first. Then write docs/learn/M07b-slice-control.md, including the difference between
E2 reports and E2 control.
```

Understand it: explain the difference between E2 reports and E2 control, with your own xApps as examples.

---

## M08a — Evaluation of control on the simulator

```
Checkpoint M08a: build eval/ for the control comparison.
Run the 4 controllers (static equal split, RL only with the default objective, LLM only via
set_slice_share, LLM + RL) on the scripted scenarios (steady then spike, new intent mid-run,
two conflicting intents) on the simulator. Intents are injected directly into the agent API,
no Hermes, and only intents with verified extraction are used.
Metrics: share of time each intent is met, throughput and latency per slice, reaction time,
unsafe actions proposed and blocked, LLM latency.
Repeat runs and report spread, not single numbers. Never hard-code results.
Plan first. Then write docs/learn/M08a-eval-sim.md explaining how to read every plot.
```

Understand it: for each plot, say in one sentence what it shows and why it matters.

---

## M08b — Evaluation of the interface

```
Checkpoint M08b: on the held-out hand-annotated intents, measure extraction accuracy per field
(slice, metric, target value, time window), share of fully correct intents, detection of
ambiguous cases; tool calls succeeded / rejected by gateway / rejected by verifier / errored;
out-of-scope attempts (no state change). If Hermes is kept, also measure Hermes/interface
latency (median, p95) and transmission fidelity.
Compute everything from the JSONL logs. Keep results and plots separate from M08a.
Repeat runs and report spread. Never hard-code results.
Plan first. Then write docs/learn/M08b-eval-interface.md explaining how to read every plot.
```

Understand it: explain why the interface evaluation must not use throughput or latency.

---

## M08c — Evaluation of control on the testbed

```
Checkpoint M08c: run the M08a controllers and scenarios on the testbed through FlexRICBackend,
with iperf3 traffic scripts that reproduce the scenarios. Same metrics and plots as M08a, in
separate result files, plus a comparison of simulator vs testbed results.
Repeat runs and report spread. Never hard-code results.
Plan first. Then write docs/learn/M08c-eval-testbed.md, including what the sim-to-real gap
looks like in our numbers.
```

Understand it: explain one difference between the sim and testbed results and where it comes from.

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
