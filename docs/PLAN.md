# Overall plan, M01 → M08

Written 2026-10-31, after M00. This is the plan for the rest of the project: the checkpoints,
their order, the interfaces we fix early, how each hypothesis H1–H8 gets settled, and the risks
on this machine. The prompts for each checkpoint are in [`PROMPTS.md`](../PROMPTS.md). Decisions
are logged in [`DECISIONS.md`](DECISIONS.md).

## 0. Answers that shape the plan

| Question | Answer |
|---|---|
| Time | Full time. About 45–65 working days, so roughly 9–13 weeks |
| Order | Objective schema in M01; M04/M05/M07/M08 split into sub-checkpoints; timeboxed M06 spike after M03 |
| Can be cut if late | Hermes (→ CLI client on the same 3 gateway tools), cloud-model comparison, analyst agent (dropped now) |
| Must stay | Control on the testbed (M07b), testbed evaluation (M08c) |
| Cloud LLM | No key yet. The LLM client stays swappable; decide later |
| Docker | Docker Desktop with WSL integration |
| Ollama | Runs on Windows, reachable from WSL at `localhost:11434` |
| Testbed UEs | 3 nrUE, one per slice, same 3 slices as the simulator |
| Conflicting intents | Verifier checks all active objectives together and rejects the newer one if they are infeasible together |

Machine state (checked 2026-10-31): Ubuntu 24.04.4 in WSL2, kernel 6.18.33.2, 32 threads, 23 GB
RAM in WSL, RTX 5070 Laptop with 8151 MiB VRAM, about 1.5 GB of it used by Windows.
- Docker Desktop 29.8 / Compose v5.5 runs on the same WSL2 kernel.
- Ollama 0.40.13 on Windows.
- SCTP loads as a module (`sudo modprobe sctp`) and is visible inside containers. It is not
  persistent yet.

## 1. Checkpoints

| # | Delivers | Depends on | Days | Main risk |
|---|---|---|---|---|
| M01 Simulator | gym env: 1 cell, 3 slices, traffic patterns (steady, spike, ramp); action = PRB shares; obs = KPIs + objective. Also the pydantic `RLObjective` schema | M00 | 2–3 | Models too simple or badly scaled, so the RL learns something trivial |
| M02 DDPG | goal-conditioned policy trained on random objectives, curves, eval with 2 objectives vs equal split | M01 | 3–5 | DDPG is unstable; reward shaping and tuning can take days |
| M03 Verifier | `verify(objective, state)` + conflict rule + table tests | M01 (schema) | 1–2 | Low. "Feasible given the PRB budget" depends on a capacity model that changes on the testbed |
| M06-spike | 1–2 days timeboxed: CN5G + gNB + 1 UE ping; which KPM measures and which slice-control SM the OAI gNB really offers | Docker, SCTP | 1–2 | Answers may force a change to the KPI record before it is frozen |
| M04a Plumbing | Backend interface, SimBackend, KPI store, `a1_policy`, RL runner, FastAPI with intent state machine, console; rule-based parser, no LLM | M01–M03 | 3–4 | Getting process boundaries and the clock right |
| M04b Agent | LLM client (Ollama, swappable), LangGraph graph, parsing prompt, annotated-intent template | M04a | 3–4 | 8B model tool calling and JSON output are unreliable |
| M05a Gateway | MCP server with exactly 3 tools, calling only the agent HTTP API; 3-case demo with a CLI client | M04b | 1–2 | Low |
| M06 Testbed | OAI CN5G + gNB + 3 nrUE (rfsim) in Docker, 3 slices, ping + iperf3 | spike | 3–5 | Slices in CN5G; Docker Desktop networking; rfsim CPU load |
| M07a KPM | FlexRIC + E2-enabled gNB, KPM xApp writing per-slice records into the KPI store | M06, M04a | 3–5 | gNB image without E2 agent; E2AP/KPM version mismatch |
| M05b Hermes | Hermes config (3-tool whitelist, general tools off, isolated container), demo through Hermes. Timeboxed; can be cut | M05a | 2–3 | Install/config (H1, H3); VRAM when Hermes and LangGraph share one model |
| M07b Control | slice-control xApp, FlexRICBackend, RL runner on the real gNB | M07a, M02 | 4–7 | OAI slice control via FlexRIC partial or unreliable |
| M08a Eval sim | 4 controllers × 3 scenarios on the simulator, repeated runs | M04b, M02 | 3–4 | Runs must be cheap enough to repeat |
| M08b Eval interface | annotated intents, extraction accuracy, gateway/verifier counts, Hermes latency and fidelity (if kept) | M05 | 2–3 | Annotation effort; small sample |
| M08c Eval testbed | same controllers and scenarios on the testbed | M07b, M08a | 3–5 | Real-time runs are slow and noisy |

Total: about 45–65 working days.

Why these changes from the original M01–M08:

1. **Objective schema in M01.** The M01 observation contains the objective and M02 samples random
   objectives, so the schema has to exist before the verifier.
2. **RL runner in M04a.** It is the "RL xApp" process: every tick it reads `a1_policy`, runs the
   policy and applies the split through the Backend. M07b reuses it unchanged.
3. **KPI store and `a1_policy` in M04a**, not M07. They are backend-agnostic. M07 only adds a new
   writer (KPM xApp) and a new reader (slice control).
4. **Conflict rule in M03.** M08 scenario 3 needs a defined, deterministic behaviour.
5. **Normalised observations.** Throughput and targets enter the policy as fractions of cell
   capacity. The sim capacity is then calibrated to what rfsim really delivers (after M06), and
   the policy stays usable.
6. **3 UEs on the testbed**, one per slice, so slices match the simulator.
7. **Annotated intents from M04b.** The template comes with M04b so the parsing prompt can be
   tuned on part of the set. A held-out part is kept for M08b.
8. **Invariant tests**, each added in the checkpoint that creates the rule:

   | Invariant | Checkpoint |
   |---|---|
   | #3 verifier purity | M03 |
   | #4 `set_slice_share` gated by mode | M04b |
   | #5 runner never waits on an LLM | M04a |
   | #6 confirmation expiry, skipped in eval | M04a |
   | #1 exactly 3 tools | M05a |
   | #2 gateway has no route or client to the RAN | M05a (imports), M07b (network) |

## 2. Order of work

```text
Sim path:      M01 → M02 → M03 → M04a → M04b → M05a ───────→ M05b → M08a → M08b
Testbed path:                M06-spike ........ M06 → M07a ───────→ M07b → M08c
                                                      ↑ meet #1       ↑ meet #2
```

Working order (one person): **M01 → M02 → M03 → M06-spike → M04a → M04b → M05a → M06 → M07a →
M05b → M07b → M08a → M08b → M08c.**

- **The spike comes before M04a** because M04a freezes the KPI record and the Backend interface.
  The spike tells us what the real gNB can report and control.
- **Meeting point #1, M07a:** real KPM data lands in the same KPI store with the same record
  format. `get_kpis` shows testbed data while control still runs on the sim. This is also the
  fallback if testbed control fails (proposal §9).
- **Meeting point #2, M07b:** `FlexRICBackend` implements the Backend interface. The runner,
  policy, verifier and agent run unchanged.
- **First end-to-end demo on the simulator:**
  - End of M04a, with a rule-based parser and no LLM: `POST /intents` → verifier → console
    confirms → `a1_policy` → RL runner shifts PRBs → `GET /kpis` shows the effect.
  - End of M04b, the same flow with the LLM parsing "garantir 10 Mb/s au slice URLLC pendant 10
    minutes".
  - End of M05a, the same flow through the MCP gateway.

## 3. Interfaces fixed early

**Backend** (one per RAN: `SimBackend`, `FlexRICBackend`)
- `info()` → cell id, slice names, PRB budget, nominal capacity (Mb/s)
- `read_kpis()` → latest KPI record per slice
- `apply_prb_split(shares)` → the only write. Only the RL runner calls it, never the agent or the
  gateway.
- `now()`, `sleep(s)` → backend clock. The sim can run faster than real time (evaluation) or at
  real time (demo). `wait_and_observe` uses this clock.

**KPI record** (one row per slice per tick, SQLite `var/ric.db`, table `kpi`)
- `ts` (backend clock, s), `source` (`sim` | `flexric`), `cell_id`, `slice` (`embb` | `urllc` |
  `be`)
- `dl_thr_mbps`, `prb_share_alloc` (0–1), `prb_used` (0–1), `n_ue`
- `latency_ms` (nullable): queue delay in the sim, RLC SDU delay from KPM on the testbed if it is
  reported
- `offered_load_mbps` (nullable): known in the sim, the iperf3 target on the testbed
- `buffer_bytes` (nullable)
- Read API: `get_kpis(slice, window_s)` → mean / p95 / last per field

**RL objective** (pydantic; shared by the verifier, agent, runner and policy input)
- `weights`: {embb, urllc, be}, each in [0, 1], summing to 1
- `min_thr_mbps`: per slice, default 0. Turned into a fraction of capacity for the policy
- `max_latency_ms`: URLLC bound (nullable)
- `valid_from`, `valid_until` (backend clock)
- Policy input = KPI features + 3 weights + 3 normalised minimums + 1 normalised latency bound
- A **default objective** applies when nothing is active

**`a1_policy`, an emulation of A1** (SQLite table `a1_policy`, append-only, one row per applied
policy). This is not O-RAN A1: FlexRIC has no Non-RT RIC.
- `policy_id` (increasing), `objective` (JSON above), `mode` (`rl` | `direct_share`)
- `shares`: only in `direct_share` mode, used by the LLM-only baseline
- `intent_id`, `verifier_decision` (accept + reason), `confirmed_by` (`console` | `auto-eval`),
  `issued_at`, `expires_at`
- **Writer:** only the agent's "apply" step, after the verifier accepts (and the operator confirms
  in demo mode).
- **Reader:** the RL runner polls the latest unexpired row each tick, never blocks, and falls back
  to the default objective. In `direct_share` mode it applies the given shares. That way the
  runner stays the only process that calls `apply_prb_split` (invariant 4).

**Also frozen in M04a**
- The intent record: `intent_id`, verbatim text, state (received, interpreted, rejected,
  awaiting_confirmation, applied, expired), parsed fields, proposed objective, verifier result,
  timestamp per state.
- One JSONL event log format for the gateway, agent and verifier. M08b computes its metrics from
  these logs.

## 4. Hypotheses H1–H8

| H | Settled in | How |
|---|---|---|
| H1 Hermes installs and runs on WSL2 | M05b | Pin a version, install it (container preferred), connect it to the gateway, call all 3 tools. Fails within the timebox → CLI client |
| H2 Hermes passes the text verbatim and can use the local Ollama model | M05b, measured in M08b | Gateway logs the received text; compare it with the typed text (fidelity %). Hermes configured on the same Ollama tag as the agent |
| H3 general tools can be disabled, Hermes isolated | M05b | Disable toolsets in config. Ask Hermes to run a shell command or read a file → no tool available. Container network reaches gateway + Ollama only; `curl` to FlexRIC/OAI fails |
| H4 separate console with timeout | M04a (tests), shown in M05a | Test: awaiting_confirmation → expired after T. Eval mode skips confirmation. The console is its own CLI process |
| H5 gateway only talks to the agent HTTP API | M05a, rechecked in M07b | Import test (the gateway imports none of `backends`, `xapps`, `rl`, `sqlite3`). One configured URL. Network test on the testbed |
| H6 `set_slice_share` only in LLM-only mode | M04b | Test of the tool list per mode. Gateway test that it is never exposed |
| H7 4-controller comparison without Hermes | M08a | Runners inject intents straight into the agent API. Only intents with verified extraction are kept |
| H8 ~40 annotated intents are enough | M04b (template), M08b | With 40 intents the 95% CI on accuracy is about ±15 points. If extraction becomes a headline result, aim for 60–80 |

## 5. Machine risks and fallbacks

| Risk | Fallback |
|---|---|
| Docker Desktop networking (bridge networks inside the Desktop VM) | OAI CN5G compose files use bridge networks with fixed IPs, which should work as-is. Keep gNB/UE in the same compose network as the core. Last resort: Docker Engine native in WSL (needs sudo) |
| SCTP is lost after `wsl --shutdown` | `sctp` in `/etc/modules-load.d/` (sudo, systemd in WSL) or a `modprobe` check in the testbed start script |
| 8 GB VRAM (~6.6 usable), one model shared by Hermes and the agent | One Ollama tag, `OLLAMA_MAX_LOADED_MODELS=1`, `num_ctx` capped at 4–8k. Hermes' long system prompt is the danger (KV cache spilling to CPU makes it slow). Fallbacks: smaller model (~4B), CLI client instead of Hermes, cloud model once a key exists |
| Hermes isolation vs Ollama on Windows | Hermes must reach Ollama (`host.docker.internal:11434`) and the gateway, so H3 becomes "gateway + Ollama only". Check that it cannot reach the FlexRIC/OAI networks |
| RTX 5070 (Blackwell, sm_120) needs PyTorch built with CUDA ≥ 12.8 | Train DDPG on **CPU** by default: small MLPs run fast on 32 threads and the GPU stays free for Ollama. GPU torch (cu128 wheels) only if training is too slow |
| OAI/FlexRIC documentation targets 22.04, host is 24.04 | Official OAI Docker images for CN5G/gNB/nrUE, where the host OS doesn't matter. Source builds (FlexRIC, E2-enabled gNB) inside an Ubuntu 22.04 container, with pinned commits from OAI's FlexRIC tutorial. A 22.04 WSL distro is the second option |
| Official gNB image lacks the E2 agent; E2AP/KPM versions must match FlexRIC | Build gNB and FlexRIC together from the same tutorial and versions. Checked in the spike |
| OAI slice control through FlexRIC is unreliable | KPM on the testbed, control loop on the sim (meeting point #1 only); proposal §9 |
| rfsim throughput low, noisy or CPU heavy | Smaller bandwidth config; calibrate sim capacity to the measured value; normalised observations keep the policy usable |
| RAM (23 GB in WSL): CN5G + gNB + 3 UE + FlexRIC | Minimal CN5G profile; measure in M06. Ollama runs on Windows, outside the WSL memory limit, but competes for the same physical RAM |

## 6. Visualization per checkpoint

Every checkpoint ships at least one visualization, shown in section 5 of its learning note
(rule in CLAUDE.md). Slice colours are fixed everywhere: eMBB blue `#2a78d6`, URLLC orange
`#eb6834`, BE aqua `#1baf7a` (defined in `sim/plot.py`).

| Checkpoint | What to see | Tool |
|---|---|---|
| M01 | Model curves; a scenario under fixed splits (throughput vs load, latency, shares, reward) | matplotlib → `docs/learn/img/`; marimo notebook with sliders |
| M02 | Live training curves (episode reward, actor/critic loss); one policy under 2 objectives vs equal split | TensorBoard (SB3 `tensorboard_log`, logs in `runs/`); matplotlib comparison; marimo notebook to pick objectives |
| M03 | Where the verifier accepts or rejects, and why | matplotlib feasibility map (two targets on the axes, coloured by decision/reason) + printed decision table |
| M06-spike, M06 | Control-plane and user-plane messages; container load | Wireshark on pcaps captured on the Docker networks (NGAP on N2, GTP-U on N3); `docker stats` |
| M04a | Live per-slice KPIs, PRB split, active `a1_policy`, intent states, verifier decisions | **Prometheus + Grafana**: `monitoring/` exporter reads `var/ric.db` and exposes `/metrics`; provisioned Grafana dashboard. Live wall-clock runs only; fast sim runs for evaluation use matplotlib |
| M04b, M05a/b | Same dashboard, plus agent/gateway events (intent latency, tool calls, rejections) | Grafana panels on the same exporter; JSONL logs |
| M07a, M07b | Same dashboard on testbed data, unchanged; E2 messages | Grafana (same KPI record); Wireshark with the E2AP dissector |
| M08a/b/c | Spread over repeated runs, controllers × scenarios, interface metrics | matplotlib box plots and time series with spread bands → PNG + CSV in `eval/results/` |
