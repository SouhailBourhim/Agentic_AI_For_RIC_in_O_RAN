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
| M02b Goal conditioning | oracle-validated goal test (`rl/goal_test.py`), reward v2, objective relabeling, filter in the training loop; dev round (ablation) on validation, confirmation on fresh seeds, held-out test run once | M02 | 2–3 | Fails again: then FAIL on record, `td3_seed2` stays deployed |
| M02c MLOps | local MLflow: tracking of every training/eval run (backfill of earlier runs), registry `slice-policy` with alias `champion` and a deterministic promotion gate, CI (GitHub Actions) | M02b | 1–2 | Heavy dependency; SQLite contention with parallel runs |
| M03 Verifier | `verify(objective, state)` + conflict rule + table tests | M01 (schema) | 1–2 | Low. "Feasible given the PRB budget" depends on a capacity model that changes on the testbed |
| M06-spike | 1–2 days timeboxed, right after M01: CN5G + gNB + UEs ping, rfsim capacity; which KPM measures and which slice-control SM the OAI gNB really offers | Docker, SCTP | 1–2 | Answers may force a change to the KPI record before it is frozen |
| M04a Plumbing | Backend interface, SimBackend, KPI store, `a1_policy`, RL runner, FastAPI with intent state machine, console; rule-based parser, no LLM | M01–M03 | 3–4 | Getting process boundaries and the clock right |
| M04b Agent | LLM client (Ollama, swappable), LangGraph graph, parsing prompt, annotated-intent template | M04a | 3–4 | 8B model tool calling and JSON output are unreliable |
| M05a Gateway | MCP server with exactly 3 tools, calling only the agent HTTP API; 3-case demo with a CLI client | M04b | 1–2 | Low |
| M06 Testbed | 3 S-NSSAIs end to end (core config + database, gNB `snssaiList`, UEs), one UE per slice, traffic scripts, ordered start/stop script | spike | 3–5 | KPM style-4 reports filter on one S-NSSAI, so probably one subscription per slice. **Done 2026-11-02** (`docs/learn/M06-testbed.md`): confirmed, the stock xApp gets only cell-level reports with SD-tagged slices, so M07a subscribes per slice |
| M07a KPM | KPM xApp (one long-lived process) writing per-slice records into the KPI store | M06, M04a | 3–5 | E2 state is fragile across xApp/RIC restarts (seen in the spike). **Done 2026-11-03** (`docs/learn/M07a-kpm.md`): restarts are clean (the RIC deletes the old subscriptions); found and patched an OAI KPM bug with several subscriptions, so our gNB image (`agentic-ric/oai-gnb:v3.0.0-kpmfix`) now exists for M07b |
| M05b Hermes | Hermes config (3-tool whitelist, general tools off, isolated container), demo through Hermes. Timeboxed; can be cut | M05a | 2–3 | Install/config (H1, H3); VRAM when Hermes and LangGraph share one model |
| M07b Control | PRB-split control on the real gNB, FlexRICBackend, RL runner on the testbed | M07a, M02 | 5–10 | **No PRB-split control over E2 in OAI v3.0.0** (SLICE is a stub): needs a custom gNB build: option 1 decided, see §7. **Done 2026-11-05** (`docs/learn/M07b-slice-control.md`): option 1 works over E2; cap linear (0.654 Mb/s per PRB); the sim-trained policy oscillates on the testbed (one tick of extra measurement delay), not retrained |
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
Sim path:      M01 ──────────→ M02 → M03 → M04a → M04b → M05a ───────→ M05b → M08a → M08b
Testbed path:     M06-spike ─(capacity)─↗ ........... M06 → M07a ───────→ M07b → M08c
                                                      ↑ meet #1       ↑ meet #2
```

Working order (one person): **M01 → M06-spike → M02 → M02b → M02c → M03 → M04a → M04b → M05a → M06 → M07a →
M05b → M07b → M08a → M08b → M08c.**

- **The spike comes right after M01** (moved there on 2026-11-01). It must come before M04a,
  which freezes the KPI record and the Backend interface, because it tells us what the real gNB
  can report and control. Running it before M02 also gives a measured rfsim capacity, so the
  policy is trained on a calibrated simulator, and it tests the riskiest part of the project first.
- **Meeting point #1, M07a:** real KPM data lands in the same KPI store with the same record
  format. `get_kpis` shows testbed data while control still runs on the sim. This is also the
  fallback if testbed control fails (proposal §9).
- **Meeting point #2, M07b:** `FlexRICBackend` implements the Backend interface. The runner,
  policy, verifier and agent run unchanged.
- **First end-to-end demo on the simulator:**
  - End of M04a, with a rule-based parser and no LLM: `POST /intents` → verifier → console
    confirms → `a1_policy` → RL runner shifts PRBs → `GET /kpis` shows the effect.
  - End of M04b, the same flow with the LLM (qwen3:8b) parsing and planning, in a sixth
    process: `POST /intents` stores the intent (`received`), the agent worker
    (`python -m agent.worker`) claims it (`processing`), runs parse → plan → verifier →
    console → observe → reflect, and may revise it once (a new intent with `supersedes`).
    Commands in `docs/learn/M04b-langgraph.md` §5.
  - End of M05a, the same flow through the MCP gateway (done 2026-11-02: accepted, rejected and
    out-of-scope cases live, `docs/learn/M05a-gateway.md` §5).

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
- `latency_ms` (nullable): queue delay in the sim, RLC SDU delay from KPM on the testbed
- `offered_load_mbps` (nullable): known in the sim, the iperf3 target on the testbed
- `buffer_bytes` (nullable)
- `policy_id` (nullable, added in M04a): the `a1_policy` row in force during the tick (NULL =
  default objective). Every switch, including the ones caused by an expiry, is visible in the data
- Read API: `get_kpis(slice, window_s)` → mean / p95 / last per field
- Table `backend_info` (one row, written by the runner at startup): cell id, source, slices,
  n_prb, capacity, URLLC base latency, served model version. With the latest `kpi.ts` (the backend
  clock) it is how the other processes learn about the cell without touching the Backend
- **Testbed mapping, confirmed in the M06 spike.** OAI KPM reports are per UE, about every
  1 s per subscription; each UE is mapped to its slice, then:
  - `dl_thr_mbps` = Σ `DRB.UEThpDl` (kbps) / 1000
  - `prb_used` = (Σ `RRU.PrbTotDl` (% of cell) / 100) / `prb_share_alloc`
  - `latency_ms` = max `DRB.RlcSduDelayDl` (µs) / 1000. It rises clearly under overload.
    It is the gNB's RLC queueing delay (~0.5 ms below capacity, ping ~22 ms), not end-to-end
    latency (M07a).
  - **M07a, for M07b:** a full cell reads ~73–75 % in `RRU.PrbTotDl`, so testbed `prb_used`
    saturates near 0.74 where the simulator reaches 1.0; decide whether to rescale before the
    policy sees it. Until M07b `prb_share_alloc` = 1.0 and `offered_load_mbps` is NULL.
  - **Decided in M07b:** the 0.74 is the TDD downlink fraction, (7 + 6/14) / 10 = 0.743, because
    the counter averages over all slots; `prb_used` = (Σ `RRU.PrbTotDl` / 100) / 0.743 /
    `prb_share_alloc`, with `prb_share_alloc` the split in force during the report.
    `offered_load_mbps` = the iperf3 rate the traffic script writes (an oracle; KPM has no
    offered load). Per-slice KPM values go through `kpm_latest`; the runner writes `kpi`.
  - The first report after each subscription is cumulative and is discarded. RSRP/SINR
    reports are invalid and unused.

**RL objective** (pydantic; shared by the verifier, agent, runner and policy input)
- `weights`: {embb, urllc, be}, each in [0, 1], summing to 1
- `min_thr_mbps`: per slice, default 0. Turned into a fraction of capacity for the policy
- `max_latency_ms`: URLLC bound (nullable)
- `valid_from`, `valid_until` (backend clock)
- Policy input = KPI features + 3 weights + 3 normalised minimums + 1 normalised latency bound
- A **default objective** applies when nothing is active

**`a1_policy`, an emulation of A1** (SQLite tables `a1_schedule` and `a1_policy`, append-only).
This is not O-RAN A1: FlexRIC has no Non-RT RIC.
- **A timetable (M04a).** Each apply or cancel writes a new schedule (`a1_schedule`: id, time,
  reason) with one `a1_policy` row per merge segment from now on (`valid_from`, `expires_at`).
  The newest schedule is in force; an expiry is the clock moving into the next segment, so no
  process has to wake up and fast runs see no gap. An empty schedule = default objective.
- `policy_id` (increasing), `schedule_id`, `objective` (JSON above), `mode` (`rl` | `direct_share`)
- `shares`: only in `direct_share` mode, used by the LLM-only baseline
- `intent_ids` (list): every intent merged into this objective (M03, see below);
  `verifier_decision` (accept + reason), `confirmed_by` (`console` | `auto-eval`), `issued_at`,
  `valid_from`, `expires_at`
- **The objective is the merge of the active intents** (`verifier.merge.merge`, decided in M03):
  per slice the highest minimum throughput, the tightest URLLC latency bound, the weights of the
  newest still-active intent (default weights if none). The verifier checks
  `merge(active + new)`, so what is verified is exactly what runs. The merge is recomputed and
  re-verified (`verifier.verify.reverify_after_removal`) whenever the active set changes; with
  the timetable, expiries are verified ahead, when the schedule is written. The runner logs
  every `policy_switch`, expiries included. "Newest" = most recently *applied* (confirmation
  time), so the verifier and the timetable agree on whose weights run. `get_status` of an older
  intent shows it is still applied as part of the merged objective (`in_effect`, `merged_with`).
- **Writer:** only `agent/apply.py`, after the verifier accepts. The agent service calls it in
  eval mode; the operator console calls it in demo mode, re-verifying at confirmation time.
  Reading the applied set and writing the schedule happen in one transaction.
- **Reader:** the RL runner reads the segment of the newest schedule covering "now" each tick,
  never blocks (zero busy-timeout; on error it keeps the last segment until it expires), and
  falls back to the default objective. In `direct_share` mode it applies the given shares. That way the
  runner stays the only process that calls `apply_prb_split` (invariant 4).

**Also frozen in M04a**
- The intent record: `intent_id`, verbatim text, state (received, interpreted, rejected,
  awaiting_confirmation, applied, expired, cancelled), parsed fields, proposed objective,
  verifier result, reason, `confirmed_by`, `issued_backend`, `applied_backend`,
  `confirm_deadline` (wall clock), timestamp per state.
- One JSONL event log format for the gateway, agent and verifier. M08b computes its metrics from
  these logs.

## 4. Hypotheses H1–H8

| H | Settled in | How |
|---|---|---|
| H1 Hermes installs and runs on WSL2 | M05b | Pin a version, install it (container preferred), connect it to the gateway, call all 3 tools. Fails within the timebox → CLI client. **Confirmed 2026-11-04:** v0.22.7 image, `docs/learn/M05b-hermes.md` |
| H2 Hermes passes the text verbatim and can use the local Ollama model | M05b, measured in M08b | Gateway logs the received text; compare it with the typed text (fidelity %). Hermes configured on the same Ollama tag as the agent. **Preview 2026-11-04, tune split:** 17/20 exact, none altered, 3 not sent (empty model answers). **Measured 2026-11-07, held-out:** 18/20 exact in each of 3 repeats (interval 70-97 %), none altered, 2 not sent |
| H3 general tools can be disabled, Hermes isolated | M05b | Disable toolsets in config. Ask Hermes to run a shell command or read a file → no tool available. Container network reaches gateway + Ollama only; `curl` to FlexRIC/OAI fails. **Confirmed 2026-11-04:** 3 tools on the wire, 17/17 network targets as expected with the testbed up; the same probe from the testbed network reaches them |
| H4 separate console with timeout | M04a (tests), shown in M05a | Test: awaiting_confirmation → expired after T. Eval mode skips confirmation. The console is its own CLI process |
| H5 gateway only talks to the agent HTTP API | M05a, rechecked in M07b | Import test (the gateway imports none of `backends`, `xapps`, `rl`, `sqlite3`). One configured URL. Network test on the testbed. **Confirmed 2026-11-05 (M07b):** with the testbed up, all 7 RAN addresses time out from WSL (where the gateway runs) and answer from the testbed network (`testbed/check_gateway_route.py`) |
| H6 `set_slice_share` only in LLM-only mode | M04b | Test of the tool list per mode. Gateway test that it is never exposed |
| H7 4-controller comparison without Hermes | M08a | Runners inject intents straight into the agent API. Only intents with verified extraction are kept. **Confirmed 2026-11-07:** no Hermes in any run, every parse matched its expected facts (`docs/learn/M08a-eval-sim.md`) |
| H8 ~40 annotated intents are enough | M04b (template), M08b | With 40 intents the 95% CI on accuracy is about ±15 points. If extraction becomes a headline result, aim for 60–80. **Answered 2026-11-07 (M08b):** 19/20 on the held-out half, interval 76-99 %: enough to tune and catch large failures, not to quote a precise accuracy |

## 5. Machine risks and fallbacks

| Risk | Fallback |
|---|---|
| Docker Desktop networking (bridge networks inside the Desktop VM) | OAI CN5G compose files use bridge networks with fixed IPs, which should work as-is. Keep gNB/UE in the same compose network as the core. Last resort: Docker Engine native in WSL (needs sudo) |
| SCTP is lost after `wsl --shutdown` | `sctp` in `/etc/modules-load.d/` (sudo, systemd in WSL) or a `modprobe` check in the testbed start script |
| 8 GB VRAM (~6.6 usable), one model shared by Hermes and the agent | One Ollama tag, `OLLAMA_MAX_LOADED_MODELS=1`, `num_ctx` capped at 4–8k. Hermes' long system prompt is the danger (KV cache spilling to CPU makes it slow). Fallbacks: smaller model (~4B), CLI client instead of Hermes, cloud model once a key exists |
| Hermes isolation vs Ollama on Windows | Hermes must reach Ollama (`host.docker.internal:11434`) and the gateway, so H3 becomes "gateway + Ollama only". Check that it cannot reach the FlexRIC/OAI networks |
| RTX 5070 (Blackwell, sm_120) needs PyTorch built with CUDA ≥ 12.8 | Train DDPG on **CPU** by default: small MLPs run fast on 32 threads and the GPU stays free for Ollama. GPU torch (cu128 wheels) only if training is too slow |
| ~~OAI/FlexRIC documentation targets 22.04, host is 24.04~~ | **Resolved in the spike**: official v3.0.0 images (gNB/nrUE on 24.04, FlexRIC on 22.04 inside its image) run unchanged on this host |
| ~~Official gNB image lacks the E2 agent~~ | **Resolved in the spike**: built with E2 (E2AP v3, KPM v3.00), same versions as `oai-flexric:v3.0.0` |
| OAI slice control through FlexRIC | **Confirmed missing in v3.0.0.** Decided (§7): custom gNB build with a slice-aware `dl_rb_alloc` policy and a real SLICE control handler; fallback to a side channel (option 2) if the E2 handler exceeds the M07b timebox |
| rfsim throughput low, noisy or CPU heavy | **Measured**: ~75 Mb/s DL, limited by the radio config (MCS 28 64-QAM, TDD 7/2), not CPU (~10 % of 32 threads, ~4 GiB). Sim calibrated to 75 Mb/s |
| ~~RAM (23 GB in WSL): CN5G + gNB + 3 UE + FlexRIC~~ | **Measured in M06**: ~4.2 GiB and 4.3 (idle) to 7.4 (overload) cores of 32 for the whole testbed. Ollama runs on Windows, outside the WSL memory limit, but competes for the same physical RAM |

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

## 7. Decision: PRB-split control on the real gNB (decided 2026-11-01: option 1)

Found in the M06 spike (details in `docs/learn/M06-spike.md` §6). OAI v3.0.0 has no working
E2 control of the PRB split: the SLICE service model is a stub, and RC only supports bearer
control and handover. The downlink scheduler does call a pluggable `mac->dl_rb_alloc` policy,
and every candidate carries its `nssai`.

| Option | What | Keeps the "RL xApp controls the RAN over E2" claim? | Cost |
|---|---|---|---|
| 1 (recommended, timeboxed) | Custom gNB build: slice-aware `dl_rb_alloc` policy + real SLICE control handler | Yes | C code in OAI, own gNB image |
| 2 | Same policy, shares sent through a side channel (file or telnet) | No (not E2) | Less C, no E2 encoding |
| 3 | Real KPM on the testbed; control loop on the simulator only (proposal §9) | Partly | None |

**Decided on 2026-11-01: option 1.** We build our own gNB image with a slice-aware
`dl_rb_alloc` policy and a real SLICE control handler, so the RL xApp sets the PRB split over
E2. The M07b plan fixes the timebox. If the E2 SLICE handler cannot be finished within it, the
fallback is option 2: the same scheduler policy driven through a side channel, reported
explicitly as not E2. M02–M05 are unaffected: they run on the simulator.

**Outcome (M07b, 2026-11-05): option 1 works**, well within the timebox (the handler worked at the
first build). The gNB accepts or refuses each SLICE control with a reason, carried in the RIC
Control Acknowledge; FlexRIC's RIC drops that answer before the xApp, so the xApp sees only that an
acknowledge arrived.

## 9. M09: proving the design beyond one scenario (decided 2026-11-07)

After M08d the comparison favours LLM + RL in one scenario only (S1). S2 and S3's first goal fit in an equal split, so they
can't show adaptation; S3's second goal tests the verifier; where LLM + RL loses, the policy is the cause. M09 makes the claim
general, with the testbed confirming rather than producing the main result:

| # | Delivers | Who runs the long part |
|---|---|---|
| M09a ✅ | Simulator profile calibrated to the testbed (flat RLC delay until saturation, real buffer, 79 Mb/s with its spread, KPIs 2 ticks late), verifier parameters to match; validated against M08c and M08d by a rule written first | Souhail (replay, ~1 h) |
| M09b ✅ | Seeded scenario generator in 5 families (F1 load, F2 new goal, F3 fast dynamics, F4 conflicts, F5 easy), the broad benchmark on dev seeds with v7; references: oracle ceiling, LLM + formula (no RL), a fairer LLM only | Souhail |
| M09c ✅ | Policy v8: calibrated profile, random 1-3 tick observation delay, penalty on share above need, objective changes and expiries mid-episode; fresh goal-test seeds | Souhail (training) |
| M09g | Diagnostics on dev, no training: the formula behind the RL's filter (alpha 0.10), whether the reward aims at the edge of a target, the actor-saturation mechanism. With the integrity groundwork (spent-set register, evidence manifest, ruff) | me |
| M09h ✅ | Policy v9, pre-registered: switching cost in the reward (M09g D1; served at alpha 0.10 like every policy), correlated exploration noise, undelayed applied split in the input, gamma 0.98 (reward margin and action regulariser out after M09g D2/D3), and a residual-on-formula arm; 5 seeds per arm, median-seed choice, fresh goal-test seeds | Souhail (training) |
| M09d ✅ | Pre-registered simulator result on test seeds, scored once, **after M09h**, with the champion of that time (v7): P1-P6 as paired differences with cluster-bootstrap intervals; written predictions for the testbed sample. Rules frozen in DECISIONS 2026-11-11 (`eval/m09d.py`) | Souhail |
| M09e | Testbed confirmation: 8 instances drawn from the test generator x 4 controllers (**static, LLM only v2, LLM + formula, LLM + RL**) x 2 repeats, one night | Souhail |
| M09f | Safety under dangerous in-scope intents (~30), LLM + RL and LLM only, unsafe changes applied must be 0 | me |

**M09a done 2026-11-08:** attempt 1 FAIL (6/8, 25/32), attempt 2 PASS (8/8, 27/32) after the agent's clock was fixed; LLM only can't be reproduced by any simulator (its choices turn on irrelevant input details), so its claims go to the testbed (M09e). Note: `docs/learn/M09a-sim-calibration.md`.

**M09b done 2026-11-09 (dev results, not a claim):** 200 instances x 7 controllers. Two references were fixed before analysis: the oracle (60 s horizon; still beaten on 25 % of F4 intents, so a reference there, a ceiling elsewhere) and LLM + formula (revision 4 of 5: peak sizing, slack drain). LLM + RL wins F1 (+0.54 on static, +0.15 on LLM only v2) and beats RL only everywhere. v7 loses F4a's two-target intent (0.27 against static's 1.00: over-serves URLLC), F2 by about 1 Mb/s, and F3 to the formula (0.60 against 0.92: no load history in its input). LLM + RL repeats within 0.006 across plays; LLM only swings by up to 0.99. Inputs to M09c: a share-above-need penalty, a load-history feature, the F2 shortfall. Open for M09d: how P6 labels feasibility where the oracle is only a reference. Note: `docs/learn/M09b-benchmark.md`.

**M09c done 2026-11-09: v8 not confirmed, v7 stays champion.** Round 1 (4 recipes x 3 seeds, dev benchmark): `v8_nooverserve` chosen by the rule (LLM + RL 0.850 against v7's 0.791; F3 0.60 -> 0.76). Ablation, one seed per recipe: the load history fixes F3, the calibrated cell matters (without it F4 drops to 0.55), the over-serving term did not help. Round 2 (seeds 40-44) on `m09_test`, scored once: seed 40's actor saturated and left no checkpoint; 41-44 pass T1b and T2 but miss T1a (0.861-0.896); the gate refused all. M09d therefore runs with v7. A later v8 attempt needs its own pre-registration (more seeds, median-seed choice, a saturation restart rule, T1a in the checkpoint rule) and fresh test seeds; `m09_test` is spent. Note: `docs/learn/M09c-policy-v8.md`.

**M09h done 2026-11-11: v9 not confirmed, v7 stays champion.** Dev round (6 arms x seeds 50-54, median seed per arm): 3 arms out by rule (no checkpoint), `v9_noswitch` chosen on the dev benchmark (LLM + RL 0.896 against v7's 0.791; ties formula_a10, 0.033 below LLM + formula). Confirm seeds 60-64, `m10_test` once: seed 60 T1a 0.857 < 0.90, recipe FAIL. Next: M09d with v7. Note: [learn/M09h-policy-v9.md](learn/M09h-policy-v9.md).

**M09d done 2026-11-11: the pre-registered test result, v7, scored once.** 200 test instances x 6 controllers (1,200 runs + 48 repeats for M09e's predictions), frozen at cbdc1ae. P1 (+0.54) and P2 (+0.28) pass, P3 fails on goal met (-0.09, though 9 s faster), P4 fails (-0.31 against LLM + formula), P5 passes (-0.013, margin 0.05), P6 passes (255/255 accepted feasible, 0 unverified applies). On this cell the chain LLM -> objective -> verifier carries the result and a formula does the actuator's job better than v7; whether RL earns its place needs a harder cell (the review's M10). Note: [learn/M09d-test-result.md](learn/M09d-test-result.md).

**After the review (2026-11-09):** M09g and M09h run before M09d (order M09g → M09h → M09d → M09e → M09f). Alongside: a rejection-cost line in the agent's reply, a marimo benchmark notebook, a related-work section, and the local-vs-remote LLM comparison of the proposal (§7) on an NVIDIA NIM key (M09i). DECISIONS 2026-11-09.

Then a "claims and evidence" synthesis in the README. Full plan: each checkpoint still starts with its own plan for approval.

## 8. Open questions

- ~~**Hard minimum PRB share per slice in the RL runner (discuss in M04a).**~~ **Decided and done (M04a): 5 % per slice, `sim.cell.apply_min_share`, retrained; see docs/learn/M04a-plumbing.md §5 and DECISIONS.md.** The M03 verifier
  only checks that the *targets* leave every slice at least 5 % of the PRBs; it cannot
  guarantee the split the policy chooses. A deterministic clamp in the runner (after the policy
  and the smoothing filter) would. If we add it, it must also be in the training environment,
  like the smoothing filter, so the policy trains with the actuator it will have when deployed.
  **Evidence from M04a** (`uv run python -m rl.plot_runner`, docs/learn/M04a-plumbing.md §5):
  during the demo URLLC surge, with "20 Mb/s URLLC within 10 ms" in force, the champion gives
  best effort 2 % of the PRBs (median latency ~41 s) for the whole 5 minutes.
- ~~**Policy jitter under the default objective.**~~ **Done 2026-11-07: alpha 0.10 served on both backends; on fresh seeds 1.8–3.3 % of the cell per s (was 5.6–8.5 %), at a cost of 2–4 s of reaction.** M04a measured 6–12 % of the
  PRBs moved per tick under the default objective (no targets, no bound), against 0.3 % under
  an explicit objective and ~2 % averaged over random objectives in M02. Options to weigh then:
  a different default objective, or a smaller filter alpha only for the default.
- **Over-serving the favoured slice. Reduced 2026-11-07, not gone: default weights unless a priority is stated, and implied targets for slices with none. S2: URLLC 68 % -> 46 % of the cell (needs 40 %), unmet eMBB + BE 28.6 -> 12.8 Mb/s (lowest possible 7.5). Importance 6 instead of 8 barely changed it. **Open:** a reward term and retraining, after M08c.** M04b live demo, URLLC surge, "at
  least 20 Mb/s and under 10 ms" with the LLM's importance 8/2/2: the policy gave URLLC its whole
  load (30.9 Mb/s, 79 % of the PRBs), so eMBB got 12 of its 37 Mb/s (p95 latency 5.3 s) and best
  effort sat at the 5 % floor. The target was met, but the cost to the other slices is
  higher than needed. M08a should report this cost, not only target compliance. Levers to weigh:
  the importance the plan step chooses (6 instead of 8), a reward term for share above target,
  or a revision triggered by over-compliance.
- ~~**May the agent relax a target the operator stated?**~~ **Decided (2026-11-02): option (c),
  only in eval mode. In demo mode a proposal that changes a stated value is sent back, and once
  the verifier has rejected the stated values the intent ends with its reason
  (`stated_changes`, agent/graph.py `set_objective`; DECISIONS.md).** M04b live demo:
  for "garantir 60 Mb/s au slice URLLC", the verifier rejected 60 Mb/s and the LLM replanned with
  30 Mb/s (the largest allowed), which the verifier accepted; the console showed it and the
  operator refused. Options: (a) keep it, and make the console flag "differs from the request";
  (b) forbid it: the plan prompt and a code check keep every stated number, so a verifier
  rejection of a stated value ends the intent with the verifier's reason; (c) allow it only in
  eval mode. Safety holds in every option (verifier + console); the question is fidelity to
  the operator.
- **FlexRIC's RIC request-id counter wraps after 65,536 requests (found in M07b, not fixed).**
  The near-RT RIC numbers every subscription and control from one counter stored as 16 bits
  (`near_ric.c`, `uint16_t ric_req_id = ric->req_id++`). After 65,536 requests a control reuses the
  id of a still-active KPM subscription; the RIC then forwards KPM reports to the slice-control
  xApp, which crashes (reproduced: crash at the 65,534th control since the RIC started). At our
  rate (~1 control per second) that is about 18 hours. **Current rule: restart the testbed
  (`testbed/down.sh && testbed/up.sh`) before ~60,000 controls**; every M07b/M08c run is far
  below. **Possible continuation:** our own RIC image built from FlexRIC v3.0.0 with a small patch
  that skips ids still in use (as we did for the gNB), or a supervisor that restarts the RIC and
  both xApps cleanly. Not done: it replaces the official RIC image, and no planned run needs it.
- **The RL policy oscillates on the testbed (found in M07b).** Cause tested on the simulator: KPIs
  about 2 s late. Mitigation in place: alpha 0.10 (since 2026-11-07 on both backends). Confirmed on the real cell, one
  run each (cell moved 10.8 → 4.3 % per s, URLLC ≥ 20 Mb/s 69 → 90 %, URLLC delay 9.4 → 2.3 s median;
  docs/learn/M07b-slice-control.md). It is reduced, not solved: the split still swings and URLLC's
  delay stays in seconds. **Open:** where the second tick of
  delay comes from (one candidate: `kpm_monitor`'s own 1 Hz tick between the KPM report and
  `kpm_latest`); and whether to retrain with a random observation delay, which would need fresh test
  seeds, since the held-out goal-test set has already been used twice. M08c: 2.0-3.9 % per s for RL only, 1.3-3.2 for LLM + RL, flat
  during S1's surge with a target in force.
- **A rejected request can leave a slice worse off (found in M08a, S3). Reduced 2026-11-07, not solved.** With eMBB and BE
  targets in force and URLLC's own request rejected as a conflict, URLLC got 8 % of the cell and about 8 s of delay. Implied
  targets (`verifier.verify.with_fair_shares`) raise it to 17.7 Mb/s and 2.9 s; an equal split gives 25 Mb/s and 2 s. The model
  still never relaxes the rejected target. **Open:** whether the agent should relax it or tell the operator what the rejection costs.
- **Done in M08d: capacity 79 Mb/s by a rule written first (`testbed/measure_capacity_m08d.py`), and traffic on a clock** (every surge started on time). The two items below record how they were found. M07b's own settings now give 73-76 Mb/s against its 69-72, which remains unexplained. **New from M08d:** the capacity depends on the split (73.5-80.2), and S3's result for LLM + RL sits inside that spread; at the right capacity RL only's swing fails S2 (0.43).
- **Testbed capacity in the verifier (decided for M08c, then contradicted by it).** `cell_info()` reports M07b's 69 Mb/s
  since 2026-11-07. M08c's standing queues show 81 Mb/s by KPM and 79 received by iperf3 (0.75 Mb/s per PRB). **Open:** why M07b
  measured 0.654 (candidate, untested: it offered 240 Mb/s in total, M08c at most 82.5), which value to use, and then rerunning
  S3, whose second intent is rejected at 69 and accepted from 75.
- **Traffic segments on a clock (found in M08c).** `eval.testbed_control.play_segments` waits for iperf3 to close, so the traffic depends
  on the controller (2 s late against 14-28 s). Fix before any further testbed matrix.
- **The swing when an intent expires (found in M08c, S2).** With eMBB's queue standing at t=300, the default objective gives eMBB up to 0.82 and
  URLLC about 0.13, and URLLC's delay reaches about 3 s for 23-28 s. `intent_met` doesn't see it, since it's outside the window. **Open:** whether to
  return to the default gradually, and whether the evaluation should score the seconds after an intent.
  **Measured in M08d** (`over_bound_after_s`): URLLC over 5 ms for 42-44 of the 60 s after S2's window for LLM + RL, 0 for static and LLM only.
- **The simulator's delay under capacity (found in M08c).** M/M/1 passes 5 ms at an equal split in S2; the real RLC delay stays near 0.6 ms until saturation.
  **Open, with any retraining:** a flat delay curve and queues beyond the 50 Mb cap (14-38 s seen on the cell).
- ~~**The testbed harness's validity check and leftover thread (found in M08d).**~~ **Fixed 2026-11-07 (`eval/testbed_control.py`, `testbed/traffic.py`):** every iperf3 (under `timeout`)
  and ping (`-w`) stops itself 60 s after the scenario's end, so a slow queue is logged (`stopped_at_deadline`), not invalid; a run is invalid if a process is still running
  30 s after that deadline (stuck); only `play()` clears `traffic_target`; leftover iperf3 containers are removed before each run; one warm-up LLM call before each LLM run's clock
  starts (`llm_warmup` event). Not yet run on the testbed: the next matrix is the first to use it. Any further matrix also needs its own results folder and table, not M08d's.
- **Over-serving blocks the S3 pair even when accepted (M08d what-if).** URLLC gets 0.63 of the cell for a 23 Mb/s target; eMBB and BE fall to 0.20 and 0.17. Same open item as
  over-serving above: a reward term and retraining.
- **LLM only's `finish` (found in M08c).** The model took a transient at 0 Mb/s for a verdict and stopped, and the log recorded `outcome: applied` next to a reason saying the goal can't be met. Not seen in M08d, where the traffic has no gap.
- **llama3.1 sets priority on guarantee intents (M04b, low priority).** Seen with llama3.1 only;
  qwen3:8b is the chosen model, so nothing depends on it unless the model changes.
- **Hermes: some turns get no reply (M05b, measured again in M08b).** The model returns an empty answer after
  retries and nothing is submitted. It recurs with the worker running (tune: t02, t16, t18 in all 3 repeats; held-out: h08),
  so the worker's queue isn't the cause. Cause not found. Fails safe; the operator sees it and can resend.
- **Agent worker is serial in demo mode (M05b).** An unconfirmed intent holds the next for up to
  120 s. Harmless in a one-operator demo; it matters only if several intents must overlap.
