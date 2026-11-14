# Agentic AI for the RAN Intelligent Controller (O-RAN)

[![tests](https://github.com/SouhailBourhim/Agentic_AI_For_RIC_in_O_RAN/actions/workflows/tests.yml/badge.svg)](https://github.com/SouhailBourhim/Agentic_AI_For_RIC_in_O_RAN/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

**A network operator types a goal in plain French or English. An LLM agent turns it into an
objective, a deterministic verifier checks it, and a reinforcement-learning controller
re-divides the radio resources of a 5G cell every second. The LLM never touches the radio.**

> *"Protège l'URLLC : au moins 20 Mb/s et une latence sous 10 ms pendant 5 minutes."*
>
> About five seconds later a verified objective is waiting for the operator's confirmation. Once
> confirmed, the low-latency slice gets the share of the cell it needs and its delay drops back
> under 10 ms.

![Live demo: the RL controller serving an operator intent](docs/learn/img/M04b-grafana-after.png)
*Live run (Grafana). An intent arrives during a traffic surge on the low-latency slice: the
controller gives it most of the radio resources and its delay falls under the 10 ms bound.*

Built end to end by **Souhail Bourhim**, on one laptop: a simulated cell, an RL controller, an
8B LLM on the local GPU, and a real open-source 5G network (OpenAirInterface core, base station
and three phones, with the FlexRIC controller) running in Docker.

## At a glance

| | |
|---|---|
| **Problem** | LLMs let operators state intents instead of writing configuration, but they are slow, non-deterministic and sometimes wrong. A mobile network can't be steered by something like that directly. |
| **Approach** | Give each part only the job it is good at: the LLM *understands*, plain Python *verifies*, RL *acts* in real time. Safety comes from what each part can reach, not from what the LLM promises. |
| **Runs on** | A Python simulator and a real OAI 5G cell, behind one interface. The agent, verifier and controller don't know which one is underneath. |
| **Scale** | ~22,000 lines of Python and C, 500+ automated tests in CI, 21 written technical notes, ~200 logged design decisions. |

## Highlights

- **The idea works on a real 5G cell.** When a goal needs the network to adapt to a traffic
  surge, the full design keeps the operator's goal met **96 % of the time**, against 34 % for a
  fixed split and 34 % for an LLM setting the split itself.
- **A local 8B model reads operator requests correctly:** 19 of 20 held-out requests in French
  and English, against 10 for a regular-expression parser, and it flags all 6 ambiguous ones
  instead of guessing.
- **The chat interface can't do harm.** It reaches exactly three tools (`submit_intent`,
  `get_kpis`, `get_status`) on an isolated network. 73 attempts to make it do anything else
  changed nothing.
- **Found and fixed a bug in OpenAirInterface.** The base station computed each slice's
  throughput against another slice's counter. After a ~50-line patch, its reports match
  independent measurements to a median 2.5 %.
- **Added a feature the open-source stack lacks:** per-slice radio-resource control over the E2
  interface (a base-station patch plus a control xApp in C).
- **Negative results are kept.** A retrained controller was refused by the project's own
  promotion gate. A simple hand-written formula currently matches the RL controller, and the
  README says so (see [Honest status](#honest-status)).

## How it works

![Architecture: language, safety, control and RAN layers](docs/img/architecture.png)

| Component | Its job | What it is never allowed to do |
|---|---|---|
| **LLM agent** (LangGraph, qwen3:8b via Ollama) | Understand the request, choose an objective, judge whether it worked | Touch the radio, set the resource split, or skip the verifier |
| **Verifier** (plain Python, 8 checks) | Accept or reject every objective, with a reason in numbers | Use an LLM, randomness, the network or the clock |
| **RL controller** (TD3, goal-conditioned) | Split the cell's resource blocks between three slices every second | Wait for the LLM: it runs on its own loop |
| **Operator console** | Confirm or refuse each change, with a timeout | Be reachable from the chat interface |

These rules are six **hard invariants**, each with an automated test. Two design choices carry
the rest:

- **The LLM changes the objective, not the actions.** The controller takes the objective as part
  of its input and was trained on randomly sampled objectives, so a new operator goal needs no
  retraining.
- **One backend interface.** The controller calls the same five methods whether the network is
  the simulator or the real cell.

<details>
<summary>One request step by step, and the agent's state machine</summary>

![One request, step by step](docs/img/request-sequence.png)

The agent is an explicit LangGraph state machine, not a free-running loop: each node has bounded
retries, a step limit and a revision limit, each with a test.

![The agent's state machine and its limits](docs/img/agent-graph.png)

</details>

## Results

Four controllers on the real cell, share of the operator's time window in which the goal was met
(median of 3 runs):

| Goal | Fixed split | RL alone | LLM alone | **LLM + RL** |
|---|---|---|---|---|
| Protect the low-latency slice before a traffic surge | 0.34 | 0.21 | 0.34 | **0.96** |
| Same kind of goal, stated mid-run (a fixed split already meets it) | **1.00** | 0.43 | **1.00** | 0.97 |
| Guarantee two other slices at once | **0.98** | 0.12 | 0.80 | 0.53 |

![Goal met per controller on the real cell](docs/learn/img/M08d-intent-met.png)

Where the goal needs adaptation, only LLM + RL delivers it. Where a fixed split is already
enough, the RL controller costs something. The detailed record, including a 200-scenario
benchmark with seven controllers and a simulator calibrated against 72 testbed runs, is in
**[docs/RESULTS.md](docs/RESULTS.md)**.

### Honest status

- **The pre-registered test result** ([M09d](docs/learn/M09d-test-result.md)): 200 fresh simulated
  scenarios, rules committed before the first run, scored once. Turning the operator's goal into an
  RL objective beats a fixed split under load (+0.54 goal met) and an RL policy that never hears the
  goal (+0.28); the verifier accepted only feasible goals (255 of 255) and nothing reached the cell
  unverified. **But RL is not what makes it work:** the same agent and verifier with a hand-written
  formula score 0.93 against LLM + RL's 0.78 (over all 265 goals), and an LLM setting the split itself
  holds fast-changing goals better (0.70 against 0.61).
- **The real cell confirms it** ([M09e](docs/learn/M09e-testbed-check.md)): 8 of the test scenarios,
  4 controllers, 2 repeats each on OAI + FlexRIC (64 runs, one night), judged against the
  simulator's predictions written beforehand. **Pass:** the predicted best controller won all 8, and
  23 of 24 predicted differences held. The negative result carries over too: on fast-changing load the
  formula beats RL on the real cell as well (0.89 and 0.82 against 0.66 and 0.65).
- **The open question.** The simulated cell is likely simple enough for arithmetic to be optimal.
  Settling whether RL is needed on a harder, more realistic cell is the next step
  ([review and roadmap](docs/REVIEW-2026-11-11.md)).
- **Limits.** One cell, simulated radio channel, one phone per slice, delay measured at the radio
  link layer (not end to end). On the real cell, the controller is told each slice's offered
  traffic rate by the traffic script, because the base station doesn't report it.
- **Still to run:** the test result's confirmation on the real cell (its predictions are written),
  a set of dangerous-but-valid requests, and a local-versus-cloud LLM comparison.

## What this project demonstrates

| Area | In practice |
|---|---|
| **AI agents** | LangGraph state machine with bounded retries; tool calling on a local 8B model; MCP server; model chosen by benchmark among five; out-of-scope tests on the chat interface |
| **Reinforcement learning** | Goal-conditioned TD3 (stable-baselines3), custom Gymnasium environment, an oracle to grade the policy against, ablations, sim-to-real calibration |
| **5G / O-RAN** | OpenAirInterface core, gNB and UEs; network slicing with three S-NSSAIs; FlexRIC; E2 service models (KPM, slice control); xApps in C and Python; patches to the gNB |
| **Safety engineering** | Deterministic verifier between the LLM and the network; least-privilege tool surface; isolated container network; human confirmation with timeout; one test per invariant |
| **MLOps** | MLflow tracking and model registry; a deterministic promotion gate that has refused a model; held-out sets that refuse in code to be scored twice |
| **Software engineering** | Python 3.12 with type hints and pydantic; FastAPI; SQLite (WAL); 500+ tests and lint in GitHub Actions; Docker; Prometheus and Grafana |
| **Scientific method** | Pass/fail criteria committed before each experiment; failures kept on record; every decision dated and justified in a [decision log](docs/DECISIONS.md) |

## Try it

Requires [uv](https://docs.astral.sh/uv/), which installs the right Python by itself. No GPU,
LLM or Docker is needed for these:

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run python -m gateway.client tools
```

The last command lists the gateway's three tools. With a trained controller in the local model
registry, `uv run python -m rl.plot_runner` plays one request through the whole chain on the
simulator in about 10 seconds.

<details>
<summary>With a local LLM, the live demo, and the real 5G testbed</summary>

With [Ollama](https://ollama.com) and `qwen3:8b` installed:

```bash
uv run python -m agent.plot_trace
```

```bash
uv run python -m eval.extraction --parser qwen3:8b --split tune
```

- Live demo (controller, agent service and worker, Grafana, console):
  [M04b note](docs/learn/M04b-langgraph.md#5-how-to-run-it-and-check-it-works); through the MCP
  gateway: [M05a note](docs/learn/M05a-gateway.md#5-how-to-run-it-and-check-it-works).
- Hermes chat interface in Docker:
  [M05b note](docs/learn/M05b-hermes.md#5-how-to-run-it-and-check-it-works).
- The testbed needs Docker, the SCTP kernel module and two images built here (patched gNB, KPM
  xApp): [testbed/README.md](testbed/README.md). `testbed/up.sh` starts everything in order.

</details>

## Tech stack

**Python 3.12, uv**, pydantic, FastAPI · **gymnasium**, **stable-baselines3** (TD3), PyTorch ·
**LangGraph**, **Ollama** (local 8B LLM behind a swappable client), **MCP** Python SDK ·
**MLflow**, SQLite, **Prometheus**, **Grafana**, matplotlib, marimo · **OpenAirInterface** (5G
core, gNB, nrUE), **FlexRIC**, C xApps, **Docker**, on WSL2 with an RTX 5070 laptop GPU.

## Find your way around

| Read this | To learn |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Every result in detail, with figures |
| [docs/REVIEW-2026-11-11.md](docs/REVIEW-2026-11-11.md) | What is missing and the roadmap |
| [docs/learn/](docs/learn/) | One technical note per checkpoint: what was built, how it works, how to run it |
| [docs/PLAN.md](docs/PLAN.md), [docs/DECISIONS.md](docs/DECISIONS.md) | The plan and every dated decision |
| [Design document (French)](docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf) | The original proposal |

<details>
<summary>Repository layout</summary>

| Folder | Contents |
|---|---|
| [sim/](sim/) | Gymnasium environment: one cell, three slices, traffic, reward |
| [rl/](rl/) | Training, goal-conditioned policy, goal test, RL runner (the xApp) |
| [verifier/](verifier/) | Objective schema, the eight checks, multi-intent merge (no LLM) |
| [agent/](agent/) | LangGraph agent, LLM client, internal tools, FastAPI service, worker |
| [gateway/](gateway/) | MCP gateway (exactly three tools), CLI client, isolated Hermes configuration |
| [console/](console/) | Operator confirmation console |
| [backends/](backends/) | Backend interface, simulator backend, FlexRIC (testbed) backend |
| [store/](store/) | Shared state: KPI table, `a1_policy` (an emulation of O-RAN A1), intents, event log |
| [xapps/](xapps/) | KPM monitoring xApp, slice-control xApp (C, E2), RL xApp image |
| [testbed/](testbed/) | OAI and FlexRIC configurations, gNB patches and build, traffic scripts |
| [eval/](eval/) | Annotated intents, controller harness, calibration, scenario generator, benchmark, oracle |
| [mlops/](mlops/) | MLflow tracking, registry, promotion gate |
| [monitoring/](monitoring/) | Prometheus exporter, Prometheus and Grafana configuration |
| [notebooks/](notebooks/) | marimo notebooks to explore the simulator and the policy |
| [tests/](tests/) | pytest, including one test per hard invariant |

</details>

<details>
<summary>Checkpoint status</summary>

| Checkpoint | What it delivers | Status |
|---|---|---|
| M01 | Gymnasium simulator: one cell, three slices, objective schema | Done |
| M02 / M02b / M02c | Goal-conditioned TD3 policy and oracle-validated goal test; MLflow, registry, CI | Done |
| M03 | Deterministic verifier (8 checks) and multi-intent merge | Done |
| M04a / M04b | Backend interface, KPI store, A1 emulation, RL runner, console, Grafana; LangGraph agent and LLM benchmark | Done |
| M05a / M05b | MCP gateway with exactly three tools; Hermes Agent as an isolated chat interface | Done |
| M06 / M07a / M07b | OAI testbed with three slices; KPM xApp and gNB fix; PRB-split control over E2 | Done |
| M08a–d | Four-controller comparison on the simulator and the real cell; interface evaluation | Done |
| M09a / M09b | Calibrated and validated simulator; generated scenarios and a 7-controller benchmark | Done |
| M09c | Policy v8 in the calibrated setting | Done: not confirmed, v7 stays champion |
| M09g | Diagnostics behind the RL-versus-formula gap | Done |
| M09h | Policy v9, pre-registered | Done: not confirmed, v7 stays champion |
| M09d | Pre-registered test-seed result, scored once | Done: P1, P2, P5, P6 pass; P3, P4 fail (the formula beats RL) |
| M09e | Testbed confirmation of M09d's predictions, scored once | Done: pass (best 8/8, pairs 23/24); the formula still beats RL on F3 |
| M09f | Dangerous intents | Planned |

</details>

## Contact

Souhail Bourhim · [GitHub](https://github.com/SouhailBourhim)
