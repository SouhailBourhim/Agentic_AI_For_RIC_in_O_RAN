# Agentic AI for the RAN Intelligent Controller (O-RAN)

**An operator types a goal in plain French or English. A local LLM agent turns it into an
objective, a deterministic verifier checks it, and a goal-conditioned reinforcement-learning
xApp reallocates the radio resources of a 5G cell, every second.**

> "Protège l'URLLC : au moins 20 Mb/s et une latence sous 10 ms pendant 5 minutes."
>
> About five seconds later the agent has a verified objective waiting for the operator. Once it
> is confirmed, the URLLC slice gets about 80 % of the cell, its latency drops back under 10 ms,
> and the agent checks the result against the operator's numbers.

Personal research and engineering project by **Souhail Bourhim**. Everything runs on one laptop:
a simulated cell, an RL controller, an 8B LLM on the local GPU, and a real OpenAirInterface 5G
testbed in Docker.

![Live demo: the RL xApp serving an operator intent](docs/learn/img/M04b-grafana-after.png)
*Live run (Grafana). An intent arrives during a URLLC traffic surge: the RL xApp gives URLLC
about 80 % of the PRBs and its latency drops under the 10 ms bound. The two panels at the bottom
show the objective the LLM chose and the verifier confirmed.*

---

## Why this project is interesting

Putting an LLM into a network controller is attractive, because operators can speak in intents
instead of configuration. It is also risky, because LLMs are slow, non-deterministic and
sometimes wrong. This project's answer is to **give each component only the job it is good
at**:

| Component | Its job | What it is never allowed to do |
|---|---|---|
| **LLM agent** (LangGraph, qwen3:8b) | Understand the request, choose an objective, judge whether it worked | Touch the radio, set the PRB split, or skip the verifier |
| **Verifier** (plain Python) | Accept or reject every objective, with a reason | Use an LLM, randomness, the network or the clock |
| **RL xApp** (TD3, goal-conditioned) | Split the PRBs between slices every second | Wait for the LLM: it runs on its own loop |
| **Operator console** | Confirm or refuse each change, with a timeout | Be reachable from the chat interface |

These rules are written as six **hard invariants**. Each one has an automated test as soon as
the component it protects exists. For example: submitting an intent never writes to the policy
channel, the verifier module imports nothing non-deterministic, the RL runner imports nothing
from the agent, and the direct PRB-split tool cannot run outside the "LLM only" baseline.

## Architecture

![Architecture: language, safety, control and RAN layers](docs/img/architecture.png)

The figure follows a request from left to right, and the KPI store at the bottom closes the
loop. Dashed boxes are planned or partly built; see [Status](#status-and-roadmap). Two design
choices carry the architecture:

- **The LLM changes the objective, not the actions.** The RL policy takes the objective as part
  of its input: per-slice weights, minimum throughputs and a URLLC latency bound. It was
  trained on randomly sampled objectives, so a new operator goal needs no retraining. Without
  this, an LLM changing goals at run time would be pointless.
- **One backend interface.** The RL runner calls the same five methods whether the RAN is the
  Python simulator or the real OAI gNB through FlexRIC. Nothing above the runner knows which
  one is behind it.

## What happens to one request

![One request, step by step](docs/img/request-sequence.png)

The agent itself is an explicit LangGraph state machine. There is no free-running ReAct loop:
each node has a bounded number of retries, a step limit and a revision limit, and each limit
has a test.

![The agent's state machine and its limits](docs/img/agent-graph.png)

The run below is one real intent, step by step: the LLM calls are the blue bars at the top, and
the cell's response is underneath.

![One intent traced through the agent](docs/learn/img/M04b-trace.png)

## Results so far

**Goal-conditioned RL that passes a test it could fail.** Showing that a policy "follows the
objective" turned out to be the hard part.
- The first criterion (M02) failed, and the failure is kept on record. Investigating it showed
  the criterion was flawed: even a perfect controller could fail it.
- The new test compares the policy with an evaluation-only **oracle** that searches 861
  candidate splits every second using the true future loads. The thresholds were fixed before
  any training.
- On a held-out test run once, **5 of 5 fresh training seeds pass**: telling the policy the
  objective helps in 97 % of the cases where it matters, and it captures 70–88 % of the
  oracle's advantage. It earns 0.28 reward per step against 0.18 for an equal split.

![Held-out goal-conditioning test](docs/learn/img/M02b-test.png)

**The same policy under two objectives, on the same traffic.** Told "protect URLLC", it hands
URLLC most of the cell during a surge and keeps URLLC latency under the bound. Told "maximise
eMBB", it does not.

![Same policy, two objectives](docs/learn/img/M02-objectives.png)

**A verifier that explains itself.** Eight checks run in a fixed order: schema, validity window,
latency physics, PRB budget, per-slice floor, the policy's training range, conflicts with active
intents, and step size. Each rejection is a sentence with the numbers in it, and the LLM reads
it when it re-plans. When several intents are active, they are merged into the single objective
the RL xApp will follow, and that merge is what gets verified.

**Safety found by running it.** The first live demo showed best effort starved at 2 % of the
PRBs, with a median latency of 41 s.
- Fix: a 5 % per-slice floor, enforced by one deterministic function in the actuator, and
  present during training as well.
- The policy was retrained with the same frozen recipe and promoted through the registry's gate.
- Result: best effort now always gets at least 5 % of the PRBs, and URLLC latency during the
  surge improved from 4.9 ms to 4.1 ms.

**An LLM chosen by measurement, not by reputation.** Five local models were benchmarked on
annotated intents, under a selection rule written down before the run.
- Two models scored as well but did not fit entirely in 8 GB of VRAM. llama3.1:8b scored lower.
  granite4.2:8b returned schema-valid but empty answers.
- **qwen3:8b** won: 19/20 with prompt v1, then 20/20 after a general fix (an answer saying
  "understood" must actually contain a fact).
- Parsing takes about 1.3 s. A held-out set of 20 intents was written before the prompt
  existed, and it is reserved for the final evaluation.

**A real 5G testbed on a laptop, with three slices end to end.** It runs the OAI 5G core, an OAI
gNB in RF-simulator mode, the FlexRIC near-RT RIC and three UEs, one per slice (eMBB, URLLC, best
effort, each with its own S-NSSAI, DNN and subnet).
- Measured capacity: about 71–75 Mb/s. The simulator is calibrated to it.
- With the stock scheduler, slices do not matter: in overload each gets an equal 23–25 Mb/s and
  a 16 s queue, URLLC included. That is the gap the RL xApp is meant to close.
- **PRB-split control over E2 does not exist in OAI v3.0.0**, so the plan includes a small custom
  gNB build, documented as a decision.

**Real per-slice KPIs over E2, after fixing a bug in the gNB.** Our KPM xApp subscribes once per
slice and writes the same KPI record as the simulator. The agent, the MCP gateway and Grafana
then read testbed data with no change.
- With one subscription per slice, OAI v3.0.0's gNB computes each slice's throughput against
  another slice's counter (left: up to 34 000 Mb/s for 10 Mb/s of traffic). A patch of about 50
  lines keeps the previous values per UE.
- On our patched gNB, KPM matches iperf3 to a median 2.5 % per second (right), and its RLC delay
  tracks the ping RTT up to 15 s in overload.

![KPM per slice on the stock and the patched gNB](docs/learn/img/M07a-stock-vs-patched.png)

## Engineering practices

- **346 automated tests**, run by GitHub Actions on every push. No GPU, LLM or testbed is
  needed: the LLM is replaced by a scripted fake.
- **Pre-registration.** Pass/fail criteria, model-selection rules and held-out sets are
  committed before the experiment runs. Failed criteria stay in the record.
- **MLOps on a laptop:** local MLflow tracking of every training and evaluation run, a model
  registry, and a deterministic promotion gate. A policy becomes "champion" only with recorded
  test evidence.
- **Decision log.** [docs/DECISIONS.md](docs/DECISIONS.md) records more than 100 decisions, each
  with its date and reason, including the alternatives rejected.
- **A learning note per checkpoint** ([docs/learn/](docs/learn/)): what was built, how it works
  with file:line references, how to run and check it, the choices made, and questions to test
  understanding.
- **Live observability:** a Prometheus exporter and a provisioned Grafana dashboard show every
  slice, the objective in force, intent states and verifier decisions.

## Status and roadmap

| Checkpoint | What it delivers | Status |
|---|---|---|
| M01 | Gymnasium simulator: one cell, three slices (eMBB, URLLC, best effort), objective schema | Done |
| M02 / M02b | Goal-conditioned DDPG/TD3 policy, oracle-validated goal test | Done (5/5 seeds pass the held-out test) |
| M02c | MLflow tracking, model registry, promotion gate, CI | Done |
| M03 | Deterministic verifier and multi-intent merge | Done |
| M04a | Processes and plumbing: backend interface, KPI store, A1 emulation, RL runner, console, Grafana | Done |
| M04b | LangGraph agent with a local LLM, model benchmark, annotated intents | Done |
| M06-spike | OAI + FlexRIC testbed reconnaissance and simulator calibration | Done |
| M05a | MCP gateway with exactly three tools, CLI MCP client, live demo of the three cases | Done |
| M06 | Three network slices end to end on the testbed (OAI core, gNB, one UE per slice), capacity and load measured | Done |
| M07a | KPM monitoring xApp (one subscription per slice), patched gNB KPM, testbed KPIs in the store | Done |
| M05b | Hermes Agent as the chat interface, isolated | Next (can be cut) |
| M07b | PRB-split control on the real gNB (custom build), FlexRIC backend | Planned |
| M08a–c | Four-controller comparison (static, RL only, LLM only, LLM + RL) on the simulator and the testbed; extraction accuracy on the held-out intents | Planned |

The full plan, with interfaces, risks and how each hypothesis gets tested, is in
[docs/PLAN.md](docs/PLAN.md). The original design document (in French) is
[docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf](docs/Proposition_IA_agentique_RIC_ORAN_v2.pdf).

## Tech stack

**Python 3.12, uv**, pydantic and FastAPI. **gymnasium** and **stable-baselines3** (TD3/DDPG)
for the RL. **LangGraph** for the agent, with **Ollama** serving a local 8B LLM behind a small
client that can be swapped for a cloud model. **MLflow** for tracking and the registry,
**SQLite** in WAL mode for the shared state, **Prometheus** and **Grafana** for live
monitoring, and **matplotlib** and **marimo** for figures and notebooks. The testbed uses
**OpenAirInterface** (5G core, gNB, nrUE in RF simulation), **FlexRIC** and **Docker**.
Everything runs on WSL2 with an RTX 5070 laptop GPU.

## Try it

Requires [uv](https://docs.astral.sh/uv/), which installs the right Python by itself.

```bash
uv sync                              # install the locked dependencies
uv run pytest -q                     # 346 tests; no GPU, LLM or Docker needed
uv run python -m gateway.client tools # the MCP gateway's three tools
uv run python -m rl.plot_runner      # one intent through the whole chain, simulated (~10 s)
```

With [Ollama](https://ollama.com) and `qwen3:8b` installed:

```bash
uv run python -m agent.plot_trace                                  # one intent through the LLM agent
uv run python -m eval.extraction --parser qwen3:8b --split tune    # extraction score
```

The six-process live demo (RL runner, agent service, agent worker, exporter with Grafana, and
the console) is described step by step in
[docs/learn/M04b-langgraph.md](docs/learn/M04b-langgraph.md#5-how-to-run-it-and-check-it-works);
the same demo through the MCP gateway is in
[docs/learn/M05a-gateway.md](docs/learn/M05a-gateway.md#5-how-to-run-it-and-check-it-works).

The testbed needs Docker, the SCTP kernel module and two images we build (the patched gNB and
the KPM xApp). [testbed/README.md](testbed/README.md) has the commands; `testbed/up.sh` starts
everything in order, and `uv run python -m xapps.kpm_monitor` fills the KPI store from it.

## Repository layout

| Folder | Contents |
|---|---|
| [sim/](sim/) | Gymnasium environment: one cell, three slices, traffic scenarios |
| [rl/](rl/) | Training, goal-conditioned policy, oracle and goal test, RL runner (the xApp) |
| [verifier/](verifier/) | Objective schema, the eight checks, multi-intent merge (no LLM) |
| [agent/](agent/) | LangGraph agent, LLM client, internal tools, FastAPI service, worker |
| [store/](store/) | Shared state: KPI table, `a1_policy` timetable, intents, event log |
| [backends/](backends/) | The backend interface and the simulator backend (FlexRIC backend planned) |
| [console/](console/) | Operator confirmation console |
| [eval/](eval/) | Annotated intents, extraction scorer, model benchmark |
| [mlops/](mlops/) | MLflow tracking, registry, promotion gate |
| [monitoring/](monitoring/) | Prometheus exporter, Prometheus and Grafana configuration and dashboard |
| [xapps/](xapps/) | KPM monitoring xApp: C subscriber on FlexRIC plus the Python writer to the KPI store |
| [testbed/](testbed/) | OAI and FlexRIC configurations, gNB patch and build, traffic scripts, measurements |
| [notebooks/](notebooks/) | marimo notebooks to explore the simulator and the policy |
| [tests/](tests/) | pytest, including one test per hard invariant |
| [docs/](docs/) | Plan, decisions, learning notes, design document |

## Contact

Souhail Bourhim: [GitHub](https://github.com/SouhailBourhim)
