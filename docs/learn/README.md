# Learning notes: index

One note per checkpoint, in the order they were built. Each one has the same seven parts: what was built, where it sits in the architecture, how it works with file:line references, how to run and check it (with a figure), the choices made, and five questions with answers. The claims these notes support are gathered in [CLAIMS.md](../CLAIMS.md).

## Read these five first

1. [M01](M01-simulator.md): the cell everything runs on.
2. [M03](M03-verifier.md): the deterministic gate between the LLM and the radio.
3. [M04b](M04b-langgraph.md): how operator text becomes a verified objective.
4. [M09d](M09d-test-result.md): the main pre-registered result, and where RL loses to a formula.
5. [M10c′](M10c-rl-vs-formula.md): the same question on a harder cell, with the same answer.

## All notes

**Setup**
- [M00](M00-setup.md): the repository, `uv`, the tooling and the first test.

**Simulator and RL**
- [M01](M01-simulator.md): a gymnasium cell with three slices, and how load, PRBs, throughput and delay relate.
- [M02](M02-ddpg.md): the goal-conditioned DDPG/TD3 policy, the oracle and the goal test it must pass.
- [M02c](M02c-mlops.md): MLflow tracking, the model registry and the promotion gate.

**Verifier and agent**
- [M03](M03-verifier.md): the verifier's ordered checks, and why it is plain deterministic Python.
- [M04a](M04a-plumbing.md): the store, `a1_policy`, the RL runner, the agent service, the console and monitoring.
- [M04b](M04b-langgraph.md): the LangGraph agent, parse → plan → verify → confirm → observe → reflect, and how its LLM was chosen.

**Gateway and chat front end**
- [M05a](M05a-gateway.md): the MCP gateway with exactly three tools.
- [M05b](M05b-hermes.md): Hermes Agent as the chat interface, isolated on its own network.

**The real 5G cell**
- [M06-spike](M06-spike.md): first look at OAI + FlexRIC, what works and what doesn't.
- [M06](M06-testbed.md): three slices end to end on OAI in rfsim.
- [M07a](M07a-kpm.md): per-slice KPIs over E2, and the gNB bug fixed on the way.
- [M07b](M07b-slice-control.md): setting the PRB split over E2 (gNB patch and slice-control xApp).

**First evaluation: four controllers, three scenarios**
- [M08a](M08a-eval-sim.md): on the simulator.
- [M08b](M08b-eval-interface.md): the interface: intent extraction on held-out texts, latency, tool calls.
- [M08c](M08c-eval-testbed.md): on the testbed.
- [M08d](M08d-eval-testbed-rerun.md): on the testbed again, at its measured capacity.

**M09: beyond three scenarios**
- [M09a](M09a-sim-calibration.md): a simulator calibrated to the testbed and validated against it.
- [M09b](M09b-benchmark.md): generated scenarios and a 7-controller dev benchmark, with an oracle and LLM + formula.
- [M09c](M09c-policy-v8.md): policy v8, retrained in the calibrated setting; not confirmed.
- [M09d](M09d-test-result.md): the pre-registered result on test seeds, scored once.
- [M09e](M09e-testbed-check.md): the testbed checks M09d's written predictions.
- [M09f](M09f-dangerous-intents.md): 32 requests written to break a limit; 0 unsafe changes applied.
- [M09g](M09g-diagnostics.md): why the formula beats RL on dev, three diagnostics.
- [M09h](M09h-policy-v9.md): policy v9, five changes tested one at a time; not confirmed.
- [M09i](M09i-local-vs-remote.md): a local 8B model against a large remote one.

**M10: a harder cell**
- [M10a](M10a-varying-channel.md): a radio channel per slice, so Mb/s per PRB is unknown and moves.
- [M10b](M10b-fair-executors.md): fair non-learned executors and a difficulty gate that failed twice.
- [M10c′](M10c-rl-vs-formula.md): RL trained on that cell against the estimating formula, pre-registered.
- [M10d](M10d-pdcp-demand.md): on the testbed, the demand the gNB measures (PDCP volume) instead of the traffic script's rate.

**M11: safety of the fast loop**
- [M11a](M11a-safe-state.md): the runner's safe state on stale KPIs, a dead actuator or an error.
- [M11b](M11b-runtime-monitor.md): the runtime monitor: the formula takes over an intent RL keeps missing.
- [M11d](M11d-shielded-rl.md): the shielded RL executor: URLLC floored at the formula's share; reported, not served.
- [M11c](M11c-threat-model.md): the threat model, each threat with the test that covers it.

**M12: presentable**
- [M12](M12-presentable.md): the one-command demo, the claims table, this index and the licence.
