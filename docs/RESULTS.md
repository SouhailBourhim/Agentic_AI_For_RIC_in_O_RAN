# Results in detail

The full record of what was measured, checkpoint by checkpoint. The [README](../README.md) has
the short version. Every result has a learning note in [learn/](learn/) with the figures, the
commands to reproduce it, and its limits. Criteria and selection rules were committed before the
runs, and failures stay on record.

Two things to keep in mind when reading the testbed numbers: the controller is given each slice's
offered rate by the traffic script (KPM does not measure it; see the
[M07b note](learn/M07b-slice-control.md)), and "latency" is the RLC delay reported by the gNB, not
an end-to-end delay.

## The building blocks work, on the simulator and on a real 5G cell

- **A goal-conditioned policy that passes a test it could fail** ([M02/M02b](learn/M02-ddpg.md)). The test
  compares the policy with an oracle that searches 861 splits per second with the true future
  loads. On a held-out set scored once, 5 of 5 fresh training seeds pass. The first criterion
  (M02) failed and the failure is kept: it turned out even a perfect controller could fail it.
- **An LLM chosen by measurement** ([M04b](learn/M04b-langgraph.md)). Five local models
  on annotated intents under a rule written first; qwen3:8b won. On 20 held-out intents scored
  once ([M08b](learn/M08b-eval-interface.md)) it read 19 correctly, against 10 for a regex
  parser, and flagged all 6 ambiguous ones.
- **A chat interface that can do almost nothing** ([M05b](learn/M05b-hermes.md)). Hermes
  Agent runs on an internal Docker network whose only exits are the gateway and Ollama. 73
  out-of-scope attempts changed nothing. Its *words* were not reliable (it once invented the
  output of a command it could not run), which is why safety rests on what it can reach.
- **Three slices end to end on OAI** ([M06](learn/M06-testbed.md)), **per-slice KPIs over E2 after fixing
  a gNB bug** ([M07a](learn/M07a-kpm.md): OAI v3.0.0 computed each slice's throughput
  against another slice's counter; a ~50-line patch fixes it, and KPM then matches iperf3 to a
  median 2.5 %), and **PRB-split control over E2**, which OAI v3.0.0 does not have
  ([M07b](learn/M07b-slice-control.md): a gNB patch plus a slice-control xApp; throughput
  follows the PRBs at 0.654 Mb/s per PRB, and bad commands are refused with a reason).

![KPM per slice on the stock and the patched gNB](learn/img/M07a-stock-vs-patched.png)

## The four controllers, on three hand-made scenarios

Static split, RL only (no goals), LLM only (the LLM sets the split itself) and LLM + RL, on the
simulator ([M08a](learn/M08a-eval-sim.md)) and the real cell
([M08c](learn/M08c-eval-testbed.md), [M08d](learn/M08d-eval-testbed-rerun.md)).
Real cell at its measured capacity (M08d), share of the operator's window in which the goal was
met, median of 3 repeats:

| Goal | Static | RL only | LLM only | LLM + RL |
|---|---|---|---|---|
| S1: URLLC ≥ 21 Mb/s, < 10 ms, before a surge | 0.34 | 0.21 | 0.34 | **0.96** |
| S2: URLLC ≥ 19 Mb/s, < 5 ms, stated mid-run | **1.00** | 0.43 | **1.00** | 0.97 |
| S3: eMBB and BE ≥ 23 Mb/s | **0.98** | 0.12 | 0.80 | 0.53 |
| S3: URLLC ≥ 23 Mb/s, < 5 ms (rejected by the verifier) | 0.00 | 0.18 | 0.00 | 0.00 |

- **Where a goal needs RL, the design works:** in S1 only LLM + RL serves the surge, on the
  simulator and on the real cell (URLLC delay 0.6 ms median).
- **Where it doesn't, it costs:** the policy over-serves URLLC, and a static split does better on
  S3. M08a's large S2 win came from the simulator's delay model and vanished on the real cell.
- **The simulator-trained policy oscillated on the real cell** (period 10–12 s) because the
  testbed shows a split's effect 2 ticks later. Stronger smoothing reduced it without retraining.

![Intent met per controller, M08d with M08c's medians](learn/img/M08d-intent-met.png)

## Beyond three scenarios (M09, in progress)

Three scenarios on a 7-minute-per-run testbed cannot carry a general claim, so M09 moves the
comparison to a simulator calibrated against the testbed, with generated scenarios.

- **A calibrated, validated simulator** ([M09a](learn/M09a-sim-calibration.md)): flat RLC
  delay, real buffer, measured capacity, KPIs 2 ticks late, every parameter from static testbed
  runs. Judged by replaying 72 testbed runs under a rule written first: **the first attempt
  failed**, the one allowed retry passed (controller order right in 8/8 panels). LLM only can't
  be reproduced by any simulator, since its choices turn on irrelevant input details.
- **A dev benchmark of 200 generated instances × 7 controllers**
  ([M09b](learn/M09b-benchmark.md)), with three references: an oracle that knows the coming
  loads, LLM + formula (the same agent and verifier with arithmetic instead of RL), and a fairer
  LLM only. Mean intent met (dev seeds, used to build, **not a claim**):

  | | F1 adapt to load | F2 new goal | F3 fast dynamics | F4 verifier edge | F5 easy |
  |---|---|---|---|---|---|
  | **LLM + RL** (policy v7) | **0.93** | 0.94 | 0.60 | 0.61 | 1.00 |
  | Static | 0.40 | 1.00 | 0.23 | 0.73 | 1.00 |
  | RL only | 0.62 | 0.70 | 0.24 | 0.36 | 0.94 |
  | LLM only v2 | 0.79 | 1.00 | 0.68 | 0.77 | 1.00 |
  | LLM + formula | 0.97 | 1.00 | 0.92 | 0.84 | 1.00 |
  | Oracle | 1.00 | 1.00 | 0.98 | 0.95 | 1.00 |

  LLM + RL beats the controllers without goals and the LLM alone on adaptation, and repeats
  within 0.006 across plays (LLM only swings by up to 0.99). But **a simple formula behind the
  same agent and verifier matches or beats it**, most clearly on fast dynamics: the RL policy had
  no load history in its input.
- **Policy v8: retrained, not confirmed** ([M09c](learn/M09c-policy-v8.md)). The training was moved to the calibrated cell
  with late KPIs, intent expiries, on/off loads and a 30 s load history in the observation. On
  the dev benchmark, LLM + RL rose from 0.79 to 0.85 mean intent met (F3 0.60 → 0.76), and a
  4-recipe ablation showed the load history is what fixes F3. On the held-out goal test, scored
  once, **the recipe failed**: one seed in five never learned, and the others missed one threshold
  by up to 0.04. By the rule set beforehand, **v7 stays the champion**.

Next: the pre-registered result on fresh test seeds (M09d), a testbed confirmation of its written
predictions (M09e), and dangerous in-scope intents (M09f). The plan is in
[PLAN.md](PLAN.md) §9.

## Limits to keep in mind

One cell, RF simulation (clean channel), one UE per slice, one author for the intents, bounds on
RLC delay rather than end to end, and an 8B local model. The RL policy has so far shown its value
over static splits and LLM-only control, not over a hand-written formula.

