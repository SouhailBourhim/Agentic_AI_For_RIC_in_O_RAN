# Claims and evidence

Every claim the project makes, with the experiment behind it, the kind of evidence, the number and its verdict. Negative results stay as rows. The numbers are quoted from the note that produced them (or from [RESULTS.md](RESULTS.md), which quotes the same notes). None is recomputed here. The final report and the defence slides start from this table.

**Kinds of evidence**, strongest first:
- **Pre-registered**: rules and hypotheses committed before the runs; scored once on held-out seeds or texts.
- **Testbed check**: on the OAI cell, rules frozen before the runs.
- **Descriptive**: measured, with no rule written beforehand, or on dev seeds used to build.
- **Engineering**: a property of the code, checked by CI tests (listed in [THREAT_MODEL.md](THREAT_MODEL.md)).

**The main result in one line:** the value is in the chain intent → objective → verifier. Behind that chain, an RL executor beats controllers without goals and the LLM alone, but a hand-written formula does the executor's job at least as well, even on a harder cell.

## 1. Can an LLM turn operator text into a correct objective?

| # | Claim | Experiment | Evidence | Number | Verdict | Note |
|---|---|---|---|---|---|---|
| C1 | A local 8B model extracts intents correctly | M08b, 20 held-out intents, scored once | Pre-registered | qwen3:8b 19/20 correct (95 % interval roughly 75–100 %, [review §3.5](REVIEW-2026-11-11.md)) against 10/20 for a regex parser; 6/6 ambiguous intents flagged | holds | [M08b](learn/M08b-eval-interface.md) |
| C2 | The model was chosen by measurement, not by taste | M04b, 5 local models, rule written first | Descriptive | qwen3:8b chosen | holds | [M04b](learn/M04b-langgraph.md) |
| C3 | A large remote model doesn't do better in the loop | M09i, qwen3:8b vs Nemotron 3 Ultra (NIM free tier) | Descriptive | held-out parse: remote 100 %, local 95 % (qwen3:8b misses one intent every time; the intervals overlap); closed loop, remote − local: LLM + RL −5.3 points [−9.6, −1.2], LLM only v2 −16.2 [−25.7, −7.3]; the loss comes from rate-limit waits, not worse choices (same objective on 37/37 intents) | holds, with that cause | [M09i](learn/M09i-local-vs-remote.md) |

## 2. Is it safe?

| # | Claim | Experiment | Evidence | Number | Verdict | Note |
|---|---|---|---|---|---|---|
| C4 | No dangerous request gets applied | M09f, 32 dangerous + 10 safe intents, 294 plays through the MCP gateway | Pre-registered | 0 unsafe changes applied; LLM + RL proposed 76 dangerous objectives, the verifier blocked 76; 30/30 safe controls applied (Wilson 95 % 0.89–1.00) | holds | [M09f](learn/M09f-dangerous-intents.md) |
| C5 | The verifier accepts only feasible goals | M09d P6, 265 goals on test seeds | Pre-registered | 255/255 accepted goals feasible; 0.962 of feasible goals accepted; 0 unverified applies | holds | [M09d](learn/M09d-test-result.md) |
| C6 | The chat front end can do almost nothing | M05b, Hermes isolated on an internal network | Descriptive (live) + Engineering | 73 out-of-scope attempts changed nothing; its words were not reliable (it once invented a command's output) | holds | [M05b](learn/M05b-hermes.md) |
| C7 | The six hard invariants hold in the code | CI on every push | Engineering | every invariant has its tests, cited row by row | holds | [THREAT_MODEL](THREAT_MODEL.md) |
| C8 | The fast loop has a safe state | M11a, injected faults | Engineering | equal split at once on stale KPIs, a dead actuator or an error; back to the policy after 3 clean ticks; simulator runs unchanged | holds | [M11a](learn/M11a-safe-state.md) |
| C9 | An accepted goal is met by the executor | M08d S3, M09b F4a | Descriptive | not always: S3 0.53 for LLM + RL against static's 0.98 | **fails**: residual risk R6; runtime monitor planned (M11b) | [THREAT_MODEL R6](THREAT_MODEL.md) |

## 3. Does RL earn its place as the executor?

| # | Claim | Experiment | Evidence | Number | Verdict | Note |
|---|---|---|---|---|---|---|
| C10 | A goal-conditioned policy can be trained to follow objectives it hasn't seen | M02b goal test, held-out | Pre-registered | 5 of 5 fresh training seeds pass (the first criterion, M02, failed and is kept) | holds | [M02](learn/M02-ddpg.md) |
| C11 | LLM + RL beats a static split when loads move (F1) | M09d P1, test seeds | Pre-registered | +0.537 [+0.490, +0.585] | holds | [M09d](learn/M09d-test-result.md) |
| C12 | LLM + RL beats RL without goals on a new goal (F2) | M09d P2 | Pre-registered | +0.279 [+0.189, +0.368] | holds | [M09d](learn/M09d-test-result.md) |
| C13 | LLM + RL beats the LLM setting the split itself on fast dynamics (F3) | M09d P3 | Pre-registered | goal met −0.089 [−0.164, −0.014], though 9.1 s faster to reach the target | **fails** | [M09d](learn/M09d-test-result.md) |
| C14 | LLM + RL is not worse than LLM + formula on F3 (margin 0.05) | M09d P4 | Pre-registered | −0.307 [−0.340, −0.277] | **fails**: the formula is better | [M09d](learn/M09d-test-result.md) |
| C15 | LLM + RL is not worse than static on easy intents (F5) | M09d P5 | Pre-registered | −0.013 [−0.028, −0.003] (margin 0.05) | holds | [M09d](learn/M09d-test-result.md) |
| C16 | The simulator's predictions hold on the real cell | M09e, 8 instances × 4 controllers × 2 on OAI | Testbed check | best controller predicted in 8/8 (rule 6); 23/24 predicted differences hold (rule 75 %); on F3 the formula beats RL on the testbed too (0.89 vs 0.66, 0.82 vs 0.65) | holds | [M09e](learn/M09e-testbed-check.md) |
| C17 | A better RL recipe closes the gap | M09c (v8), M09h (v9), held-out goal tests | Pre-registered | v8: one seed in five never learned; v9: one seed in five at 0.857 on T1a | **not confirmed**: v7 stays champion | [M09c](learn/M09c-policy-v8.md), [M09h](learn/M09h-policy-v9.md) |
| C18 | A harder cell (varying channel) is hard enough to test RL | M10b difficulty gate, dev | Pre-registered gate | the oracle met ≥ 0.90 of the window on 57.8 % / 61.5 % of intents (rule 80 %) | **fails**: M10 stopped as registered | [M10b](learn/M10b-fair-executors.md) |
| C19 | On that cell, RL beats a formula that measures each slice's rate | M10c′, 160 fresh test instances (F1-F3, F5), scored once | Pre-registered | m10 RL − formula_est −0.046 [−0.072, −0.022] (dev expectation −0.060); RL wins F2 +0.058 and F5 +0.039, loses F1 −0.037 and F3 −0.250 | **fails**: "formula better" | [M10c′](learn/M10c-rl-vs-formula.md) |

## 4. Does it run on a real 5G cell?

| # | Claim | Experiment | Evidence | Number | Verdict | Note |
|---|---|---|---|---|---|---|
| C20 | Per-slice KPIs over E2 are right, after a gNB fix | M07a; fix proposed upstream (duranta-project/openairinterface5g#1124) | Descriptive | stock OAI v3.0.0 mixes slices' counters; patched, KPM matches iperf3 to a median 2.5 % | holds | [M07a](learn/M07a-kpm.md) |
| C21 | The PRB split can be set over E2 | M07b, gNB patch + slice-control xApp | Descriptive | throughput follows the PRBs at 0.654 Mb/s per PRB; bad commands refused with a reason | holds | [M07b](learn/M07b-slice-control.md) |
| C22 | Where a goal needs RL on the real cell, LLM + RL serves it | M08d S1, median of 3 | Descriptive | S1 goal met: LLM + RL 0.96 against 0.34 (static, LLM only) and 0.21 (RL only) | holds, on one scenario | [M08d](learn/M08d-eval-testbed-rerun.md) |
| C23 | A simulator calibrated on the testbed reproduces it | M09a, 72 testbed runs replayed, rule written first | Pre-registered | first attempt failed; the one allowed retry passed (controller order right in 8/8 panels); LLM only can't be reproduced | holds, after one retry | [M09a](learn/M09a-sim-calibration.md) |
| C24 | On the testbed, the controller can run on the demand the gNB measures | M10d: calibration (rules first), then S1 × {offered, measured, throughput} × {LLM + RL, LLM + formula} × 3 | Testbed check | PDCP read 31 Mb/s offered while 4 were served; S1 intent met LLM + RL 0.956 measured vs 0.953 oracle (H-d margin 0.05), formula 0.960 vs 0.960; reaction 1–3 s slower | holds, on one scenario, after one calibration revision | [M10d](learn/M10d-pdcp-demand.md) |

## What is not claimed

- **Generality beyond one cell.** One cell, RF simulation (a clean channel on the testbed, a simulated channel in M10), one UE per slice.
- **End-to-end delay.** Latency is the RLC delay reported by the gNB.
- **Intent coverage.** The intents were written and annotated by one author; 20 held-out texts give a wide interval (C1).
- **Demand beyond arrivals.** The measured demand is what arrives at PDCP, not what is queued: a slice with a backlog and no new arrivals reads 0 (M10d). Only S1 was run on it.
- **That RL is needed.** Every comparison of RL with a hand-written executor behind the same chain went to the formula (C14, C16, C19).
