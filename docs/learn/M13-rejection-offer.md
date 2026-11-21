# M13: the verifier answers a rejection with what would fit

## 1. What we built

When the verifier rejects a request that does not fit, it now also computes an **offer**. The offer is the largest
scaled-down version of the request that would be accepted, with the same latency bound, window and weights, sized
to fit a cell 7 % smaller than the one the verifier is told about (`verifier/counter.py`).

The offer is information, not an action:
- it travels with the decision to the intent record, so `get_status` shows it;
- in demo mode the operator reads it in the rejection's reason, and can submit it as a new request;
- in eval mode the LLM sees it after the rejection, and can propose it;
- in every case it is applied only after a new `verify()` and the usual apply step.

On the simulator, with rules written first, an operator who takes the offer gets more of what they asked for
without harming the guarantee already in force. On the fresh test split, the share of the request delivered rose
from **0.848 to 0.954**, and the guarantee in force was met 0.999 of the time in both arms. The LLM, given the offer,
copied it in 50 of 50 runs, with the same result.

**On the testbed the offer did harm.** In S3 the operator asks URLLC for 23 Mb/s under 5 ms, but URLLC's traffic is
31.5 Mb/s. The offer (16 Mb/s under 5 ms) fits on paper, yet a 5 ms bound can only be held by serving all of the
traffic. The monitor's fallback did exactly that, and the guarantee already in force (eMBB and BE at 23 Mb/s) fell
from 0.96 met to 0.22, in all three repeats. The verifier sizes a bound from the stated target and never sees the
load. That gap existed before M13; the offer is what let such a request through. **Offers are therefore off by
default (`RIC_OFFERS=0`)** until M13b gives the verifier the load.

## 2. Where it sits in the architecture

It lives in the **deterministic verifier** box of figure 2, plus a small change in how the **LangGraph agent**
reports a rejection.
- `with_offer` is plain Python: no LLM, no randomness, no clock. `tests/test_verifier_purity.py` now scans
  `counter.py` with the rest of the verifier.
- `verify()` itself is unchanged.
- Invariant 3 holds: an offer is never applied directly; it is a new proposal, verified like any other.
- Invariants 1 and 2 hold: the gateway keeps its three tools, and the offer reaches the operator through
  `get_status`.
- Invariant 6 holds: in demo mode the stated values are still never relaxed by the agent. The operator decides.

## 3. How it works

1. **The search.** `with_offer` (`verifier/counter.py:47`) runs only for the checks a smaller target can fix:
   budget, floor, training range, conflict and step. For p = 99 %, 98 % … 1 %, `scaled` (`verifier/counter.py:35`)
   multiplies every stated target by p and rounds down to whole Mb/s. Each stated slice must keep at least 1 Mb/s.
2. **Two verifications per candidate.** The first candidate that `verify()` accepts in a cell `OFFER_MARGIN`
   smaller (`verifier/counter.py:32`, `:52`) **and** in the real cell (`:61-62`) is the offer. The margin is
   1 − 73.5/79: the testbed measured 73.5-80.2 Mb/s depending on the split (M08d), against the 79 the verifier is
   told. An offer sized to the told cell would squeeze the guarantees already in force whenever the real cell is
   smaller.
3. **Where it is stored.** `Offer` and the new field `Decision.offer` are in `verifier/verify.py:75` and `:98`.
   `propose_objective` always attaches an offer when one exists (`agent/tools.py:160`), and `decision_dict` stores
   and logs it (`agent/apply.py:69`).
4. **What the agent does with it** (only with `Settings.offers`, env `RIC_OFFERS`, `agent/worker.py:35`):
   - in eval mode, the model's `VERIFIER REJECTED` line ends with "the largest that fits: set_rl_objective with
     {…}", so an 8B model can copy it (`agent/graph.py:296`, arguments built by `offer_arguments`, `:432`);
   - when the intent ends rejected, its reason ends with `offer_text`, e.g. "what fits now: urllc 16 Mb/s with
     latency below 5 ms (about 73% of the request), for the same duration" (`agent/graph.py:119`).
5. **The evaluation.**
   - `apply_scripted` plays an operator who takes the offer: it proposes the offer again, then applies it, and logs
     `offer_taken` (`eval/harness.py:127-132`).
   - The new family F6 (`eval/generator.py:185`) is a URLLC guarantee followed by an eMBB + best-effort request
     that does not fit beside it. Together with F4b it forms the set (`m13_instances`, `:261`).
   - `eval/m13.py` scores `delivered` (`:127`) per run (`row_of`, `:176`), checks that both arms got the same answer
     (`same_offers`, `:223`) and applies the frozen rules (`verdict`, `:233`).

```mermaid
flowchart LR
    P[proposal] --> V{verify}
    V -->|accepted| A[apply]
    V -->|rejected: budget, floor,<br/>training range, conflict, step| W[with_offer:<br/>p = 99..1 %, floor to Mb/s]
    W --> V2{verify in a cell 7 % smaller<br/>and in the real cell}
    V2 -->|first accepted| O[offer in the decision:<br/>intent record, get_status, events]
    O -->|eval: the model may propose it<br/>demo: the operator may re-submit it| P
```

## 4. How to run it and check it

```
uv run pytest -q tests/test_counter.py        # 19 tests: S3's offer, 400 random cases, no offer where it can't help,
                                              # the agent with offers on and off, the scripted operator, F6
uv run python -m eval.m13 --dev && uv run python -m eval.m13 --score   # ~1 min; -> docs/learn/data/M13-dev*
uv run python -m eval.m13 --score --final     # refused now: m13/test is spent
uv run python -m eval.plot_m13
```

**Result (rules in DECISIONS 2026-11-21, written and committed at baa2c81 before any scored run):**

| first-rejected intent | dev: reject | dev: offer | test: reject | test: offer |
|---|---|---|---|---|
| F4b delivered (10) | 0.844 | 0.874 | 0.807 | 0.900 |
| F6 delivered (40) | 0.854 | 0.961 | 0.858 | 0.967 |
| **all delivered (50)** | 0.852 | **0.944** | 0.848 | **0.954** |
| met as stated | 0.127 | 0.515 | 0.110 | 0.526 |
| met against the offer ("kept") | | 0.940 | | 0.942 |
| F6 guarantee in force, met | 0.999 | 0.999 | 0.999 | 0.999 |

- **G** (gain ≥ +0.05): +0.092 on dev, **+0.106** on test.
- **H** (no harm ≥ −0.02): 0.000 on both.
- Every first proposal was rejected with an offer, and both arms got the same one.
- The offer was taken and applied in 50 of 50 runs.

![Simulator results](img/M13-sim.png)

How to read it: one marker per arm and measure.
- The violet dot (offer taken) sits above the grey diamond (rejected) for every delivered measure, and far above it
  for "met as stated".
- On the right, the two markers coincide: the guarantee already in force loses nothing.

![One F6 run](img/M13-f6.png)

How to read it: dev f6_39, where the offer gains most.
- The window of the rejected request is lightly shaded, and the offered load is the grey area.
- Before 150 s both arms are the same run, so the violet line hides the grey one.
- Rejected (grey): neither eMBB nor best effort has a target, and the policy swings capacity between them, so each
  falls under its load every few seconds.
- Offer taken (violet): both slices have a target (24 and 25 Mb/s, under the stated 28 and 29) and are served flat
  at their full load.
- This is why "met as stated" jumps: the stated target counts as min(target, load), and the load here sits under
  the offer.

**LLM on the simulator** (`uv run python -m eval.m13 --llm-dev`, then `--llm-score`; dev, 50 instances × 2 arms,
reported, no gate; data `docs/learn/data/M13-llm.json`):

| first-rejected intent | llm_nooffer | llm_offer |
|---|---|---|
| ends applied | 0 / 50 | 50 / 50 |
| applied objective equal to the offer | | 50 / 50 |
| delivered (all) | 0.846 | **0.954** |
| met as stated | 0.124 | 0.534 |
| met against the offer ("kept") | | 0.907 |
| F6 guarantee in force, met | 0.993 | 0.993 |

The 8B model copied the offer every time it saw it, and without it never got the rejected intent applied. The gain
matches the scripted operator's (0.954 there too).

**The testbed's S3** (`uv run python -m eval.m13_testbed --play`, then `--score > docs/learn/data/M13-testbed.json`;
3 paired repeats, every repeat counted, 0.74 Mb/s per PRB):

| repeat | intent 2 delivered: nooffer | offer | intent 1 met: nooffer | offer |
|---|---|---|---|---|
| 1 | 0.004 | 0.933 | 0.980 | 0.237 |
| 2 | 0.006 | 0.943 | 0.957 | 0.201 |
| 3 | 0.006 | 0.939 | 0.960 | 0.224 |
| **median change** | | **+0.933** (helps) | | **−0.742** (rule ≥ −0.05: **harms**) |

![Testbed S3](img/M13-testbed.png)

How to read it: the left panel has one marker per repeat. The violet dots are high for the URLLC request and low for
the guarantee in force: the offer swapped one intent for the other. On the right, repeat 1 over time:
- Before 125 s both arms are the same.
- Rejected (grey): URLLC has no bound, gets 26 Mb/s with a queue of seconds (so delivered ≈ 0 against "under 5 ms"),
  and eMBB and BE keep their 23 Mb/s.
- Offer taken (violet): at 125 s the offer is applied. RL gives URLLC 0.32 of the cell, the queue grows, and after
  5 s of missed latency the monitor's fallback (shaded) takes over. To hold 5 ms with 31.5 Mb/s of traffic it gives
  URLLC 0.76 of the cell, and keeps doing so until the end. eMBB and BE drop to 7 Mb/s, under their 23 Mb/s guarantee.
- The LLM, seeing the latency missed in the first 20 s, asked for 23 Mb/s again, was rejected, took the offer again,
  and stopped at its revision limit.

Why the simulator did not see it: F6's URLLC load was 0.8-1.0 of its target, so a URLLC target always covered its
traffic. In S3 the traffic is 1.37 times the stated target and 1.97 times the offer.

## 5. Choices and simplifications

- **The offer ignores the load (the reason it is off).** A latency bound is a promise about the whole queue: if
  the traffic is above the throughput target, the bound costs the PRBs of the traffic, not of the target. The
  verifier only knows targets, so it accepts "16 Mb/s under 5 ms" for 31.5 Mb/s of traffic. Without offers the same
  gap stays for an operator who asks for it directly (THREAT_MODEL R10). M13b: give the verifier the measured load
  and size a bounded slice from max(target, load).
- **Same bound, smaller throughput.** *Rejected:* also offering the same throughput under a looser bound. On the
  testbed's S3 that would fit (23 Mb/s under 10 ms needs 36 % of the cell, not 49 %), but the bound is what makes
  a request URLLC; trading it away is the operator's call. A next step is a second offer, "the full request with
  this bound", or "the full request when i0001 ends".
- **A margin of 7 %, fixed before any run**, from a measurement (M08d's lowest capacity), not tuned. *Rejected:*
  no margin. The offer would then fill the told cell exactly, and the guarantee in force would pay whenever the
  real cell is smaller. With the margin, H held at 0.000.
- **Proportional scaling, whole Mb/s.** All stated targets shrink by the same factor, so a two-slice request keeps
  its own balance. Rounding down keeps the offer in the operator's units and on the safe side.
- **Offers off for earlier studies.** The harness passes `offers` explicitly, and `eval.testbed_control.play` sets
  `RIC_OFFERS=0` unless told otherwise, so every earlier run replays as it was.
- **F6's order is the reverse of S3's.** With the simulator's URLLC base latency (0.59 ms) the bound costs little
  headroom, and the step rule caps first guarantees at 0.3 C. So "URLLC second" only conflicts at 30-31 Mb/s under
  5 ms. The testbed's S3 (told base latency 2 ms) keeps S3's order.
- **The scripted operator always says yes.** It measures what taking the offer is worth, not how often an operator
  would. In the LLM arm, the model decides.
- **F4b's offer is bounded by the step rule** (0.3 of the smaller cell, about 22 Mb/s from an idle cell), not by
  the training range. That is why its gain is the smaller one.
- **A real system** would let the operator accept the offer in one click on the console, would offer the "later"
  and "looser bound" alternatives, and would record which offers operators accept.

## 6. Questions to check understanding

1. *Why is the offer computed in the verifier and not by the LLM?* The verifier is the only place that knows
   exactly what fits (budget, floor, conflicts, steps). An LLM guess would still need verifying, and an 8B model
   doesn't do this arithmetic reliably; the testbed runs showed it repeating the rejected proposal.
2. *Why is every candidate verified twice?* In the smaller cell, so the offer keeps the 7 % margin; in the real
   cell, because that is the check the offer will face when it is proposed.
3. *Can an offer break invariant 3?* No. It is only data in the decision. To run, it is proposed and passes
   `verify()` again, then `apply_intent`, as `test_the_scripted_operator_takes_the_offer_through_the_verifier`
   checks.
4. *Why did the offer harm the guarantee in force on the testbed and not on the simulator?* On the testbed URLLC's
   traffic (31.5 Mb/s) was well above the offer (16 Mb/s). Holding 5 ms means serving the whole queue, so the
   monitor's fallback gave URLLC 0.76 of the cell. On the simulator F6's URLLC traffic never exceeded its target.
5. *Why did "met as stated" rise from 0.11 to 0.53 when the offer asks for less than was stated?* Met counts a
   target as min(target, load). In many F6 runs the real load is under the offer, so a slice that now has a target
   is served its whole load. Rejected, it had no target and the policy left it short.
