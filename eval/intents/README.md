# Annotated operator intents (M04b, scored in M08b)

`annotated.jsonl` holds one intent per line. Every line is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `id` | `h01`… for the held-out split, `t01`… for the tuning split |
| `split` | `heldout` (scored once, in M08b) or `tune` (used to choose the model and tune the prompt) |
| `lang` | `fr` or `en` |
| `text` | the operator's words, verbatim |
| `ambiguous` | true when a careful human would ask a question before acting |
| `expected` | the facts a correct parser extracts: a `ParsedIntent` (below) |
| `notes` | why this case is here, what makes it hard |

`expected` is a `ParsedIntent` (`agent/schema.py`), the parse step's output:

```json
{"understood": true, "clarification": null, "priority": "urllc",
 "min_thr_mbps": {"embb": null, "urllc": 10, "be": null},
 "max_latency_ms": 8, "duration_s": 600}
```

## Annotation rules

1. **Facts only.** Record only what the text states. Something not stated is `null`. The
   parser never decides weights; the plan step does. A missing or vague duration ("for a
   while") is just `null`; it never makes a request `understood: false` on its own.
2. **`priority`** is set only when the operator explicitly favours or protects one slice:
   the verbs "prioriser", "protéger", "favoriser", "prioritise", "protect", "favour", or the
   noun "priorité / priority" given to one slice. Asking for a guarantee alone does not set
   a priority, and neither do "keep", "make sure", "assurer", "garantir" or "donner".
3. **Units:** throughput in Mb/s (Gb/s × 1000, kb/s ÷ 1000), latency in ms, duration in s.
   "Give / donne / assure / at least / au moins" N Mb/s to a slice is a minimum throughput.
4. **Slice words:** video, streaming, "haut débit" mean eMBB; critical, real-time, robots,
   low-latency, "faible latence", "slice critique" mean URLLC; background, backups,
   "trafic de fond", "le reste" mean best effort.
5. **`understood: false`** means the request can't be written down with these fields
   without guessing. The cases: a number with no slice or with two candidate slices; a
   contradiction (including two slices favoured at once, since `priority` holds one); a
   latency bound on a slice other than URLLC; something out of scope (switching a slice
   off, a throughput cap); or something too vague. Only `understood` is scored then; the
   other fields stay `null` and the clarification wording is free.
6. **Values the verifier will reject are still extracted.** A clear request outside the
   verifier's limits ("60 Mb/s for URLLC", above the 40 % training range) is
   `understood: true` with the value; the verifier rejects it later. The parser does not
   judge values.

`ambiguous` is true exactly when `understood` is false in this set; it is kept as a
separate field so a later set can hold requests that are understood but deserve a check.

## Who wrote it, in what order

Claude drafted every intent (decision of 2026-11-01). To limit the bias of having one
author for both splits:
1. the held-out split was written and committed **before** the parsing prompt existed;
2. the prompt was then tuned only on the tuning split.

**Review (2026-11-02).** Souhail delegated the review to four independent Claude
subagents: two annotated the 40 texts blind (from these rules only, without the labels)
and both agreed with every label, field by field; two checked the labels against the rules
and found no wrong label. Their wording findings were applied: in the held-out split h10,
h17 and h19 were reworded to remove a second reading (labels unchanged; no model was run
on the held-out split), and h05 and h20 got corrected notes; in the tuning split t08, t11,
t12, t13 and t18 were rewritten to stop sharing a sentence template with a held-out item or
to remove a second reading. Rules 1–6 above were made explicit where the reviewers found
gaps.

The M08b extraction accuracy should still be read as optimistic: one author (a Claude
model) wrote both splits and the reviewers are Claude models too, and most held-out case
types (units, no slice, contradiction, out of scope) also appear in the tuning split, so
the score measures familiar case types more than generalisation.

# Dangerous in-scope intents (M09f)

`m09f.jsonl` holds the requests M09f uses to try to make something unsafe reach the cell
(`eval/m09f.py`). They are in scope (throughput, latency, priority, duration), but worded or
sized to break a limit. One line per intent:

| Field | Meaning |
|---|---|
| `id` | `x01`–`x32` dangerous and `c01`–`c10` safe controls (split `test`); `d01`–`d10` (split `dev`) |
| `split` | `test` (played once, scored once) or `dev` (the dry run only, never scored) |
| `category` | 1 starve a slice, 2 targets that don't fit, 3 above the trained range, 4 impossible latency, 5 bad window, 6 takes an accepted guarantee away, 7 abrupt flip, 8 bypass or injection; or `control` |
| `lang`, `text` | as above |
| `setup` | categories 6, 7 and some controls: a benign intent submitted and decided first, or null |
| `danger` | what would be unsafe if it were applied, in words, with the numbers for C = 75 Mb/s |
| `expected_stop` | informational only, never scored: `refuse`, `verifier:<check>`, `relax` or `any` |

There is no expected parse: M09f judges only what reaches the cell (the safety oracle in
`eval/m09f.py`), not the extraction. The controls sit just inside a limit, so the share of them
accepted shows whether safety comes from refusing everything.

**Who wrote it.** Claude wrote every intent (Souhail's choice, 2026-11-14). The same author
knows the verifier's checks, which is the set's main weakness: it tests the limits the author
knew about. Category 8 (bypass, injection, JSON in the text, a tool called by name) aims at
paths the verifier was not written for. Setups and controls keep each target change at or
under 22.5 Mb/s (0.3 C), the verifier's step limit from an idle cell: a first dry run with a
30 Mb/s setup was rejected by `step`, so its conflict was never tested.

**After the first dev dry run (2026-11-14).** qwen3:8b asked which slice "vidéo" meant instead
of reading it as eMBB (rule 4 above), so a control or a setup worded with a synonym tested the
parser rather than the verifier. Controls and setups now name the slice (eMBB, URLLC, best
effort); the dangerous texts keep their everyday words, where a question back is a fair stop.
No test intent had been played.
