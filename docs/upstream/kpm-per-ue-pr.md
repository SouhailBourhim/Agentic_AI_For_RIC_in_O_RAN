# Upstream pull request: KPM previous counters per UE (draft)

Status on 2026-11-16: **PR opened**, https://github.com/duranta-project/openairinterface5g/pull/1124
(from the fork `SouhailBourhim/openairinterface5g`, branch `kpm-per-ue-prev-values`).
Checks at opening: signed commits and commit emails passed; EasyCLA failing until the CLA is
signed; the CI label check waits for a maintainer to add a label.

- Upstream: `duranta-project/openairinterface5g`, base branch `develop` (at `cb98db5b`, integration 2026.w46).
- Local branch: `kpm-per-ue-prev-values` in `var/build/oai-upstream` (git-ignored clone).
- Two commits:
  1. `E2 KPM: keep previous counters of delta measurements per UE` (the fix, one file).
  2. `E2 KPM: remove the ue_idx parameter of the measurement functions` (cleanup, can be dropped
     if the maintainers prefer).

## What changed compared with `testbed/gnb/kpm-per-ue.patch` (v3.0.0)

| | v3.0.0 patch (M07a) | Upstream branch |
|---|---|---|
| Key of the table | RNTI | `rrc_ue_id` (also filled in CU and CU-UP, where there is no RNTI) |
| PDCP SDU volume (`DRB.PdcpSduVolumeDL/UL`) | not fixed | fixed, same table |
| Table full | slot `rnti % 16` (could be a live UE) | the entry reported the longest time ago |
| `ue_idx` parameter | marked unused | removed (second commit) |

## What is left for you

1. Sign the CLA through the EasyCLA link in the bot's comment on the PR (with the commit email).
2. Answer the review. Commits are signed with `~/.ssh/id_ed25519_git_signing` (registered on
   GitHub as a signing key; git is configured for it in `var/build/oai-upstream` only) and carry
   `Signed-off-by` (yours) and `Assisted-by: Claude:claude-fable-5-2`. After a change, push with
   `git push --force-with-lease fork kpm-per-ue-prev-values`.

## PR title

E2 KPM: keep previous counters of delta measurements per UE

## PR text

`DRB.PdcpSduVolumeDL/UL`, `DRB.UEThpDl/Ul` and `RRU.PrbTotDl/Ul` are reported as the difference
between a counter and its value at the previous report. The previous values are stored in arrays
indexed by `ue_idx`, the position of the UE in the list of UEs matched by one subscription.

With one KPM subscription per S-NSSAI and one UE per slice, every subscription has its UE at
position 0, so the three UEs share one slot and each difference is taken against another UE's
counter. The unsigned differences wrap around.

**How to reproduce:** rfsim, three UEs on three S-NSSAIs, one KPM subscription per S-NSSAI
(report style 4, 1 s period) asking for `DRB.UEThpDl` and `RRU.PrbTotDl`, iperf3 downlink on
each UE.

**Before / after** (rfsim, 106 PRBs, monolithic gNB, iperf3 UDP downlink 30 / 10 / 5 Mb/s on the
three slices at once, mean over 20 one-second reports):

| Slice | iperf3 received | `DRB.UEThpDl`, `develop` | `DRB.UEThpDl`, this PR | `RRU.PrbTotDl`, `develop` | `RRU.PrbTotDl`, this PR |
|---|---|---|---|---|---|
| eMBB (1, 000001) | 30.0 Mb/s | 512 Mb/s | 30.7 Mb/s | 469 % | 27 % |
| URLLC (2, 000001) | 10.0 Mb/s | 33 954 Mb/s | 10.2 Mb/s | 20 094 500 % | 9 % |
| BE (1, 000002) | 5.0 Mb/s | 34 258 Mb/s | 5.1 Mb/s | 40 324 919 % | 4 % |

`RRU.PrbTotDl` is the median. `develop` is the image `oaisoftwarealliance/oai-gnb:develop` of
2026-11-15; "this PR" is `develop` at `cb98db5b` plus the two commits. KPM counts RLC PDU bytes,
so it reads slightly above iperf3.

**The change:**

- the previous values are kept in a table looked up by `rrc_ue_id`;
- when the table is full, the entry reported the longest time ago is reused;
- for `RRU.PrbTotDl/Ul`, the PRBs of the cell at the previous report are kept per UE, so the UE
  and cell differences cover the same interval (`node_stats[0]` is updated after every
  indication of any subscription);
- second commit: `ue_idx` is not used by any measurement anymore and is removed.

**Not changed:** two subscriptions to the same UE still share the previous values (existing
limitation, comment kept).

## Tests

| Test | Result |
|---|---|
| `e2_ran_func_du_cucp_cuup` and `e2_ran_func_cuup` compile with `-Werror`, after each commit (ran-base container, `-DE2_AGENT=ON -DKPM_VERSION=KPM_V3_00 -DE2AP_VERSION=E2AP_V3`) | passed |
| Full build with OAI's Dockerfiles (`ran-build` with `-Werror`, then `oai-gnb`), image `agentic-ric/oai-gnb:develop-kpmfix` | passed |
| rfsim, 3 slices, stock `develop` gNB: KPM per slice against iperf3 | bug reproduced (table above) |
| rfsim, 3 slices, patched gNB: KPM per slice against iperf3 | passed (table above) |

Not tested: `DRB.PdcpSduVolumeDL/UL` and the uplink measurements (our xApp subscribes to
`DRB.RlcSduDelayDl`, `DRB.UEThpDl` and `RRU.PrbTotDl` only), CU/DU and CU-UP splits, a UE leaving
during a subscription, and more than 16 UEs over time (the reuse of the oldest entry).

To run the two rfsim tests again (needs `sudo modprobe sctp`):

```bash
testbed/upstream_kpm_run.sh oaisoftwarealliance/oai-gnb:develop stock_develop
testbed/upstream_kpm_run.sh agentic-ric/oai-gnb:develop-kpmfix patched_develop
uv run python testbed/upstream_kpm_compare.py testbed/results/upstream_kpm/stock_develop
uv run python testbed/upstream_kpm_compare.py testbed/results/upstream_kpm/patched_develop
```

Raw data: `testbed/results/upstream_kpm/<label>/` (KPM lines, iperf3 JSON, gNB log).

Reference from M07a (OAI v3.0.0, the RNTI-keyed patch, not this branch): stock gNB read about
34 000 Mb/s and 40 million % PRB; patched gNB matched iperf3 to a median 2.5 %.
