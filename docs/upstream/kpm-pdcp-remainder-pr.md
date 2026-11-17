# Upstream pull request: carry the remainder of DRB.PdcpSduVolumeDL/UL (draft)

Status on 2026-11-17: **not opened.** The change is written, uncommitted, on branch `kpm-pdcp-volume-remainder` in `var/build/oai-upstream` (git-ignored clone), stacked on `kpm-per-ue-prev-values` (PR duranta-project/openairinterface5g#1124), which this change needs: it uses #1124's per-UE table.

## What is left before opening

1. **Your decision on the trailer.** OAI's CONTRIBUTING.md asks AI-assisted commits to carry `Assisted-by: <agent>:<model>`. The project's CLAUDE.md forbids Claude attribution lines in commits, so Claude can't write this commit until you say which rule applies upstream. #1124's commits carry `Assisted-by: Claude:claude-fable-5-2`.
2. **Commit, push and open the PR:**
   - `git commit -s` in `var/build/oai-upstream`, with the message below and the trailer you choose;
   - `git push fork kpm-pdcp-volume-remainder`;
   - open the PR against `develop`, saying it depends on #1124. GitHub will show #1124's two commits too until #1124 merges.

## Commit message

```
E2 KPM: carry the remainder of DRB.PdcpSduVolumeDL/UL to the next report

DRB.PdcpSduVolumeDL/UL are reported in whole Mb. The previous code took
(counter - previous) * 8 / 1000000 and then set previous = counter, so
the bytes under 1 Mb were dropped at every report: the measurement reads
about 0.5 Mb per period low on average (over 100 s at 8.4 Mb/s, 800 Mb
reported instead of 840).

Advance the previous counter by exactly the bytes reported instead, so
the remainder is carried to the next report and the reports add up to
the real volume. Each report still has 1 Mb resolution, as the unit
requires. The computation no longer multiplies the difference by 8, so
it cannot overflow 32 bits, and a counter that wraps still gives the
right difference.
```

## PR title

E2 KPM: carry the remainder of DRB.PdcpSduVolumeDL/UL to the next report

## PR text

Depends on #1124 (it uses that PR's per-UE table of previous counters); only the last commit is new.

`DRB.PdcpSduVolumeDL/UL` are reported in whole Mb as `(counter - previous) * 8 / 1000000`, and `previous` is then set to the counter. The bytes under 1 Mb are dropped at every report, so the measurement is biased low by about 0.5 Mb per period on average. At a 1 s period that is 0.5 Mb/s: 6 % of an 8 Mb/s flow.

**The change:** a small helper reports the whole Mb in `counter - previous` and advances `previous` by exactly those bytes, so the remainder is carried to the next report. The reports then add up to the real volume. Each report keeps its 1 Mb resolution and integer type, so consumers are unchanged. The difference is no longer multiplied by 8 before dividing, so it can't overflow 32 bits. Unsigned arithmetic keeps a counter wrap correct.

## Tests

| Test | Result |
|---|---|
| Standalone C test of the helper, extracted from the patched file: 100 s at 8.4 Mb/s, the same across a uint32 wrap, 1000 s at 39.5 Mb/s; compared with the old arithmetic | carried: 840 / 840 / 39 500 Mb, exactly the volume sent; old: 800 / 800 / 39 000 Mb |
| The same function in an OAI v3.0.0 gNB (our testbed patch), rfsim, 3 slices, iperf3 UDP downlink: steady 30 / 10 / 5 Mb/s, steps, and an overload (30 Mb/s offered into 5 PRBs) | the volume matched the iperf3 rate within 1 Mb/s, about 0.5 Mb/s above it (IP/UDP headers are in the PDCP SDU); in the overload it read 31 Mb/s while 4 Mb/s were served |
| `e2_ran_func_du_cucp_cuup` and `e2_ran_func_cuup` compile with `-Werror` (`ran-build` image of `develop` + #1124, `-DE2_AGENT=ON -DKPM_VERSION=KPM_V3_00 -DE2AP_VERSION=E2AP_V3`, this commit's file swapped in) | passed |

Not tested on `develop`: a full gNB build and an rfsim run with this commit (tested on v3.0.0 with the same function, see above); the uplink measurement.
