# CANNbot skill routing for FIA

Use this table as an adapter layer. FIA source and protocol knowledge stays in this package; generic AscendC workflows stay under `/opt/zsy/cannbot-skills/ops`.

| FIA task | Route | FIA-specific addition |
|---|---|---|
| New tiling or split policy | `ascendc-tiling-design/SKILL.md` | Prove GQA ownership, FD producer/consumer count and optimistic/real KV interval safety. |
| Platform capacity or product mapping | `npu-arch/SKILL.md` | Map 910B/910_93 to arch22/DAV_2201 and 950 to arch35/DAV_3510 in the checked-out tree. |
| AscendC API or pipeline change | `ascendc-api-best-practices/SKILL.md` | Audit FIA mixed AIC/AIV events, PA tails and FD workspace accesses. |
| DAV_3510 RegBase change | `ascendc-regbase-best-practice/SKILL.md` | Keep arch22 compatibility separate and verify actual FIA dispatch. |
| White-box test design | `ascendc-whitebox-design/SKILL.md` | Model each `(batch, kv-head, M-row, S2-range)` task and FD slot. |
| UT/ST design | `ascendc-ut-develop/SKILL.md`, `ascendc-st-design/SKILL.md` | Add metadata assertions and paired exact-host versus upper-host/real-device cases. |
| Precision failure | `ascendc-precision-debug/SKILL.md` | Compare exact-length baseline first; then isolate CopyIn, MM1, softmax, MM2 and FD merge. |
| ACLNN/tiling/kernel lookup failure | `ascendc-runtime-debug/SKILL.md` | Confirm public V-version, custom OPP install, tiling key and kernel binary closure. |
| Hang, AIC error or memory corruption | `ascendc-crash-debug/SKILL.md` | Audit mixed-core flags and missing FD producers before generic memcheck. |
| Simulator | `ops-simulator/SKILL.md` | Ascend 950 only; never claim 910B/910C coverage from it. |
| On-board profiling | `ops-profiling/SKILL.md` | Compare exact, optimistic no-FD and auto-FD with identical inputs. |
| Iterative optimization | `ascendc-perf-optimize/SKILL.md` | FIA is non-communication: skip card-level Step 2; run inter-core Step 3 only when synchronization is present; always run single-core Step 4. |
| Precision acceptance | `ops-precision-standard/SKILL.md` | Preserve FIA plan/operator thresholds when stricter. |

Before invoking any route, read its `SKILL.md` completely. Relative references and scripts in that skill resolve from its own directory. Do not copy destructive cleanup commands blindly; scope them to generated build/cache artifacts after resolving exact paths.
