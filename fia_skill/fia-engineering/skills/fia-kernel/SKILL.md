---
name: fia-kernel
description: White-box guide to FusedInferAttentionScore AscendC kernels. Use for FIA kernel-entry and tiling-key dispatch, AIC/AIV mixed-core pipelines, MM1-softmax-MM2 execution, GQA addressing, paged KV, masks and sparse ranges, actual Q/KV length parsing, task ownership, tail handling, Flash Decoding partial writes and reduction, or kernel correctness debugging.
---

# FIA Kernel

Identify the exact SoC and tiling key first. Never reason from an arbitrary same-named kernel class.

## Trace dispatch

1. Use `$fia-soc-routing` to select the build/kernel entry.
2. Start at `op_kernel/fused_infer_attention_score.cpp` or `fused_infer_attention_score_apt.cpp`.
3. Find the matching tiling-key macro or template dispatcher.
4. Record template parameters: layout, input/output type, GQA/MLA, quant mode, mask, PA layout, FD, prefix/padding and head dimensions.
5. Follow the instantiated kernel class and cube/vector/FD block types.

Read [references/kernel-pipeline.md](references/kernel-pipeline.md) for the A5 GQA pipeline and [references/memory-and-offsets.md](references/memory-and-offsets.md) before changing addresses or lengths.

Before introducing or replacing AscendC APIs, read `/opt/zsy/cannbot-skills/ops/ascendc-api-best-practices/SKILL.md` and verify the exact target-architecture API signature. In particular, audit GM↔UB alignment, `DataCopyPad` tails, repeat limits, queue/event pairing, buffer lifetime and production use of `GlobalTensor::GetValue/SetValue`. For DAV_3510 RegBase code, additionally route to `/opt/zsy/cannbot-skills/ops/ascendc-regbase-best-practice/SKILL.md`; do not assume a RegBase optimization applies to arch22.

## Audit task enumeration

For A5 nonquant GQA, inspect `op_kernel/arch35/fia_kernel_noquant_gqa.h`:

- `Init`: tiling metadata, GM tensors, actual-length parsers and buffers.
- task loop and `GetTaskDealMode`: host range traversal plus device-side validity.
- `CalcCurS2StartEndNoSparse/WithSparse`: real device KV range and host core boundary intersection.
- `ExecuteTask`: AIC MM1/MM2 and AIV Vec1/Vec2 pipeline.
- `Process`: cross-core setup, FA phase and optional FD phase.

Prove these properties:

- actual Q/KV are cached for the correct `bIdx`;
- device bounds only shrink the current host-owned region;
- `s2End-s2Start`, tail sizes and offsets cannot underflow;
- all AIC/AIV partners execute compatible synchronization;
- empty Q/KV rows write required outputs;
- FD metadata never asks for an unwritten partial.

## Change actual-length behavior

Do not merely replace the pointer passed into an existing parser. Determine whether the same input was used by host tiling. If host metadata was built from an upper bound, clamp kernel work with real length while retaining host ownership boundaries.

For no mask, expected structure is conceptually:

```cpp
realEnd = CeilDiv(realKv, s2BaseSize);
curStart = max(realStart, hostStart);
curEnd = min(realEnd, hostEndIfPresent);
```

For sparse modes, recompute token-space validity using real Q/KV, then intersect the resulting block interval with host ownership.

## Review FD

FD has two contracts:

1. FA producers write partial accumulated output, LSE max and LSE sum to the slot determined by host metadata.
2. FD consumers load exactly `fdS2SplitNum` parts from the workspace and combine them.

Any runtime skip that removes a producer invalidates the contract unless it writes a mathematically and operationally valid identity slot.

## Verify

Use `$fia-validation`. Inspect generated tiling data and workspace initialization in addition to numerical output. Run on hardware for changes involving mixed-core events, GM offsets, PA or FD.
