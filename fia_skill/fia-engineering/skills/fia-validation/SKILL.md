---
name: fia-validation
description: Validate FusedInferAttentionScore changes with white-box and NPU tests. Use for FIA host-tiling or kernel regression design, SoC matrices, reference comparisons, sequence-length boundary tests, FD/Stream-K metadata checks, paged-attention and mask cases, ACLGraph capture/replay, workspace poisoning, performance profiling, or review acceptance criteria.
---

# FIA Validation

Validate both numerical output and the host/kernel protocol. A passing output sample does not prove all advertised tasks or workspace slots are valid.

## Build a scoped matrix

Record dimensions before testing:

- SoC: 910B, 910C/910_93, 950.
- path: specialized template and fallback.
- GQA/MLA, layout, dtype/quant mode.
- contiguous/list/paged KV and block size.
- sparse mode/mask, prefix, padding, rope, LSE.
- Q length, KV length, batch and KV-head count.
- FD chosen, FD rejected, forced no-FD and S1-out split.
- eager and ACLGraph replay.

Limit the first matrix to the declared feature scope, then add negative gates for unsupported combinations.

## Establish references

For runtime real-KV work, compare:

1. baseline: host and kernel both receive exact real lengths;
2. candidate: host receives upper lengths and kernel receives device real lengths;
3. mathematical PyTorch reference where feasible.

Use identical Q/K/V, block tables, masks and output dtype. Report max absolute/relative difference and failing coordinates.

Select acceptance thresholds from `/opt/zsy/cannbot-skills/ops/ops-precision-standard/SKILL.md`; plan-specific thresholds override generic defaults. Do not silently reuse one tolerance across FP16, BF16 and FP32.

## Attack topology boundaries

For S2 base `B`, include lengths around every boundary:

```text
B-1, B, B+1, 2B-1, 2B, 2B+1
```

For `deltaMax=16`, test delta `0,1,5,7,15,16`, especially `upper=kB+1` and `real<=kB`. Include multiple batches whose lengths cross different boundaries.

Read [references/test-matrix.md](references/test-matrix.md) for mandatory assertions.

## Inspect metadata

- dump or expose used core count and each core's BN2/M/S2 boundaries;
- inspect `fdNum`, `fdS2SplitNum`, FD workspace indices and FD vector assignment;
- assert no-FD runtime mode has `streamK=false`, `fdNum=0` and non-FD tiling key;
- assert task coverage has no gaps or overlaps at complete-row granularity;
- poison FD workspace with nonzero/NaN patterns before execution where test infrastructure allows it.

## Check performance separately

Measure decode latency/throughput and NPU utilization for short-Q/long-KV, low batch and varying KV heads. Compare default auto-FD, forced no-FD and any interval-stable FD recovery. Do not trade an unproven topology for performance.

## Run proportionate checks

- host parser/split changes: host UT plus NPU accuracy.
- kernel changes: NPU execution on every affected SoC.
- PTA/schema: Meta/compile/eager tests.
- ACLGraph changes: capture once, replay varied in-place length values and compare to eager.
- build/package changes: inspect wheel/install tree and symbol resolution.

Record hardware, CANN, torch/torch_npu, operator commit and exact commands with results.

## Route specialized test generation

- White-box path/task-contract design: `/opt/zsy/cannbot-skills/ops/ascendc-whitebox-design/SKILL.md`.
- UT coverage: `/opt/zsy/cannbot-skills/ops/ascendc-ut-develop/SKILL.md`; for this vLLM Ascend custom project use its `repo_type=custom` route and note that its custom path covers `opapi` and `ophost`, not `opkernel`.
- ACLNN system-test factor and L0/L1/L2 generation: `/opt/zsy/cannbot-skills/ops/ascendc-st-design/SKILL.md`.
- Source review: `/opt/zsy/cannbot-skills/ops/ascendc-code-review/SKILL.md`.

Each routed skill has its own mandatory workflow. Read its `SKILL.md` completely before use; do not bypass questionnaires, TODO gates, confirmation points or required execution order.
