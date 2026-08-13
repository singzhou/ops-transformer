---
name: fia-performance
description: Optimize FusedInferAttentionScore performance through tiling modeling, Ascend 950 Simulator analysis, real-NPU profiling, multicore load balance, Cube/Vector/memory pipeline diagnosis, FD versus no-FD comparison, and tiling feedback while preserving correctness.
---

# FIA Performance

Enter this workflow only after numerical and host/kernel protocol validation passes. Keep exact and optimistic-length correctness baselines in every performance run.

## Follow the optimization loop

1. Use `$fia-tiling` and `/opt/zsy/cannbot-skills/ops/ascendc-tiling-design/SKILL.md` to record theoretical tiles, core ownership, buffer budget and expected bottleneck.
2. For Ascend 950, use `/opt/zsy/cannbot-skills/ops/ops-simulator/SKILL.md`: inspect `summary.json` first, then trace only when causal detail is needed. Simulator does not cover 910B/910C.
3. On each target SoC, use `/opt/zsy/cannbot-skills/ops/ops-profiling/SKILL.md` for warm, repeated hardware measurements and pipeline counters.
4. Use `/opt/zsy/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md` for the layered analysis. FIA is not a communication operator, so skip card-level Step 2. Run inter-core Step 3 when FD or mixed-core synchronization is involved; always run single-core Step 4.
5. Feed observed imbalance/bound/overlap back into tiling, then repeat correctness and performance validation.

Do not route FlashAttention to `ascendc-performance-best-practices` expecting a ready implementation: its NN/FlashAttention family is currently marked planned. Its common DataCopy/tail/UB practices may still be consulted after verifying applicability.

## Compare the right variants

For each representative GQA case measure:

- exact host length baseline;
- optimistic host plus real device length with forced no-FD;
- default auto-FD when proven safe;
- any interval-stable FD proposal.

Keep inputs, warm-up, repetitions, output options and hardware clocks identical. Separate kernel time from end-to-end effects and record the selected tiling metadata.

Read [references/performance-playbook.md](references/performance-playbook.md) for FIA metrics and stop conditions.
