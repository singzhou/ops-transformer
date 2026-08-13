---
name: fia-tiling
description: White-box guide to FusedInferAttentionScore host tiling. Use for FIA input parsing and validation, template registry and IsCapable routing, tiling-key generation, workspace/block-dim calculation, GQA split-core cost and ownership, S1-out split, Stream-K, Flash Decoding selection, sparse S2 ranges, or host/device sequence-length consistency.
---

# FIA Tiling

Trace values from the tiling context to every kernel-consumed metadata field. Treat split topology as a correctness protocol, not merely a performance choice.

## Follow the host chain

1. Read `op_host/fused_infer_attention_score_tiling_register.cpp` for `TilingInputsDataDependency` and the common entry.
2. Read `fused_infer_attention_score_tiling_info_parser.cpp` and `fia_tiling_info.h` for normalized shapes, layouts, flags and actual lengths.
3. Read `attention/common/op_host/fia_tiling_templates_registry.h` for SoC routing and ascending priority order.
4. Locate all `REGISTER_TILING_TEMPLATE_FIA` registrations for the target NPU arch.
5. Read each earlier-priority template's `IsCapable`; do not jump directly to the expected template.
6. Trace `DoOpTiling` through split policy, tiling-data fill, tiling-key generation, workspace sizing and block dimension.

Paths are relative to `/opt/zsy/ops-transformer/attention/fused_infer_attention_score` unless stated otherwise.

## Analyze GQA split-core

Read `/opt/zsy/ops-transformer/attention/common/op_host/split_core_v2.{h,cpp}` completely when changing core ownership.

The V2 splitter:

1. Resolves per-batch Q and KV sizes with `GetS1SeqSize`/`GetS2SeqSize`.
2. Builds `mBaseNum`, `s2BaseNum`, tail sizes and sparse S2 ranges.
3. Computes a cost for each complete `(bN2, M)` row.
4. Tries a range of used core counts.
5. Assigns in this order: complete BN2, complete M row, S2 block, forced block.
6. Records FD metadata only when one M row spans multiple cores.
7. Compares FD and no-FD plans with `CheckChooseWithFd`.

Read [references/split-core-and-fd.md](references/split-core-and-fd.md) for exact invariants and pseudocode.

## Build a tiling design proof

Use [references/design-checklist.md](references/design-checklist.md) to cover the FlashAttention design nodes: macro shape, parallel strategy, tile/Roofline model, online softmax, platform resources, multicore balance, buffer budget, compile-time specialization, host tiling, pipeline/synchronization, baseline and optimization closure. Flash Decoding is a split-KV parallel strategy, not a separate FIA family.

For new or redesigned tiling, route to `/opt/zsy/cannbot-skills/ops/ascendc-tiling-design/SKILL.md`. Do not hardcode core counts or L2 sizes; route platform facts through `/opt/zsy/cannbot-skills/ops/npu-arch/SKILL.md` and query runtime platform information where available.

## Audit actual lengths

Current host tiling obtains values through `gert::Tensor::GetData<int64_t>()` in specialized templates. Kernel code independently parses GM actual lengths. If a new device-only real KV length is added:

- keep an exact host Q length for M topology;
- keep a host KV upper bound for maximum allocation;
- do not register the device-only real KV tensor as a host value dependency;
- ensure all topology generated from the upper bound remains valid for every legal real length;
- disable unsafe topology features or prove interval stability.

For the GQA optimistic-KV design, read [references/runtime-kv-safety.md](references/runtime-kv-safety.md).

## Change tiling safely

- Update `IsCapable` so unsupported cases return `GRAPH_PARAM_INVALID`, allowing the next template to try.
- Preserve template priority semantics; smaller numeric priority wins.
- Update tiling data, host fill, kernel read and key encoding in one change.
- Size workspace for the largest possible runtime path.
- Log or assert ownership metadata in debug/test builds.
- When disabling FD, set `streamK=false` before `SplitCore`, reject S1-out split when it depends on uncertain KV topology, assert `fdRes.fdNum==0`, and emit `isFd=false`.

## Verify

Use `$fia-validation`. At minimum, inspect metadata for varying batch lengths, `S2=0/1`, block tails, exact block boundaries, Right-Down causal, paged KV, FD chosen/not chosen, and all target SoCs.
