---
name: fia-engineering
description: White-box engineering guide for Ascend FusedInferAttentionScore (FIA) across ops-transformer and vLLM Ascend. Use for tracing FIA ACLNN/opdef/host-tiling/kernel execution, mapping Ascend 910B/910C/950 code paths, understanding GQA/MLA/quantization/Flash Decoding/paged attention/actual sequence lengths, integrating FIA PTA and custom OPP into vLLM Ascend, reviewing changes, or designing correctness and performance tests.
---

# FIA Engineering

Treat the checked-out source as authoritative. Use this skill to locate the right subsystem, load only its detailed child skill, trace host metadata through the tiling key into the kernel, and validate changes on every affected SoC.

## Establish scope

1. Read `/opt/zsy/vllm-ascend/AGENTS.md` before modifying vLLM Ascend.
2. Confirm the available repositories. Default source roots are:
   - `/opt/zsy/ops-transformer`: current FIA operator reference.
   - `/opt/zsy/vllm-ascend`: integration and prospective in-tree custom OPP.
   - `/opt/zsy/op-plugin`: legacy PTA reference only; do not require it when an in-tree PTA is requested.
3. Record SoC, layout, dtype/quant mode, attention mode, KV storage mode, sparse mode, Q/KV length representation, ACLGraph use, and whether the request is analysis or implementation.
4. Restrict the first proof to the requested path. Do not generalize a GQA proof to MLA, quantized, prefix, padding, or other SoCs.

## Route to child skills

Read the selected child `SKILL.md` completely before acting. Child skills are standalone and can be invoked directly by another agent.

- Host parsing, template selection, tiling key, workspace, split-core, S1-out split, Stream-K, or Flash Decoding: [fia-tiling](skills/fia-tiling/SKILL.md).
- Kernel entry, AIC/AIV pipeline, actual length parsing, masks, offsets, paged KV, softmax, or FD reduction: [fia-kernel](skills/fia-kernel/SKILL.md).
- Ascend 910B, 910C/910_93, 950, `arch22`, `arch35`, build dispatch, or runtime template routing: [fia-soc-routing](skills/fia-soc-routing/SKILL.md).
- ACLNN API, opdef, custom OPP, PTA, vLLM Ascend build/package, torch registration, or ACLGraph: [fia-integration](skills/fia-integration/SKILL.md).
- Unit/system tests, differential accuracy, boundary cases, metadata assertions, NPU profiling, or regression review: [fia-validation](skills/fia-validation/SKILL.md).
- Runtime errors, crashes/hangs, precision failures, stale binaries, pipeline synchronization, or illegal memory access: [fia-debugging](skills/fia-debugging/SKILL.md).
- Tiling/performance modeling, Simulator traces, on-board profiling, FD/no-FD comparison, load balance, or iterative optimization: [fia-performance](skills/fia-performance/SKILL.md).

For changes spanning subsystems, read child skills in execution order: `fia-integration` → `fia-tiling` → `fia-kernel` → `fia-soc-routing` → `fia-validation`. Route failures to `fia-debugging`; enter `fia-performance` only after correctness and protocol checks pass.

## Apply the white-box workflow

1. Trace the public API and opdef input index; never infer an input's host/device semantics from its name.
2. Trace `DoOpTilingFusedInferAttentionScore` through parser, checker, registered template and `IsCapable`/`DoTiling`.
3. Decode the tiling key and identify the exact kernel entry/template specialization.
4. Map every modified tiling field to all kernel consumers before editing it.
5. Separate task topology from task bounds:
   - topology: `(batch, kv-head, M-block, S2-block)` ownership, core boundaries, FD count and workspace indices;
   - bounds: real Q/KV lengths, sparse valid range, tail size and padding.
6. Prove host/kernel consistency. A device-side real length may safely shrink work only when it cannot delete a host-promised producer, FD slot, output owner, or synchronization participant.
7. Validate source closure and SoC routing; directory names alone do not prove runtime selection.
8. Add a reference comparison plus boundary-focused tests and run available static/build checks.

## Preserve critical invariants

- Keep exact Q length whenever it determines M-block topology.
- Treat host-visible actual-length inputs registered with `TilingInputsDataDependency` as synchronization-sensitive.
- Do not disable FD by changing only the tiling key. Disable S2 cross-core splitting at its source and assert that no FD metadata was generated.
- Do not let the kernel expand beyond host-owned task boundaries. Device lengths may clamp work; they may not invent ownership.
- Keep tiling-key encoding, tiling-data layout and kernel decoding atomic across a change.
- Do not hand-copy generated `aclnnInner_*.h` files.
- Mark planned behavior as planned. The runtime NPU KV-length V6 path documented in this skill is not assumed to exist until verified in the current tree.

## Use bundled navigation

Run `scripts/fia_nav.sh <topic> [ops-root] [vllm-root]` for compact source entry points. Supported topics are `soc`, `tiling`, `split`, `kernel`, `api`, `build`, `tests`, and `runtime-seqlen`.

Read [references/system-map.md](references/system-map.md) for the end-to-end map and terminology. Read [references/change-review.md](references/change-review.md) before reviewing or implementing a cross-layer change.
Read [references/cannbot-routing.md](references/cannbot-routing.md) to reuse the local CANN engineering skills without duplicating their workflows. Before invoking a routed skill, read that skill's `SKILL.md` completely and obey its mandatory ordering and confirmation gates.
