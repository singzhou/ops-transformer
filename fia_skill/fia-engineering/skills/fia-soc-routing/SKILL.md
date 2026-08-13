---
name: fia-soc-routing
description: Map FusedInferAttentionScore implementation and build/runtime routing across Ascend 910B, Ascend 910C (ascend910_93), and Ascend 950. Use when identifying FIA arch22/arch35 sources, NpuArch template registration, kernel entry, tiling implementation, compiler flags, feature differences, or files that must change for multi-SoC support.
---

# FIA SoC Routing

Separate the build arch directory, runtime NPU arch, registered tiling template and kernel tiling-key dispatch. Verify all four.

## Use the canonical build mapping

From `/opt/zsy/ops-transformer/CMakeLists.txt`:

| Product | Build name | Arch directory | Runtime tiling arch |
|---|---|---|---|
| Ascend 910B | `ascend910b` | `arch22` | `NpuArch::DAV_2201` |
| Ascend 910C | `ascend910_93` | `arch22` | `NpuArch::DAV_2201` |
| Ascend 950 | `ascend950` | `arch35` | `NpuArch::DAV_3510` |

910B and 910C share the FIA `arch22` code family in this tree. Do not invent a separate 910C FIA directory unless source evidence changes.

Architecture budget reference: DAV_2201 has 512 KiB L1, 64 KiB L0A/L0B, 128 KiB L0C and 192 KiB UB; DAV_3510 has 512 KiB L1, 64 KiB L0A/L0B, 256 KiB L0C and 248 KiB UB. Treat these as architecture capacities, not SKU core-count/L2 facts. Query core count and L2 at runtime rather than hardcoding them. Reconfirm current values in `/opt/zsy/cannbot-skills/ops/npu-arch/SKILL.md` before design decisions.

## Route 910B/910C

- Host specialized templates: `op_host/arch22/fia_tiling_nonquant.cpp`, `fia_tiling_nonquant_mla.cpp`, empty tensor template and arch22 checkers/legacy tiling.
- Kernel entry: `op_kernel/fused_infer_attention_score.cpp`.
- Kernel implementation/dispatch: `op_kernel/arch22/fused_infer_attention_score_v3.cpp`, `flash_attention_interface.cpp`, plus arch22 headers and reused IFA/PFA code.
- Specialized host registration currently targets `DAV_2201` with priorities such as MLA 9 and nonquant 29.

Check product macros and generated tiling keys before assuming 910B and 910C execute identical specializations for every feature.

## Route 950

- Host specialized templates: `op_host/arch35/fia_tiling_nonquant_gqa.cpp`, `fia_tiling_fullquant_gqa.cpp`, `fia_tiling_fullquant_mx.cpp`.
- Host fallback: `op_host/arch35/fused_infer_attention_score_tiling_impl.cpp` at priority 999.
- Kernel entry: `op_kernel/fused_infer_attention_score_apt.cpp`.
- Kernel implementation: `op_kernel/arch35/` GQA/MLA/fullquant/cube/vector/FD headers and template dispatch.
- Registration targets `DAV_3510`; nonquant GQA priority is 28, fullquant paths 210/211, fallback 999.
- The op-host CMake adds Ascend950-specific compiler options under `CONDITION_UNIT=ascend950`.

## Handle cross-arch host compilation

`op_host/CMakeLists.txt` lists arch22, arch35 and arch38 host sources together. Runtime registry filters by `GetCurNpuArch`, then tries templates in ascending priority. Therefore source inclusion does not mean a template runs on every SoC.

## Make a SoC-safe change

1. Identify which registered template accepts the target case.
2. Identify its tiling-data type and kernel tiling key.
3. Check whether a shared parser/checker/common splitter affects other SoCs.
4. For shared API/opdef changes, update all kernel entries even if only one SoC enables the feature.
5. Return `GRAPH_PARAM_INVALID` for intentionally unsupported templates so fallback routing remains valid.
6. Add per-SoC build and execution tests.

Simulator is not cross-SoC evidence: the current `/opt/zsy/cannbot-skills/ops/ops-simulator/SKILL.md` flow supports Ascend 950 only. 910B/910C require build checks and real-hardware execution for kernel behavior.

Read [references/soc-file-map.md](references/soc-file-map.md) for a fuller file matrix and verification commands.
