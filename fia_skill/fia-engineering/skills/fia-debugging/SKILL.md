---
name: fia-debugging
description: Diagnose FusedInferAttentionScore precision failures, ACLNN/runtime errors, kernel lookup or tiling failures, hangs, AIC/AIV crashes, illegal memory access, stale custom OPP binaries, and optimistic-versus-real sequence-length regressions.
---

# FIA Debugging

Classify the symptom before changing code. Preserve one fixed reproducer, exact input tensors, selected tiling key/data, SoC/CANN version and exact-host baseline.

## Route by symptom

- Wrong output, NaN/Inf, zeros or random values: read `/opt/zsy/cannbot-skills/ops/ascendc-precision-debug/SKILL.md` completely and follow its mandatory prechecks.
- ACLNN error, 161xxx/361xxx/561xxx, tiling failure, kernel not found or plog diagnosis: read `/opt/zsy/cannbot-skills/ops/ascendc-runtime-debug/SKILL.md`.
- Hang, timeout, segmentation fault, AIC error, intermittent corruption or suspected out-of-bounds access: read `/opt/zsy/cannbot-skills/ops/ascendc-crash-debug/SKILL.md`.
- Uncertain CANN/custom OPP/device environment: read `/opt/zsy/cannbot-skills/ops/ascendc-env-check/SKILL.md` first.

Do not merge these workflows. Start with the primary symptom, then cross-route only when evidence changes the classification.

## Apply FIA-specific isolation

1. Confirm the loaded public ACLNN V-version, custom `libcust_opapi.so`, tiling template, tiling key and kernel binary are from the edited build.
2. Compare exact-host/exact-kernel length with upper-host/real-device length using identical Q/K/V and metadata.
3. Force no-FD. If the failure disappears, audit split ownership, partial-slot writes and FD merge rather than numerical tolerances.
4. If it remains, isolate CopyIn/addressing → MM1 → mask/online softmax → MM2 → CopyOut using staged dumps or the routed precision workflow.
5. For hangs, audit AIC/AIV event counts, cross-core flag producer/consumer pairs, empty runtime tasks and FD drain paths before broad changes.
6. For illegal accesses, verify host-owned interval intersection, PA block-table index, tails, workspace index/count and real `<=` upper validation.

Read [references/symptom-matrix.md](references/symptom-matrix.md) for likely FIA root causes. Limit speculative fixes; after repeated non-progress, switch to path bisection as required by the precision-debug workflow.
