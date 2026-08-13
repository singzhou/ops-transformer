# FIA performance playbook

## Capture with every result

- SoC/SKU, CANN/operator commits, layout/dtype, batch, Q/KV heads and D.
- Q length, real KV, host upper KV and delta.
- template, tiling key, `mBaseSize`, `s2BaseSize`, blockDim/used cores.
- Stream-K/S1-out/FD flags, `fdNum`, per-row split count and workspace size.
- warm-up, repetitions, median/p90 kernel time and end-to-end decode time.

## Diagnosis order

1. Validate profiler core coverage; one sampled core does not prove single-core execution or balance.
2. Check max/mean per-core time and task-weight imbalance before micro-optimizing a core.
3. Identify dominant AIC/AIV pipeline and Cube/Vector/MTE overlap.
4. Check GM/L1/L0/UB traffic and redundant round trips.
5. Inspect tail frequency, PA gather granularity, scalar address overhead and softmax/MM overlap.
6. Compare FD parallel gain against partial-write, workspace-read, reduction and synchronization cost.

## Stop conditions

- Any numerical, metadata, hang or workspace-poison regression stops optimization.
- Any win limited to one delta/shape must not become the default without a selection rule and negative cases.
- A no-FD safety path remains available until FD stability is proven for every legal real length in the declared upper-bound interval.
