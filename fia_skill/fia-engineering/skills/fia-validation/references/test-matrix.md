# FIA validation matrix

## Correctness cases

- Q exact, KV exact baseline.
- Host upper equals device real.
- Host upper greater by `1,5,7,15,16`.
- `real=0` where supported.
- mixed-batch lengths and cumulative TND lengths.
- contiguous and paged KV; PA block boundary crossings.
- no mask and Right-Down causal; other supported sparse modes.
- FD selected by default, rejected by cost comparison and forcibly disabled.
- output LSE enabled/disabled.

## Topology assertions

- Every legal `(bN2,M)` row has exactly one final-output owner.
- No-FD: no row is split in S2 across cores and `fdNum==0`.
- FD: every `fdS2SplitNum` matches the number of produced partial slots.
- Workspace indices are in bounds, nonoverlapping where required and initialized before consumption.
- Real device length never expands beyond host-owned S2 interval.
- Query topology is unchanged when only KV varies.

## Boundary generators

Let `B` be the selected S2 base, normally 128 or 256 in the A5 GQA template. Generate:

```text
real = kB + r
upper = real + delta
r in {-1,0,1, B-1 where valid}
delta in {0,1,5,7,15,16}
```

Also vary Q/GQA so `qLen*gSize` falls around the selected M base.

## ACLGraph

- Allocate fixed-shape length buffers before capture.
- Capture with maximum legal workspace requirements.
- Replay at several values without reallocating tensors.
- Verify no host synchronization in the hot path using profiling/tracing.
- Compare each replay to an eager exact-length baseline.

## Failure-mode tests

- wrong device, dtype, rank, length count and noncontiguous runtime tensor;
- real length greater than upper;
- upper-real greater than declared bound;
- unsupported quant/layout/prefix/padding combination;
- missing custom ACLNN symbols/package;
- template refuses and fallback behavior is explicit.
