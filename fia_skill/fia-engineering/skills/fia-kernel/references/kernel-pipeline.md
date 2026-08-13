# A5 GQA kernel pipeline

## Contents

- [Entry and specialization](#entry-and-specialization)
- [Initialization](#initialization)
- [Task creation](#task-creation)
- [Compute pipeline](#compute-pipeline)
- [Flash Decoding](#flash-decoding)

## Entry and specialization

The APT entry includes arch35 template dispatch and eventually instantiates GQA classes such as `FiaKernelNoQuantGqa` with cube, vector and FD block implementations. The tiling key determines layout, aligned S1/S2/D/DV templates, mask, PA layout and `isFd` at compile time.

## Initialization

`FiaKernelNoQuantGqa::Init` loads base/FA/FD metadata, initializes Q and KV actual-sequence parsers, initializes cube/vector blocks and allocates L1/UB workspace. Actual lengths are obtained per batch, while host metadata supplies the core's `(bN2,M,S2)` start/end ownership.

## Task creation

`GetTaskDealMode` caches actual lengths when entering a batch, derives real S2 loop counts and real M counts, and recomputes sparse or non-sparse S2 range on each row.

Possible outcomes include create task, not started, S2 end, skip, zero-length handling and S1-out filtering. The task loop must make progress for all outcomes without violating paired-core synchronization.

## Compute pipeline

For each valid task:

```text
AIC ComputeMm1: Q * K^T
AIV ComputeVec1: scale + mask/PSE + softmax/update
AIC ComputeMm2: probabilities * V
AIV ComputeVec2: update/normalize and direct output or FD partial output
```

Preload/ping-pong logic overlaps stages. Cube and vector block headers own detailed GM-to-L1/L0/UB copies and output addressing.

## Flash Decoding

With `isFd=true`, vector output writes partial `accumOut`, `lseMax` and `lseSum` to workspace. `FiaBlockVecFlashDecode` loads the number of parts specified by host `s2SplitNumOfFdHead`, applies stable softmax recombination and writes final output/LSE.

FD consumer iteration is metadata-driven, not dynamically derived from real KV block count. This is the central reason host upper-bound and kernel real-length topology must agree.
