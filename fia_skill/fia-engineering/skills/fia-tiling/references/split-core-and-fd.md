# Split-core and Flash Decoding

## Contents

- [Coordinate model](#coordinate-model)
- [Cost model](#cost-model)
- [Assignment](#assignment)
- [FD creation and selection](#fd-creation-and-selection)
- [Safe FD disablement](#safe-fd-disablement)

## Coordinate model

`bN2` enumerates `(batch, KV head)`. Each `bN2` contains one or more query/GQA `M` blocks. Each M row contains a sparse-dependent interval of S2 blocks.

For no mask:

```cpp
s2Start = 0;
s2End = CeilDiv(kvLen, s2BaseSize);
```

For a mask, `CalcS2Range` maps the M block's first/last query token through `preTokenLeftUp` and `nextTokenLeftUp`, clips against KV length and converts tokens to an S2 block interval.

Right-Down causal uses:

```cpp
nextTokenLeftUp = kvLen - qLen;
```

Therefore a KV-length change can change a per-M range even when total `ceil(kvLen/B)` is unchanged.

## Cost model

`CalcCost(basicM, basicS2)` aligns M by 16 and S2 by 64, then combines their weights. `CalcMCache` distinguishes normal/tail M and normal/tail S2. Costs are estimates used to balance ownership; they are not runtime bounds.

`CalcCostInfo` multiplies each batch's BN2 cost by `n2Size`, so every KV head in one batch shares that batch's KV length and cost. Different batches can carry different lengths.

## Assignment

For each proposed core count, `CalcSplitPlan` computes an average remaining cost. It attempts:

```text
AssignByBatch -> AssignByRow -> AssignByBlock -> ForceAssign
```

- `AssignByBatch`: add complete BN2 units while within cost limit.
- `AssignByRow`: add complete `(bN2,M)` rows.
- `AssignByBlock`: add individual S2 blocks only when `streamK=true`.
- `ForceAssign`: guarantee progress only for Stream-K plans.

When `streamK=false`, the per-core cost limit is at least `maxMCost`, so a complete row fits and `AssignByBlock` returns immediately.

## FD creation and selection

Crossing a core boundary inside the same M row increments `curKvSplitPart`. After leaving that row, `RecordFDInfo` records:

- `fdBN2Idx` and `fdMIdx`;
- `fdS2SplitNum`;
- `fdWorkspaceIdx`;
- M rows participating in the reduction.

After finding the best Stream-K plan, `SplitCore` constructs a no-FD plan with the same used-core count. `CheckChooseWithFd` selects FD only when its maximum cost wins by more than the configured tolerance. In A5 nonquant GQA, `streamK=true`, `fdTolerance=9`, and `fdLeastBlock=0` are currently set by `CreateSplitInput`.

FD tends to win for short Q, long KV and insufficient `(batch * kvHeads * MBlocks)` parallelism.

## Safe FD disablement

Do:

```cpp
splitParam.streamK = false;
SplitCore(aicNum, baseInfo, splitParam, result);
CHECK(result.fdRes.fdNum == 0);
flashDecodeFlag = false;
```

Do not set only `flashDecodeFlag=false`. That can leave an S2-split row whose local softmax results are never combined.

With exact Q topology and uncertain KV only, no-FD converts an upper/real mismatch from a producer/consumer topology mismatch into a cost-estimation mismatch. It can reduce balance and utilization but does not advertise missing FD partials.
