# Runtime real KV with a host upper bound

## Scope

This is a design rule for GQA where Q length is exact, host KV is an upper bound, device KV is real, and `0 <= upper-real <= deltaMax`. Verify implementation status before applying it.

## First safe version

Use host upper length to allocate and estimate. Use the device real length only inside the kernel to clamp S2 work. In this mode:

```cpp
splitParam.streamK = false;
enableS1OutSplit = false;
```

Assert no FD metadata. Keep each complete `(bN2,M)` row on one core. This makes the uncertain KV tail affect work within an owner, not ownership itself.

## Why a small delta is insufficient

For `B=128`, `upper=129`, `real=113`, the upper plan has two S2 blocks and the real plan one. An upper FD plan can advertise a second partial that the kernel skips.

## Conditional performance recovery

Host tiling cannot read `real` without synchronization. It can use only the interval:

```text
real in [max(0, upper-deltaMax), upper]
```

FD is eligible only if every M row's S2 topology is invariant over this interval. Conservative host pseudocode:

```cpp
for each batch b:
  lower = max(0, upper[b] - deltaMax)
  for each m block:
    if CalcS2Range(m, lower) != CalcS2Range(m, upper):
      stable = false
```

Initial implementation should require all batches stable before enabling the global `streamK` flag. A later splitter may add per-batch S2-split eligibility.

Comparing only total block counts is insufficient for sparse/Right-Down causal ranges.

## More complex alternative

Keeping upper-based FD for unstable intervals requires every planned partial to write a valid reduction value, including empty partitions. The mathematical identity is generally `max=-inf`, `sum=0`, `accumOut=0`, but correctness also requires fixed workspace indices, all synchronization participants, LSE handling and every quantization branch. Do not implement this as the first version.
