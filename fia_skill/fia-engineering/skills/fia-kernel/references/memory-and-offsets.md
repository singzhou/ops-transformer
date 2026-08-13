# FIA memory, layouts and offsets

## Layout and GQA

Normalize Q and KV layouts before interpreting shapes. GQA uses `n2Size` KV heads and `gSize` query heads per KV head. Some internal buffers use S1G order even when external tensors are TND/BSH/BNSD.

## KV addressing

- Contiguous KV: offset calculator uses batch/head/sequence/head-dim and actual lengths for packed layouts.
- TND/NTD: cumulative actual-length arrays may determine batch starts.
- Paged KV: block table maps logical token positions to physical blocks; block size and PA layout affect address formula.
- Tensor-list KV: per-batch tensors and lengths follow a different host representation.

Do not reuse one layout's offset proof for another.

## S2 tail

The task coordinate is a block index. Convert to token start, clamp against real sequence end and compute a strictly nonnegative tail. Audit unsigned subtraction carefully when a host S2 block becomes empty under a device real length.

## Mask alignment

Sparse range controls which S2 blocks exist; attention-mask code controls individual token validity inside a block. Right-Down causal alignment depends on `realKv-realQ`. Using an upper KV value for mask alignment can expose rejected speculative tokens even if GM loads are otherwise bounded.

## Output ownership

Without FD, each complete `(bN2,M)` row has one owner and writes final output. With FD, FA cores write workspace partials and a vector reduction owner writes final output. Keep these paths distinct when changing task skips.

## Synchronization

AIC and AIV execute different branches of a mixed-core kernel while sharing task order and events. A runtime early return on one side can deadlock the other. Prefer common task classification followed by branch-specific work over unilateral early exits.
