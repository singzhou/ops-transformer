# FIA symptom matrix

| Symptom | First FIA checks |
|---|---|
| Exact length passes, optimistic length fails | Real/upper block boundary, host ownership intersection, sparse range recomputation, Q topology unchanged. |
| Forced no-FD passes, auto-FD fails | `fdNum`, `fdS2SplitNum`, producer slot count, workspace indices, skipped shortened task. |
| Only ACLGraph replay fails | Address stability, in-place length update, capture-stable workspace and optional inputs, hidden host read. |
| Random output or NaN | Unwritten output/FD slot, queue event mismatch, tail alignment, UB/workspace overflow. |
| AIV hang or drain timeout | Missing cross-core flag, empty task taking a different synchronization path, Alloc/Free or EnQue/DeQue imbalance. |
| 561002 | Template acceptance, tiling-data field/range, workspace and tiling-key generation. |
| 561003 | Product build selection, installed custom OPP, kernel name/key, stale or conflicting package. |
| 910B/910C pass, 950 fails | arch35 entry/template/API/RegBase path and larger L0C/UB assumptions. |
| 950 passes, 910B/910C fails | arch22 entry, unsupported API, smaller L0C/UB budget and separate kernel specialization. |
