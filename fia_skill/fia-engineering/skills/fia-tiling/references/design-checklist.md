# FIA tiling design checklist

Adapted from the FlashAttention pattern in `ascendc-tiling-design`; use the source code for facts.

1. Define the GQA macro shape: batch, Q heads, KV heads, group size, Q length, KV upper bound, real KV length, D and output layout.
2. Choose parallel axes. Treat FD as optional S2 split-KV; state why it is safe or disabled.
3. Model tile candidates against Cube/Vector work and GM/L1/L0/UB traffic.
4. Prove online-softmax state ownership and merge order.
5. Budget buffers against the target architecture, including ping-pong and FD workspace.
6. Simulate per-core cost for heterogeneous batch lengths; record max/mean imbalance.
7. Define tail/alignment behavior and token/block conversions explicitly.
8. Encode only compile-time distinctions in the tiling key; keep runtime lengths in data/GM.
9. Fill every kernel-consumed tiling field and size workspace for the host upper bound.
10. Prove AIC/AIV and cross-core synchronization for empty or shortened runtime tasks.
11. Keep an exact-length baseline tiling and output for comparison.
12. Optimize only after the protocol and boundary matrix pass.

For optimistic KV, run the checklist twice: forced no-FD safety baseline, then any proposed FD recovery. A small token delta does not imply stable S2 block count, core ownership or FD slot count near a boundary.
