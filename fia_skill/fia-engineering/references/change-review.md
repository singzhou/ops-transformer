# FIA cross-layer change review

## Before editing

- State supported SoCs, layouts, dtype/quant modes, GQA/MLA, sparse modes and KV layouts.
- State whether Q and KV lengths are host arrays, device tensors or both.
- Find every host `GetData<T>()` and every kernel GM consumer of the affected input.
- Identify all tiling templates that can win by priority and `IsCapable`.

## Host proof

- Check parser validation and shape maxima.
- Check `mBaseSize`, `s2BaseSize`, valid sparse range and cost model.
- Check core ownership metadata, Stream-K, S1-out split, FD count/workspace and block dim.
- Check workspace sizing against the largest legal runtime case.
- Check tiling key changes and all registered tiling-data classes.

## Kernel proof

- Trace actual-length parser initialization and per-batch caching.
- Verify task enumeration cannot underflow or expand beyond host ownership.
- Verify empty/invalid tasks still satisfy required writes and synchronization.
- Verify PA offsets, mask alignment, tail sizes, prefix/padding and output offsets.
- If FD remains enabled, prove every advertised partial slot is written exactly once.

## Integration proof

- Keep public ACLNN ABI versioned for signature changes.
- Generate inner API headers through the custom OPP build.
- Register PyTorch schema, PrivateUse1 implementation and Meta implementation together.
- Preserve address stability and shapes during ACLGraph capture/replay.
- Package `libcust_opapi.so`, op implementation and vendor scripts with the wheel.

## Validation proof

- Compare against a real-host-length baseline.
- Test block boundaries and `delta` values, including zero and maximum.
- Prefill workspace with nonzero/NaN patterns to expose missing producers.
- Check FD/no-FD tiling metadata, not only output values.
- Run on every supported SoC; compiling an arch directory is not runtime coverage.
