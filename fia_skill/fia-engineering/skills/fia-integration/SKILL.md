---
name: fia-integration
description: Integrate FusedInferAttentionScore into vLLM Ascend without relying on op-plugin. Use for FIA public/inner ACLNN API and opdef changes, PyTorch Adapter and torch.library registration, custom OPP build/package integration, source/header dependency closure, vllm_ascend_C linking, ACLGraph capture/replay constraints, or optimistic CPU and real NPU sequence-length plumbing.
---

# FIA Integration

Keep PyTorch dispatch and custom operator packaging as two products managed by one repository build.

## Use the in-tree architecture

```text
vllm_ascend_C.so
  -> PyTorch schema + PrivateUse1/Meta implementation
  -> EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV*)
  -> dlopen repo-local libcust_opapi.so

vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  -> public ACLNN/op host/tiling/kernel/config package
```

The top-level CMake builds `vllm_ascend_C`; `setup.py` invokes `csrc/build_aclnn.sh` first when custom kernels are enabled. The script builds selected custom OPP operators and installs the run package under `_cann_ops_custom`, which is copied into the wheel.

Read [references/build-and-dependencies.md](references/build-and-dependencies.md) before moving FIA sources.

## Add the PTA adapter

Follow the existing sparse-flash-attention pattern:

1. Put a header-only adapter under `csrc/attention/fused_infer_attention_score/` and include it in `csrc/torch_binding.cpp`; or explicitly add a nested `.cpp` to CMake because current source glob is not recursive.
2. Validate tensor device, dtype, contiguity, shape and optional arguments.
3. Allocate output tensors without reading device values.
4. Invoke the versioned public ACLNN symbol with `EXEC_NPU_CMD`.
5. Register schema and `PrivateUse1` implementation in `torch_binding.cpp`.
6. Register a shape-only Meta implementation in `torch_binding_meta.cpp`.

Do not include generated inner API headers in the PTA. `op_api_common.h` resolves public symbols dynamically from custom or system op-api libraries.

## Add or version ACLNN inputs

For a new device runtime-KV input:

- add a new public API version rather than changing an existing ABI;
- add the tensor to opdef and generated inner API inputs;
- keep the CPU upper-bound array separate from the NPU real tensor;
- do not mark the real device tensor as a host tiling value dependency;
- update normal and max-workspace APIs consistently;
- ensure kernel signature/input index and every SoC entry agree.

Never copy `aclnnInner_fused_infer_attention_score.h`; custom OPP generation owns it.

## Integrate source closure

FIA's own CMake declares dependencies on `attention/incre_flash_attention`, `attention/prompt_flash_attention` and `attention/common`. Preserve relative include structure or explicitly refactor it. Audit existing partial `csrc/attention/common` before merging; do not overwrite unrelated vLLM Ascend operator files.

## Preserve ACLGraph behavior

- Allocate address-stable runtime tensors outside capture.
- Update contents in place before replay.
- Keep shapes, optional-input presence, output shapes and workspace contract capture-stable.
- Avoid `.item()`, `.tolist()`, CPU conversion or a temporary dtype conversion in the hot path.
- Compare eager and replay against a synchronized real-host-length baseline.

## Verify packaging

Check the installed tree contains `libcust_opapi.so`, op implementation/config files and vendor scripts. Confirm runtime environment discovery in `vllm_ascend/utils.py` and rpath in the top-level CMake.

Use `$fia-validation` for ABI, Meta, eager, graph and NPU tests.
