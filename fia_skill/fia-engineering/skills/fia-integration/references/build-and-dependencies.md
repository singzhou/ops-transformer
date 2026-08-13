# FIA build and dependency closure

## vLLM Ascend extension

`/opt/zsy/vllm-ascend/CMakeLists.txt` includes PyTorch, torch_npu and CANN headers and links Torch, `torch_npu`, `ascendcl`, `tiling_api`, `register`, `platform`, `ascendalog`, `dl` and `opapi`.

Current `VLLM_ASCEND_SRC` globs only top-level `csrc/*.cpp`, `csrc/aclnn_torch_adapter/*.cpp`, and one explicit tiling source on supported products. A nested FIA adapter must be header-only or explicitly listed.

The rpath includes:

```text
$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib
```

`op_api_common.h` also searches custom vendor library paths and falls back to system `libopapi.so`.

## Custom OPP

`csrc/build_aclnn.sh` chooses operator names per SoC, runs:

```bash
bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}"
```

and installs the single generated run package under `vllm_ascend/_cann_ops_custom`.

Add `fused_infer_attention_score` to the intended A2/A3/A5 operator arrays. Do not add unsupported SoCs silently.

## FIA local dependency set

The upstream FIA host CMake declares:

```text
attention/fused_infer_attention_score
attention/incre_flash_attention
attention/prompt_flash_attention
attention/common
```

FIA includes common host split/shape code, common kernel math/vector/offset code, and IFA/PFA tiling structs or kernel fallbacks. Copy or refactor the dependency closure while preserving licenses.

## External headers

CANN/toolkit supplies headers such as `register/tilingdata_base.h`, `exe_graph/runtime/tiling_context.h`, `tiling/tiling_api.h`, `platform/platform_info.h`, `kernel_operator.h`, `kernel_vec_intf.h` and `kernel_cube_intf.h`. Do not vendor them.

Generated custom-op build files supply `aclnnInner_*.h`. Do not vendor them either.

## Required registration surfaces

- custom OPP opdef and public ACLNN API;
- `csrc/build_aclnn.sh` operator list;
- PTA header included by `torch_binding.cpp`;
- torch schema and PrivateUse1 function;
- Meta function and registration;
- Python call site and graph/max-workspace path;
- packaging/runtime vendor path.
