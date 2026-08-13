# FIA SoC file map

## Shared sources

- opdef and infer shape: `op_host/fused_infer_attention_score_def.cpp`, `fused_infer_attention_score_infershape.cpp`.
- registration and entry: `fused_infer_attention_score_tiling_register.cpp`, `fused_infer_attention_score_tiling.cpp`.
- parser and checkers: `fused_infer_attention_score_tiling_info_parser.*`, `op_host/checkers/`.
- tiling registry/base: `attention/common/op_host/fia_tiling_templates_registry.h`, `fia_tiling_base.*`.
- split logic: `attention/common/op_host/split_core*.{h,cpp}`.
- public API: `op_api/`.

## 910B/910C (`arch22`, `DAV_2201`)

```text
op_host/arch22/fia_tiling_nonquant.cpp
op_host/arch22/fia_tiling_nonquant_mla.cpp
op_host/arch22/fia_tiling_empty_tensor.cpp
op_host/arch22/fused_infer_attention_score_tiling_v3.cpp
op_host/arch22/fused_infer_attention_score_tiling_check*.cpp
op_kernel/fused_infer_attention_score.cpp
op_kernel/arch22/fused_infer_attention_score_v3.cpp
op_kernel/arch22/flash_attention_interface.cpp
op_kernel/arch22/*.h
```

The generic kernel entry may also include/reuse increment and prompt flash-attention code depending on dynamic-compile macros.

## 950 (`arch35`, `DAV_3510`)

```text
op_host/arch35/fia_tiling_nonquant_gqa.cpp
op_host/arch35/fia_tiling_fullquant_gqa.cpp
op_host/arch35/fia_tiling_fullquant_mx.cpp
op_host/arch35/fused_infer_attention_score_tiling_v4.cpp
op_host/arch35/fused_infer_attention_score_tiling_impl.cpp
op_kernel/fused_infer_attention_score_apt.cpp
op_kernel/arch35/fia_kernel_*.h
op_kernel/arch35/fia_block_cube_*.h
op_kernel/arch35/fia_block_vec_*.h
op_kernel/arch35/flash_attention_*.h
```

## Verification commands

```bash
rg -n 'SOC_VERSION_LIST|ARCH_DIRECTORY_LIST' /opt/zsy/ops-transformer/CMakeLists.txt
rg -n 'REGISTER_TILING_TEMPLATE_FIA' \
  /opt/zsy/ops-transformer/attention/fused_infer_attention_score/op_host
rg -n 'DAV_2201|DAV_3510' \
  /opt/zsy/ops-transformer/attention/fused_infer_attention_score/op_host
rg -n 'TILING_KEY_IS|INVOKE_FIA|template dispatcher' \
  /opt/zsy/ops-transformer/attention/fused_infer_attention_score/op_kernel
```

Re-run these commands after rebases because routing evolves with CANN and operator versions.
