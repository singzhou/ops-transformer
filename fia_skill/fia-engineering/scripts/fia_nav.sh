#!/usr/bin/env bash
set -euo pipefail

topic="${1:-all}"
ops_root="${2:-/opt/zsy/ops-transformer}"
vllm_root="${3:-/opt/zsy/vllm-ascend}"
fia_root="${ops_root}/attention/fused_infer_attention_score"
common_root="${ops_root}/attention/common"

case "${topic}" in
  soc)
    rg -n "SOC_VERSION_LIST|ARCH_DIRECTORY_LIST|DAV_2201|DAV_3510|ASCEND_COMPUTE_UNIT" \
      "${ops_root}/CMakeLists.txt" "${fia_root}/op_host" -g '*.{cpp,h}'
    ;;
  tiling)
    rg -n "DoOpTilingFusedInferAttentionScore|REGISTER_TILING_TEMPLATE_FIA|IsCapable|DoOpTiling|GenTilingKey" \
      "${fia_root}/op_host" "${common_root}/op_host" -g '*.{cpp,h}'
    ;;
  split)
    rg -n "SplitCore|AssignByBatch|AssignByRow|AssignByBlock|CheckChooseWithFd|streamK|fdS2SplitNum" \
      "${common_root}/op_host/split_core_v2.cpp" "${common_root}/op_host/split_core_v2.h"
    ;;
  kernel)
    rg -n "fused_infer_attention_score\(|INVOKE_FIA|GetTaskDealMode|CalcCurS2StartEnd|ComputeMm1|ComputeVec1|FlashDecode" \
      "${fia_root}/op_kernel" -g '*.{cpp,h}'
    ;;
  api)
    rg -n "GetWorkspaceSize|aclnnInnerFusedInferAttentionScore|actualSeqLengths" \
      "${fia_root}/op_api" "${fia_root}/op_host/fused_infer_attention_score_def.cpp" -g '*.{cpp,h}'
    ;;
  build)
    rg -n "fused_infer_attention_score|build_aclnn|CUSTOM_OPS_ARRAY|_cann_ops_custom|VLLM_ASCEND_SRC" \
      "${fia_root}/CMakeLists.txt" "${fia_root}/op_host/CMakeLists.txt" \
      "${vllm_root}/CMakeLists.txt" "${vllm_root}/csrc/build_aclnn.sh" "${vllm_root}/setup.py"
    ;;
  tests)
    rg -n "fused_infer_attention_score|FusedInferAttentionScore|flashDecode|actual_seq" \
      "${fia_root}/tests" "${vllm_root}/tests" -g '*.{cpp,h,py,json}' 2>/dev/null || true
    ;;
  runtime-seqlen)
    rg -n "actualSeqLengthsKV|actual_seq_lengths_kv|TilingInputsDataDependency|GetActualSeqLength|streamK" \
      "${fia_root}" "${common_root}/op_host/split_core_v2.cpp" -g '*.{cpp,h}'
    ;;
  all)
    for next_topic in soc tiling split kernel api build tests runtime-seqlen; do
      printf '\n[%s]\n' "${next_topic}"
      "$0" "${next_topic}" "${ops_root}" "${vllm_root}"
    done
    ;;
  *)
    printf 'usage: %s {soc|tiling|split|kernel|api|build|tests|runtime-seqlen|all} [ops-root] [vllm-root]\n' "$0" >&2
    exit 2
    ;;
esac
