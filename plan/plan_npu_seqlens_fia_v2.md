# FIA 支持 CPU 乐观 KV seqlen + NPU 真实 KV seqlen 计划 V2

> 状态：设计评审稿，尚未实施  
> 目标目录：`/opt/zsy/ops-transformer/plan/plan_npu_seqlens_fia_v2.md`  
> 白盒分析基线：`/opt/zsy/ops-transformer/attention/fused_infer_attention_score`、`/opt/zsy/op-plugin`、`/opt/zsy/vllm-ascend`  
> 首期范围：arch35、非量化 GQA、PageAttention、TND Query、right-down causal、DSpark speculative decode

---

## 1. 结论摘要

V1 中“tiling 使用 CPU 乐观长度、Execute 时仅替换 kernel seqlen 地址、op_host/op_kernel 无需修改”的方案不可用。原因是 CPU `actual_seq_lengths_kv` 不仅决定 workspace，还直接参与 `SplitCore`，生成每核的 `(BN2, M, S2)` 起止坐标和 FlashDecode 归约 metadata。若 kernel 改读更小的 NPU 真值，host 与 kernel 会使用不同的任务图。

V2 采用以下正确性优先方案：

1. **只动态化 KV seqlen**。Q 的 `actual_seq_qlen` 保持 CPU 精确累积值；它描述 TND Query 的真实分段，不能使用乐观值。
2. 新增独立算子输入 `actual_seq_lengths_kv_runtime`：
   - 原 `actual_seq_lengths_kv` 保留为 CPU 乐观上界，经 `aclIntArray` 转换后供 tiling 使用；
   - 新输入是 NPU `int64` tensor，只供 kernel 使用，不标记 `ValueDepend`，tiling 不读取其内容。
3. runtime-KV 模式下，host 强制 `streamK=false` 并禁用 S1-out-split，使每个 `(batch, kv_head, q-block)` 的完整 S2 行只归属一个核，彻底关闭依赖 CPU KV 长度的 S2 跨核与 FlashDecode。
4. kernel 仍必须按 NPU 真值计算 S2 循环，并对 host metadata 做交集裁剪和无符号下溢保护。
5. ACLNN 新增 V6，直接把 runtime tensor 作为新增的算子输入传入；不采用 executor 地址覆盖等未被当前代码证明存在的机制。
6. op-plugin 新增独立的 `npu_fused_infer_attention_score_v3` schema，避免改变 V2 的 ABI/语义。
7. vLLM Ascend 使用 CPU `seq_lens_cpu_upper_bound` 构造 tiling list，使用地址稳定的 NPU `int64` buffer 作为 runtime KV seqlen；ACLGraph capture/replay 不创建临时 `.to(torch.int64)` tensor。

首期会牺牲一部分长序列 S2 跨核性能，但其任务覆盖集合与真实 KV 长度无关，能够建立明确的正确性证明。第二阶段再设计 device-side 动态分核/归约。

---

## 2. 场景与语义定义

### 2.1 两套 KV 长度

对 batch 中第 `b` 个请求定义：

```text
kv_upper[b] = CPU 乐观 KV 长度
kv_real[b]  = NPU 真实 KV 长度

必须满足：0 <= kv_real[b] <= kv_upper[b]
```

DSpark 中 `kv_upper` 假设上一轮 draft token 全部接受；`kv_real` 已在设备侧减去 rejected token。因此该设计的目的只是消除 `kv_real` 的 D2H 同步，不改变注意力数学语义。

### 2.2 Q 长度必须精确

当前 vLLM Ascend 的 Query 使用 TND，并由 CPU `query_start_loc` 生成精确累积长度：

```python
# vllm_ascend/attention/attention_v1.py，当前实现
actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
```

TND 下 Q parser 使用 ACCUM 模式：

```cpp
// op_kernel/arch35/memory_copy_arch35.h，当前实现
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND ||
                  LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    }
    return ActualSeqLensMode::BY_BATCH;
}
```

因此 V2 不新增 runtime Q seqlen，也不允许 Q 使用乐观长度。若以后动态化 Q，需要单独设计动态 M 任务映射，不能复用本计划。

### 2.3 PageAttention 下 KV tensor 使用 BY_BATCH

现有代码在 PageAttention 下强制 KV seqlen 使用 BY_BATCH，即便 Query layout 是 TND：

```cpp
// op_kernel/arch35/memory_copy_arch35.h，当前实现
template <LayOutTypeEnum LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND ||
                  LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return ActualSeqLensMode::ACCUM;
    }
    return ActualSeqLensMode::BY_BATCH;
}
```

所以 `actual_seq_lengths_kv_runtime` 的格式为 `[B]` 的 per-batch 长度，不做 cumsum。

---

## 3. 当前实现的白盒风险

### 3.1 CPU KV seqlen 进入 SplitCore

现有 GQA tiling 把 `actual_seq_lengths_kv` 的 CPU 值复制到 `BaseInfo.actualSeqS2Size`：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，当前实现
const gert::Tensor *actSeqLenDataKV =
    fiaInfo_->opParamInfo.actualSeqLengths.tensor;
if (actSeqLenDataKV != nullptr) {
    baseInfo.actualSeqS2Size.reserve(baseInfo.bSize);
    const int64_t *s2Ptr = actSeqLenDataKV->GetData<int64_t>();
    for (uint32_t i = 0; i < baseInfo.bSize; i++) {
        baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
    }
}
```

随后直接参与分核：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，当前实现
split_core_v2::FAMetaData result{platformInfo_.aicNum,
                                 platformInfo_.cvRatio};
split_core_v2::SplitCore(platformInfo_.aicNum,
                         baseInfo, splitParam, result);
SetSplitOutput(result);
```

这证明 CPU 长度不是单纯的 workspace 上界。

### 3.2 streamK 会按 S2 block 切分一行

现有 GQA 路径默认开启 streamK：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，当前实现
splitParam.mBaseSize = sOuterFactor_ * CV_RATIO;
splitParam.s2BaseSize = sInnerFactor_;
splitParam.gS1BaseSizeOfFd = 8;
splitParam.streamK = true;
splitParam.fdTolerance = 9;
splitParam.fdLeastBlock = 0;
```

`split_core_v2` 在 `streamK=true` 时允许按 S2 block 分配：

```cpp
// attention/common/op_host/split_core_v2.cpp，当前实现
void AssignByBlock(const SplitContext &splitContext,
                   AssignContext &assignContext)
{
    if (assignContext.isFinished || !splitContext.splitParam.streamK) {
        return;
    }
    // ...
    assignContext.curS2Idx++;
}
```

一旦跨核，就记录依赖 CPU 长度的 FD 分片数和 workspace index：

```cpp
// attention/common/op_host/split_core_v2.cpp，当前实现
result.fdRes.fdBN2Idx[result.fdRes.fdNum] =
    result.bN2End[assignContext.curCoreIdx - 1U];
result.fdRes.fdMIdx[result.fdRes.fdNum] =
    result.mEnd[assignContext.curCoreIdx - 1U];
result.fdRes.fdS2SplitNum[result.fdRes.fdNum] =
    assignContext.curKvSplitPart;
result.fdRes.fdWorkspaceIdx[result.fdRes.fdNum] =
    assignContext.preFdDataNUM;
```

### 3.3 kernel 真值会被 host 尾边界覆盖

kernel 先从 GM 读取 seqlen：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
actSeqLensKv = kvActSeqLensParser.GetActualSeqLength(bIdx);
actSeqLensQ = qActSeqLensParser.GetActualSeqLength(bIdx);
cachedS2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
```

但尾核又可能使用 host metadata 覆盖真实尾边界：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
curS2End = (static_cast<uint32_t>(actSeqLensKv) +
            s2BaseSize - 1) / s2BaseSize;
if (!constInfo.enableS1OutSplit &&
    (bN2Cur == constInfo.bN2End) &&
    (gS1Cur == constInfo.gS1OEnd)) {
    tailS2Split = constInfo.s2OEnd != 0U;
    curS2End = constInfo.s2OEnd;
}
```

若 `s2OEnd` 来自更大的 CPU 上界，它可能大于 `ceil(kv_real / s2BaseSize)`。

后续尾块长度采用无符号减法：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
    info.actSingleLoopS2Size =
        info.actS2Size - s2Cur * s2BaseSize;
}
```

当 `s2Cur * s2BaseSize > kv_real` 时存在下溢风险。

### 3.4 FD kernel 完全信任 host metadata

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
uint32_t fdS2SplitNum = fiaMetaDataGm.GetValue(
    GetFDMetaDataIndex(constInfo.aivIdx, FD_S2_SPLIT_NUM_INDEX));
uint32_t fdWorkspaceIdx = fiaMetaDataGm.GetValue(
    GetFDMetaDataIndex(constInfo.aivIdx, FD_WORKSPACE_IDX_INDEX));
FDparamsX fdParams = {fdCoreEnable, fdBN2Idx, fdMIdx,
                      fdS2SplitNum, mStart, mLen,
                      fdWorkspaceIdx};
vecFdBlock.FlashDecode(fdParams);
```

因此不能让 FA 根据 NPU 真值少产出分片，却仍让 FD 按 CPU 上界的分片数归约。

### 3.5 当前 ACLNN 输入同时服务 tiling 和 kernel

算子定义把 `actual_seq_lengths_kv` 标记为 ValueDepend：

```cpp
// op_host/fused_infer_attention_score_def.cpp，当前实现
this->Input("actual_seq_lengths_kv")
    .ParamType(OPTIONAL)
    .ValueDepend(OPTIONAL)
    .DataTypeList({ge::DT_INT64})
    .FormatList({ge::FORMAT_ND})
    .AutoContiguous();
```

V5 wrapper 将同一个输入交给 inner op：

```cpp
// op_api/aclnn_fused_infer_attention_score_v5.cpp，当前实现
ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    query, tensorListKey, tensorListValue,
    pseShiftOptional, attenMaskOptional,
    actualSeqLengthsOptional,
    actualSeqLengthsKvOptional,
    // ...
    workspaceSize, executor);
```

所以 V2 必须在算子 schema 中显式拆分上界输入和 runtime 输入。

---

## 4. 正确性不变量

实施时必须同时满足以下不变量：

1. `actual_seq_lengths_kv_runtime.dtype == int64`，shape 为 `[B]`，连续且位于 NPU。
2. 对每个 batch：`0 <= kv_real <= kv_upper`。
3. `actual_seq_qlen` 是精确的 TND 累积长度，最后一个元素等于 Query 的 T 维。
4. runtime-KV 首期模式下：
   - `streamK == false`；
   - `fdRes.fdNum == 0`；
   - 每核的 `s2OStart == 0 && s2OEnd == 0`，核边界只能落在完整行之间；
   - `enableS1OutSplit == false`。
5. kernel 的有效 S2 访问范围始终是：

```text
[0, ceil(kv_real / s2BaseSize))
```

6. CPU `kv_upper` 只允许影响核间负载均衡和容量，不能改变逻辑结果。
7. ACLGraph replay 时 runtime tensor 的地址、shape、dtype 不变，只更新其内容。

---

## 5. 修改点 1：算子原型新增 runtime KV tensor

### 5.1 现有代码依据

当前 `actual_seq_lengths_kv` 已占据 kernel 的固定输入位置：

```cpp
// op_kernel/fused_infer_attention_score_apt.cpp，当前实现
__global__ __aicore__ void fused_infer_attention_score(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *pse_shift,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKV,
    // ...
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
```

### 5.2 拟修改代码

在现有 KV seqlen 后新增可选 tensor，保持旧输入语义：

```cpp
// op_host/fused_infer_attention_score_def.cpp，拟修改
this->Input("actual_seq_lengths_kv_runtime")
    .ParamType(OPTIONAL)
    // 不能添加 ValueDepend：host tiling 禁止读取其 NPU 内容
    .DataTypeList({ge::DT_INT64})
    .FormatList({ge::FORMAT_ND})
    .AutoContiguous();
```

kernel 入口同步增加参数：

```cpp
// op_kernel/fused_infer_attention_score_apt.cpp，拟修改
__global__ __aicore__ void fused_infer_attention_score(
    // ...
    __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKV,        // CPU upper bound 的设备副本
    __gm__ uint8_t *actualSeqLengthsKVRuntime, // NPU real value
    // ...
)
```

所有 arch 的入口签名必须同步更新，但首期只有 arch35 nonquant GQA 使用新指针；其他模板继续使用 `actualSeqLengthsKV`。

### 5.3 tiling 参数增加模式标志

当前 kernel 从 tiling data 获取长度维数：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
constInfo.actualSeqLenKVSize =
    fiaBaseParams.actualSeqLengthsKVSize;
```

增加显式标志，避免用空指针猜测模式：

```cpp
// arch35 tiling data，拟修改
BEGIN_TILING_DATA_DEF(FiaBaseParams)
    // ... existing fields
    TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
    TILING_DATA_FIELD_DEF(uint8_t, useRuntimeKvSeqLen);
END_TILING_DATA_DEF;
```

host 设置：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，拟修改
tilingData_.baseTiling.fiaBaseParams.useRuntimeKvSeqLen =
    fiaInfo_->runtimeKvSeqLenFlag ? 1U : 0U;
```

该字段只改变 kernel 选择哪个输入地址，不需要新增 tiling key/template 实例。

---

## 6. 修改点 2：FiaInfoParser 只检查 runtime tensor 元信息

### 6.1 现有代码依据

当前 parser 会通过 `GetData<int64_t>()` 判断原 seqlen 是否可供 host 读取：

```cpp
// op_host/fused_infer_attention_score_tiling_info_parser.cpp，当前实现
if ((opParamInfo_.actualSeqLengths.tensor != nullptr &&
     opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() == nullptr) ||
    (opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
     opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr)) {
    isMaxWorkspace_ = true;
}
```

### 6.2 拟修改代码

在 `FIAParaInfo` 中新增字段：

```cpp
// op_host/fia_tiling_info.h，拟修改
struct FIAParaInfo {
    // ...
    FIAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    FIAOptionalParaInfo actualSeqLengthsKvRuntime = {nullptr, nullptr};
    // ...
};

struct FiaTilingInfo {
    // ...
    bool runtimeKvSeqLenFlag = false;
};
```

parser 只读取 shape/desc，不读取 runtime data：

```cpp
// op_host/fused_infer_attention_score_tiling_info_parser.cpp，拟修改
void FiaInfoParser::GetOptionalInputParaActualSeqLengthInfo()
{
    // existing upper-bound inputs
    opParamInfo_.actualSeqLengths.tensor =
        context_->GetOptionalInputTensor(ACTUAL_SEQ_LENGTHS_KV_INDEX);

    // new device runtime input
    opParamInfo_.actualSeqLengthsKvRuntime.tensor =
        context_->GetOptionalInputTensor(
            ACTUAL_SEQ_LENGTHS_KV_RUNTIME_INDEX);
    opParamInfo_.actualSeqLengthsKvRuntime.desc =
        context_->GetOptionalInputDesc(
            ACTUAL_SEQ_LENGTHS_KV_RUNTIME_INDEX);

    runtimeKvSeqLenFlag_ =
        opParamInfo_.actualSeqLengthsKvRuntime.tensor != nullptr;
}
```

新增 shape/dtype 校验：

```cpp
// 拟新增 checker
ge::graphStatus CheckRuntimeKvSeqLen(const FiaTilingInfo &info)
{
    if (!info.runtimeKvSeqLenFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const auto *runtime =
        info.opParamInfo.actualSeqLengthsKvRuntime.tensor;
    OP_CHECK_IF(runtime->GetShapeSize() != info.bSize,
                OP_LOGE(info.opName,
                        "runtime KV seqlen size must equal batch size"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.actualLenKvDims != info.bSize,
                OP_LOGE(info.opName,
                        "upper-bound KV seqlen size must equal batch size"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

注意：host 无法验证逐元素 `kv_real <= kv_upper`，该条件由调用方保证，并通过 debug/device test 覆盖。禁止为了做这个检查把 runtime tensor D2H。

---

## 7. 修改点 3：为首期模式设置严格路由门禁

### 7.1 现有代码依据

当前 GQA 模板支持范围较宽，只拒绝部分 feature：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，当前实现
bool FiaTilingNonQuantArch35::IsCapableFeatureCheckGqa()
{
    if (fiaInfo_->sysPrefixFlag ||
        fiaInfo_->pseShiftFlag ||
        fiaInfo_->enableAlibiPse ||
        fiaInfo_->qPaddingSizeFlag ||
        fiaInfo_->kvPaddingSizeFlag ||
        fiaInfo_->isOutQuantEnable ||
        fiaInfo_->learnableSinkFlag ||
        fiaInfo_->isQKVDDifferent ||
        fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return false;
    }
    return true;
}
```

### 7.2 拟修改代码

新增 runtime 模式专用 capability check：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，拟修改
bool FiaTilingNonQuantArch35::CheckRuntimeKvSeqLenScope()
{
    if (!fiaInfo_->runtimeKvSeqLenFlag) {
        return true; // legacy path unchanged
    }

    return fiaInfo_->quantMode == FiaQuantMode::NO_QUANT &&
           fiaInfo_->mlaMode == MlaMode::NO_MLA &&
           fiaInfo_->pageAttentionFlag &&
           fiaInfo_->qLayout == FiaLayout::TND &&
           fiaInfo_->sparseMode == SPARSE_MODE_RIGHT_DOWN &&
           !fiaInfo_->sysPrefixFlag &&
           !fiaInfo_->pseShiftFlag &&
           !fiaInfo_->qPaddingSizeFlag &&
           !fiaInfo_->kvPaddingSizeFlag &&
           !fiaInfo_->learnableSinkFlag;
}
```

不满足门禁时 V6 返回明确错误，不静默退回使用 CPU 乐观值计算。vLLM 调用侧应回退旧 V2/V5 路径并执行现有 CPU 同步，保证正确性。

---

## 8. 修改点 4：runtime 模式禁止 S2 跨核和 FlashDecode

### 8.1 现有代码依据

`SplitParam.streamK=false` 会阻止 `AssignByBlock`：

```cpp
// attention/common/op_host/split_core_v2.cpp，当前实现
if (assignContext.isFinished || !splitContext.splitParam.streamK) {
    return;
}
```

当 `streamK=false` 时，分核器把 `coreCache.costLimit` 至少提升到一整行的最大 cost，从而允许按完整行分配：

```cpp
// attention/common/op_host/split_core_v2.cpp，当前实现
if (!splitContext.splitParam.streamK) {
    assignContext.coreCache.costLimit =
        std::max(avgCost, costInfo.maxMCost);
} else {
    assignContext.coreCache.costLimit = avgCost;
}
```

### 8.2 拟修改代码

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，拟修改
void FiaTilingNonQuantArch35::CreateSplitInput(
    split_core_v2::BaseInfo &baseInfo,
    split_core_v2::SplitParam &splitParam)
{
    // existing population of exact Q + upper-bound KV
    // ...

    splitParam.mBaseSize = sOuterFactor_ * CV_RATIO;
    splitParam.s2BaseSize = sInnerFactor_;

    if (fiaInfo_->runtimeKvSeqLenFlag) {
        // CPU KV is only a cost/capacity upper bound. Never split one row
        // across cores because the real number of S2 blocks is device-only.
        splitParam.streamK = false;
        splitParam.fdTolerance = 0;
        splitParam.fdLeastBlock = 0;
    } else {
        splitParam.streamK = true;
        splitParam.fdTolerance = 9;
        splitParam.fdLeastBlock = 0;
    }
}
```

同时禁用 S1-out-split，缩小首期调度状态空间：

```cpp
// op_host/arch35/fia_tiling_nonquant_gqa.cpp，拟修改
enableS1OutSplit =
    fiaInfo_->runtimeKvSeqLenFlag ? false : CheckS1OutSplit();
```

分核后增加硬断言：

```cpp
// 拟修改
split_core_v2::SplitCore(platformInfo_.aicNum,
                         baseInfo, splitParam, result);

if (fiaInfo_->runtimeKvSeqLenFlag) {
    OP_CHECK_IF(result.fdRes.fdNum != 0U,
                OP_LOGE(fiaInfo_->opName,
                        "runtime KV mode must not generate FD tasks"),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < result.usedCoreNum; ++i) {
        OP_CHECK_IF(result.s2End[i] != 0U,
                    OP_LOGE(fiaInfo_->opName,
                            "runtime KV core boundary must be row-aligned"),
                    return ge::GRAPH_FAILED);
    }
}
```

`SplitPolicy()` 当前返回 `void`，落地时应改为 `ge::graphStatus`，让上述断言能向上传递失败，而不是只打日志。

### 8.3 正确性理由

Q 长度精确，因此所有 `(BN2, M)` 行集合在 host 与 kernel 间一致。`streamK=false` 后，CPU `kv_upper` 只影响一整行的 cost 和不同核拿到多少行，不改变行集合，也不会产生 S2 跨核归约。每个核进入一行后，再由 NPU `kv_real` 决定该行实际循环多少 S2 block。

---

## 9. 修改点 5：kernel 改读 runtime KV tensor

### 9.1 现有代码依据

当前 kernel 把 `actualSeqLengthsKv` 解释成 `uint64_t`：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
actualSeqLengthsGmKv.SetGlobalBuffer(
    (__gm__ uint64_t *)actualSeqLengthsKv,
    constInfo.actualSeqLenKVSize);
kvActSeqLensParser.Init(actualSeqLengthsGmKv,
                        constInfo.actualSeqLenKVSize,
                        constInfo.s2Size);
```

### 9.2 拟修改代码

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，拟修改
__aicore__ inline void Init(
    // ...
    __gm__ uint8_t *actualSeqLengthsKvUpper,
    __gm__ uint8_t *actualSeqLengthsKvRuntime,
    // ...
)
{
    // Q remains exact and unchanged.
    actualSeqLengthsGmQ.SetGlobalBuffer(
        (__gm__ uint64_t *)actualSeqLengths,
        constInfo.actualSeqLenSize);
    qActSeqLensParser.Init(actualSeqLengthsGmQ,
                           constInfo.actualSeqLenSize,
                           constInfo.s1Size);

    __gm__ uint8_t *kvSeqLenAddr =
        constInfo.useRuntimeKvSeqLen ?
            actualSeqLengthsKvRuntime : actualSeqLengthsKvUpper;

    actualSeqLengthsGmKv.SetGlobalBuffer(
        (__gm__ uint64_t *)kvSeqLenAddr,
        constInfo.actualSeqLenKVSize);
    kvActSeqLensParser.Init(actualSeqLengthsGmKv,
                            constInfo.actualSeqLenKVSize,
                            constInfo.s2Size);
}
```

并在 `InitConstInfo()` 读取新标志：

```cpp
// 拟修改
constInfo.useRuntimeKvSeqLen =
    fiaBaseParams.useRuntimeKvSeqLen != 0U;
```

Cube、Vec 和 output offset calculator 接收的 KV seqlen 地址也必须统一使用 `kvSeqLenAddr`，不能只有 scheduler 改读真值：

```cpp
// 拟修改
vecFaBlock.InitVecBlock(tPipe,
                        actualSeqLengths,
                        kvSeqLenAddr,
                        attenMask, softmaxLse,
                        attentionOut, workspace);

cubeBlock.InitCubeBlock(tPipe, &l1BufferManager,
                        query, key, value, blockTable,
                        queryRope, keyRope,
                        actualSeqLengths,
                        kvSeqLenAddr);
```

否则 scheduler、Cube/Vec 地址计算仍会看到不同的 KV 长度。

---

## 10. 修改点 6：kernel 对 host metadata 做防御性裁剪

即使首期禁止 S2 跨核，也必须防止将来配置回归或异常 metadata 引发越界。

### 10.1 现有代码依据

当前尾核会无条件用 host `s2OEnd` 覆盖按 NPU seqlen 算出的 `curS2End`：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
curS2End = (static_cast<uint32_t>(actSeqLensKv) +
            s2BaseSize - 1) / s2BaseSize;
if (!constInfo.enableS1OutSplit &&
    (bN2Cur == constInfo.bN2End) &&
    (gS1Cur == constInfo.gS1OEnd)) {
    tailS2Split = constInfo.s2OEnd != 0U;
    curS2End = constInfo.s2OEnd;
}
```

### 10.2 拟修改 `CalcCurS2StartEndNoSparse`

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，拟修改
__aicore__ inline void CalcCurS2StartEndNoSparse(
    uint32_t bN2Cur, uint32_t gS1Cur)
{
    const uint32_t realS2End =
        (static_cast<uint32_t>(actSeqLensKv) +
         s2BaseSize - 1U) / s2BaseSize;

    uint32_t hostStart = 0U;
    uint32_t hostEnd = realS2End;

    if ((bN2Cur == constInfo.bN2Start) &&
        (gS1Cur == constInfo.gS1OStart)) {
        hostStart = constInfo.s2OStart;
    }
    if (!constInfo.enableS1OutSplit &&
        (bN2Cur == constInfo.bN2End) &&
        (gS1Cur == constInfo.gS1OEnd) &&
        constInfo.s2OEnd != 0U) {
        hostEnd = constInfo.s2OEnd;
    }

    // Host metadata may narrow a range, but can never expand beyond NPU real.
    curS2Start = AttentionCommon::Min(hostStart, realS2End);
    curS2End = AttentionCommon::Min(hostEnd, realS2End);
    if (curS2Start >= curS2End) {
        curS2Start = curS2End;
    }
}
```

带 sparse/mask 的版本也应采用同样的最终交集：

```cpp
// CalcCurS2StartEndWithSparse 末尾，拟修改
curS2Start = AttentionCommon::Min(curS2Start, realS2End);
curS2End = AttentionCommon::Min(curS2End, realS2End);
```

### 10.3 防止尾块无符号下溢

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，拟修改
const uint64_t s2Offset =
    static_cast<uint64_t>(s2Cur) * s2BaseSize;
if (s2Offset >= info.actS2Size) {
    info.isValid = false;
    info.actSingleLoopS2Size = 0U;
    return;
}

info.actSingleLoopS2Size = AttentionCommon::Min(
    static_cast<uint64_t>(s2BaseSize),
    info.actS2Size - s2Offset);
```

`CreateTask()` 只有在 `actSingleLoopS2Size > 0` 时才调用 `EnableTask()`：

```cpp
// 拟修改
CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
if (runInfo.actSingleLoopS2Size != 0U) {
    EnableTask(runInfo);
}
```

---

## 11. 修改点 7：新增 ACLNN V6，不覆盖 executor 地址

### 11.1 现有代码依据

V5 把 `aclIntArray` 直接传给 inner op，框架负责生成 tensor 输入：

```cpp
// op_api/aclnn_fused_infer_attention_score_v5.cpp，当前实现
ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    // ...
    actualSeqLengthsOptional,
    actualSeqLengthsKvOptional,
    // ...
    workspaceSize, executor);
```

### 11.2 V6 API 原型

```cpp
// op_api/aclnn_fused_infer_attention_score_v6.h，拟新增
aclnnStatus aclnnFusedInferAttentionScoreV6GetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    // ... V5 parameters ...
    const aclIntArray *actualSeqLengthsOptional,       // exact Q
    const aclIntArray *actualSeqLengthsKvUpperOptional,// CPU upper
    const aclTensor *actualSeqLengthsKvRuntimeOptional,// NPU real
    // ...
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFusedInferAttentionScoreV6(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

### 11.3 V6 inner 调用

```cpp
// op_api/aclnn_fused_infer_attention_score_v6.cpp，拟新增
ret = aclnnInnerFusedInferAttentionScoreV6GetWorkspaceSize(
    query, tensorListKey, tensorListValue,
    pseShiftOptional, attenMaskOptional,
    actualSeqLengthsOptional,
    actualSeqLengthsKvUpperOptional,
    actualSeqLengthsKvRuntimeOptional,
    // ...
    attentionOut, placeHolder,
    workspaceSize, executor);
```

Execute 只执行已经包含三个 seqlen 输入的 executor：

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV6(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerFusedInferAttentionScoreV6(
        workspace, workspaceSize, executor, stream);
}
```

禁止采用以下 V1 计划中的伪接口：

```cpp
// 不采用
executor->OverrideInputTensorAddr(...);
```

原因是当前仓库没有该接口依据，而且地址覆盖也无法修复 host metadata。

### 11.4 兼容性

- V1-V5 原型和行为完全不变。
- V6 若 runtime tensor 为空，可退化为 legacy 输入，但 vLLM 新路径必须显式传入。
- V6 MaxWorkspace API 也必须接收 runtime tensor，使图 capture 时算子 schema 和 replay 一致。

---

## 12. 修改点 8：op-plugin 增加独立 V3 PyTorch op

### 12.1 现有代码依据

当前 schema 的 seqlen 都是 `SymInt[]`：

```yaml
# op_plugin/config/op_plugin_functions.yaml，当前实现（摘要）
- func: npu_fused_infer_attention_score_v2(
    Tensor query, Tensor key, Tensor value,
    *,
    SymInt[]? actual_seq_qlen=None,
    SymInt[]? actual_seq_kvlen=None,
    Tensor? block_table=None,
    ...
  ) -> (Tensor, Tensor)
```

当前 C++ 在非 950 调 V4，在 950 调 V5：

```cpp
// FusedInferAttentionScoreV2KernelNpuOpApi.cpp，当前实现
if (c10_npu::GetSocVersion() != c10_npu::SocVersion::Ascend950) {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnFusedInferAttentionScoreV4, /* ... */);
} else {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnFusedInferAttentionScoreV5, /* ... */);
}
```

### 12.2 拟新增 schema

```yaml
# op_plugin/config/op_plugin_functions.yaml，拟新增
- func: npu_fused_infer_attention_score_v3(
    Tensor query, Tensor key, Tensor value,
    *,
    SymInt[]? actual_seq_qlen=None,
    SymInt[]? actual_seq_kvlen_upper=None,
    Tensor? actual_seq_kvlen_runtime=None,
    Tensor? block_table=None,
    ...
  ) -> (Tensor, Tensor)

- func: npu_fused_infer_attention_score_v3.out(
    Tensor query, Tensor key, Tensor value,
    *,
    SymInt[]? actual_seq_qlen=None,
    SymInt[]? actual_seq_kvlen_upper=None,
    Tensor? actual_seq_kvlen_runtime=None,
    ...,
    Tensor? workspace=None,
    Tensor(a!) attention_out,
    Tensor(b!) softmax_lse
  ) -> (Tensor(a!), Tensor(b!))

- func: _npu_fused_infer_attention_score_v3_get_max_workspace(
    Tensor query, Tensor key, Tensor value,
    *,
    SymInt[]? actual_seq_qlen=None,
    SymInt[]? actual_seq_kvlen_upper=None,
    Tensor? actual_seq_kvlen_runtime=None,
    ...
  ) -> Tensor
```

C++ adapter 进行严格检查后调用 V6：

```cpp
// 拟新增 FusedInferAttentionScoreV3KernelNpuOpApi.cpp
TORCH_CHECK(actual_seq_kvlen_runtime.has_value(),
            "actual_seq_kvlen_runtime is required");
const at::Tensor &runtime = actual_seq_kvlen_runtime.value();
TORCH_CHECK(runtime.device().type() == c10::DeviceType::PrivateUse1,
            "runtime KV seqlen must be on NPU");
TORCH_CHECK(runtime.scalar_type() == at::kLong,
            "runtime KV seqlen must be int64");
TORCH_CHECK(runtime.is_contiguous(),
            "runtime KV seqlen must be contiguous");

EXEC_NPU_NO_FORMAT_CHECK_CMD(
    aclnnFusedInferAttentionScoreV6,
    query_wrapper, keyTensors_wrapper, valueTensors_wrapper,
    // ...
    actual_seq_qlen,
    actual_seq_kvlen_upper,
    runtime,
    // ...
    outTensor_wrapper, softmax_lse);
```

新文件还需要实现 `.out` 和 max-workspace 变体，保持 ACLGraph 当前的 workspace 缓存方式。

---

## 13. 修改点 9：vLLM Ascend 分离 upper list 与 runtime tensor

### 13.1 现有代码风险依据

metadata builder 当前先选择 CPU seqlen，再直接 `.tolist()`：

```python
# vllm_ascend/attention/attention_v1.py，当前实现
if common_attn_metadata._seq_lens_cpu is not None:
    seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
elif common_attn_metadata.seq_lens_cpu is not None:
    seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
else:
    seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

# parallel drafting 又会改回 NPU tensor
elif self.speculative_config and \
     self.speculative_config.parallel_drafting:
    seq_lens = common_attn_metadata.seq_lens

seq_lens_list = seq_lens.tolist()
```

parallel drafting 时最后一行会触发 D2H 同步。

### 13.2 metadata 增加明确字段

```python
# vllm_ascend/attention/attention_v1.py，拟修改
@dataclass
class AscendMetadata:
    # existing
    seq_lens: torch.Tensor = None
    seq_lens_list: list[int] = None

    # new explicit semantics
    seq_lens_kv_runtime: torch.Tensor | None = None
    seq_lens_kv_upper_bound_list: list[int] | None = None
```

builder 分离数据源：

```python
# 拟修改
seq_lens_kv_runtime = common_attn_metadata.seq_lens[:num_reqs]

upper_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
if upper_cpu is None:
    upper_cpu = common_attn_metadata._seq_lens_cpu
if upper_cpu is None:
    upper_cpu = common_attn_metadata.seq_lens_cpu

# runtime mode must never fall back to NPU->CPU.
assert upper_cpu is not None
upper_cpu = upper_cpu[:num_reqs]
seq_lens_kv_upper_bound_list = upper_cpu.tolist()

attn_metadata = self.metadata_cls(
    # Keep existing fields for legacy consumers.
    seq_lens=seq_lens_kv_runtime,
    seq_lens_list=seq_lens_kv_upper_bound_list,
    # New unambiguous fields.
    seq_lens_kv_runtime=seq_lens_kv_runtime,
    seq_lens_kv_upper_bound_list=
        seq_lens_kv_upper_bound_list,
    # ...
)
```

只有满足 V6 scope gate 的 DSpark 路径使用新字段；其他 backend 保持旧逻辑。

### 13.3 调用 V3

当前 graph 路径调用 V2：

```python
# vllm_ascend/attention/attention_v1.py，当前实现
torch_npu.npu_fused_infer_attention_score_v2.out(
    # ...
    actual_seq_qlen=actual_seq_lengths_q,
    actual_seq_kvlen=actual_seq_lengths_kv,
    # ...
)
```

拟修改为：

```python
torch_npu.npu_fused_infer_attention_score_v3.out(
    query=query,
    key=key,
    value=value,
    atten_mask=attn_metadata.attn_mask,
    block_table=block_table,
    input_layout="TND",
    block_size=block_size,
    actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
    actual_seq_kvlen_upper=
        attn_metadata.seq_lens_kv_upper_bound_list,
    actual_seq_kvlen_runtime=
        attn_metadata.seq_lens_kv_runtime,
    num_key_value_heads=self.num_kv_heads,
    num_query_heads=self.num_heads,
    sparse_mode=3,
    pre_tokens=SWA_INT_MAX,
    next_tokens=0,
    softmax_scale=self.scale,
    workspace=workspace,
    out=[output, softmax_lse],
)
```

max-workspace 调用同步切换到 V3，参数必须完全一致。

---

## 14. 修改点 10：ACLGraph 使用地址稳定的 int64 runtime buffer

### 14.1 数据类型依据

kernel parser 默认按 `uint64_t` 读：

```cpp
// op_kernel/arch35/fia_kernel_noquant_gqa.h，当前实现
GlobalTensor<uint64_t> actualSeqLengthsGmKv;
```

而 vLLM 当前 seqlen 通常是 `int32`。不能把 int32 地址直接传给 kernel，也不能在每次调用中创建临时：

```python
# 禁止作为 ACLGraph 最终方案
seq_lens_runtime = seq_lens.to(torch.int64)
```

这虽不发生 D2H，但会创建新 tensor，地址和生命周期不适合作为 replay 输入。

### 14.2 拟修改代码

为 FIA 单独分配持久化 buffer：

```python
# model runner / DSpark graph buffer 初始化，拟修改
self.fia_seq_lens_kv_runtime = torch.empty(
    self.max_num_reqs,
    dtype=torch.int64,
    device=self.device,
)
```

在 graph capture 的固定位置执行 device cast/copy：

```python
# 每轮、进入 FIA 之前，拟修改
self.fia_seq_lens_kv_runtime[:num_reqs].copy_(
    common_attn_metadata.seq_lens[:num_reqs],
    non_blocking=True,
)
self.fia_seq_lens_kv_runtime[num_reqs:].zero_()

attn_metadata.seq_lens_kv_runtime = \
    self.fia_seq_lens_kv_runtime[:num_reqs_padded]
```

要求：

- 目标 buffer 在 capture 前创建，replay 期间不重新分配；
- 源 `seq_lens` 同样是已有持久化 graph input；
- `copy_`/cast 位于 capture group 内或在 replay 前按 stream/event 顺序完成；
- profiler 必须确认没有 `DtoH`/stream synchronize；
- 若上游允许，长期方案是让 DSpark 直接维护 int64 seqlen，移除 cast kernel，但不能盲目修改全局 `seq_lens` dtype，以免破坏其他 int32 kernel。

### 14.3 graph 参数缓存

当前代码把动态 tensor 保存在 graph params 中：

```python
# vllm_ascend/attention/attention_v1.py，当前实现（摘要）
graph_params.attn_params[num_tokens].append(
    (
        weak_ref_tensors(query),
        weak_ref_tensors(key),
        weak_ref_tensors(value),
        # ...
        actual_seq_lengths_kv,
        # ...
    )
)
```

拟修改为保存 runtime buffer 的 weak ref，并把 CPU upper list 作为 capture-time tiling 参数单独保存：

```python
graph_params.attn_params[num_tokens].append(
    (
        # ...
        attn_metadata.seq_lens_kv_upper_bound_list,
        weak_ref_tensors(
            attn_metadata.seq_lens_kv_runtime),
        # ...
    )
)
```

---

## 15. 修改点 11：fallback 与错误处理

### 15.1 现有代码依据

当前 graph FIA 调用会根据 sliding window 选择 sparse mode，并把 sink 直接传入算子；这些 feature 都会改变 mask/归约语义，不能未经验证进入首期 runtime 模式：

```python
# vllm_ascend/attention/attention_v1.py，当前实现
sparse_mode=4 if self.sliding_window is not None else 3,
pre_tokens=(self.sliding_window
            if self.sliding_window is not None
            else SWA_INT_MAX),
next_tokens=0,
learnable_sink=self.sinks,
```

### 15.2 拟修改代码

调用侧必须有显式 capability gate：

```python
# vllm_ascend/attention/attention_v1.py，拟新增
def _can_use_runtime_kv_seqlen(self, metadata) -> bool:
    return (
        self.speculative_config is not None
        and self.speculative_config.parallel_drafting
        and metadata.block_tables is not None
        and self.vllm_config.quant_config is None
        and self.sliding_window is None
        and self.sinks is None
    )
```

行为：

```python
if self._can_use_runtime_kv_seqlen(attn_metadata):
    return self._run_fia_v3_runtime_kv(...)

# Legacy fallback: keep current synchronized path for correctness.
return self._run_fia_v2(...)
```

以下情况不得静默传 upper bound 当真值：

- runtime tensor 缺失、dtype 非 int64或 shape 不匹配；
- 非 PageAttention；
- 非 TND Query；
- fullquant/antiquant；
- sliding-window、sink、PSE、padding；
- runtime mode 意外生成 FD metadata。

---

## 16. 文件修改清单

### 16.1 ops-transformer

| 文件 | 修改 |
|---|---|
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp` | 新增 runtime KV tensor 输入，不设 ValueDepend |
| `.../op_host/fia_tiling_info.h` | 新增参数描述和 runtime 模式标志 |
| `.../op_host/fused_infer_attention_score_tiling_index.h` | 新增输入 index |
| `.../op_host/fused_infer_attention_score_tiling_info_parser.cpp` | 解析 runtime tensor 元信息，禁止读取内容 |
| `.../op_host/checkers/*actual_seq_len*` | 增加 dtype/shape/scope 校验 |
| `.../op_host/arch35/fia_tiling_nonquant_gqa.cpp/.h` | runtime gate、streamK=false、禁用 S1-out-split、metadata 断言 |
| `.../op_kernel/fused_infer_attention_score_apt.cpp` | kernel 入口增加 runtime 指针 |
| `.../op_kernel/arch35/fia_kernel_noquant_gqa.h` | 选择 runtime KV、边界交集、下溢保护 |
| `.../op_kernel/arch35/fia_block_cube_noquant_gqa.h` | 统一接收 runtime KV 地址 |
| `.../op_kernel/arch35/fia_block_vec_noquant_gqa.h` | 统一接收 runtime KV 地址 |
| `.../op_kernel/arch35/fia_block_vec_flashdecode.h` | 首期不执行；签名兼容检查 |
| `.../op_kernel/arch35/*tiling_data*` | 增加 `useRuntimeKvSeqLen` |
| `.../op_api/aclnn_fused_infer_attention_score_v6.h/.cpp` | 新增 V6 API |
| `.../tests/ut/op_host/arch35/*` | tiling 与 metadata 单测 |
| `.../tests/pytest/*gqa*` | NPU 真值差异化正确性测试 |

### 16.2 op-plugin

| 文件 | 修改 |
|---|---|
| `op_plugin/config/op_plugin_functions.yaml` | 新增 V3、V3.out、V3 max-workspace schema |
| `op_plugin/ops/opapi/FusedInferAttentionScoreV3KernelNpuOpApi.cpp` | tensor 校验和 ACLNN V6 调用 |
| 生成的 binding/opapi 文件 | 按项目生成流程刷新 |
| 对应 UT | schema、dtype、device、out/workspace 测试 |

### 16.3 vllm-ascend

| 文件 | 修改 |
|---|---|
| `vllm_ascend/attention/attention_v1.py` | 分离 upper list/runtime tensor，调用 V3 |
| `vllm_ascend/attention/utils.py` | metadata 字段透传和 unpadded 支持 |
| `vllm_ascend/worker/model_runner_v1.py` | 分配/更新稳定 int64 runtime buffer |
| `vllm_ascend/worker/v2/input_batch.py` | 若 V2 runner 纳入首期，增加对应 buffer 字段 |
| `vllm_ascend/worker/v2/model_runner.py` | 若 V2 runner 纳入首期，传递 upper/runtime |
| `vllm_ascend/spec_decode/*dspark*` | graph capture/replay 参数绑定 |
| `tests/ut/ops/*attention*` | 无 D2H 路径和参数选择 UT |
| `tests/ut/spec_decode/*dspark*` | rejected token/upper-bound regression |

建议首个实现 PR 只接 v1 model runner 的 DSpark 路径；v2 runner 后续单独提交，避免一次扩大验证面。

---

## 17. 测试计划

### 17.1 host tiling UT

对相同 Q、不同 `kv_upper` 运行 tiling，验证：

```text
useRuntimeKvSeqLen == 1
fdRes.fdNum == 0
enableS1OutSplit == false
所有 used core 的 s2End == 0
任务的 (BN2, M) 行集合完整且无重复
```

用例：

| B | GQA heads | Q lens | kv_upper |
|---:|---|---|---|
| 1 | 32/8 | `[1]` | `[128]` |
| 1 | 32/8 | `[4]` | `[132]` |
| 4 | 32/8 | `[4,8,12,16]`（累积） | `[127,128,129,260]` |
| 16 | 64/8 | 每请求 4 token | 长短混合 |

### 17.2 kernel/NPU 精度测试

同一组 Q/K/V/block table 分别运行：

1. baseline：CPU list = `kv_real`，旧 V5；
2. candidate：CPU upper = `kv_upper`，NPU tensor = `kv_real`，新 V6；

比较 attention output 和 softmax LSE。

必须覆盖 S2 block 边界：

```text
(real, upper):
(1, 2)
(63, 64)
(64, 65)
(127, 128)
(127, 129)
(128, 129)
(129, 132)
(255, 256)
(255, 260)
```

多 batch 异构：

```text
real  = [127, 128, 129, 255]
upper = [132, 132, 132, 260]
```

差值覆盖 speculative steps：`upper-real = 0, 1, K-1, K`。

### 17.3 非法输入测试

- runtime dtype=int32：报错；
- runtime 位于 CPU：报错；
- runtime shape 不等于 B：报错；
- upper list 长度不等于 B：报错；
- runtime mode + non-PA/fullquant/sliding-window：明确拒绝；
- `kv_real > kv_upper`：debug kernel 或测试 harness 检出，不作为生产 D2H 检查；
- runtime 空 tensor：只有显式 legacy 调用才允许回退。

### 17.4 内存安全

- 使用 guard region/canary 检查 output、workspace、KV cache 前后区域；
- workspace 预填充 NaN/不同随机值，验证结果不依赖未写 FD 分片；
- 开启可用的 sanitizer/异常检测；
- 特别验证 `real=127, upper=129`，它会跨过 128 的 host S2 block 边界。

### 17.5 ACLGraph

1. capture 时固定 `B/num_tokens` 和 runtime buffer 地址；
2. 连续 replay，不重新 capture：

```text
real:  [128, 128] -> [127, 126] -> [124, 128]
upper: [132, 132]（capture-time 上界不变）
```

3. 每次与 eager baseline 比较；
4. profiler 确认无 `.item()`、`.tolist()` on NPU、DtoH memcpy、host synchronize；
5. 验证 graph params weak ref 未失效且地址未变化。

### 17.6 性能

分别测：

- 旧路径 D2H 同步耗时；
- 新路径 int32->int64 device copy/cast；
- `streamK=false` 对长 KV 的性能损失；
- batch 1/4/16/32，KV 1K/4K/16K/64K；
- ACLGraph replay latency 和吞吐。

验收优先级：正确性和零 D2H 为硬门槛；若 `streamK=false` 性能回退过大，进入第二阶段动态调度，不得恢复不安全的 CPU metadata/真实长度混用。

---

## 18. 实施顺序

### 阶段 A：算子正确性闭环

1. 扩展 op schema、parser、tiling data 和 kernel signature。
2. 实现 runtime scope gate。
3. 设置 `streamK=false`、禁用 S1-out-split并加入 metadata 断言。
4. kernel 全链路切换 runtime KV 地址。
5. 加入边界交集和下溢保护。
6. 完成 tiling UT 和 NPU pytest，先不接 vLLM。

### 阶段 B：ACLNN/op-plugin

1. 新增 ACLNN V6 和 max-workspace API。
2. 新增 op-plugin V3/out/max-workspace。
3. 完成 eager C++/Python smoke 和错误输入测试。

### 阶段 C：vLLM Ascend eager

1. metadata 分离 upper/runtime。
2. 增加稳定 int64 buffer。
3. DSpark eager 接 V3；与旧同步路径逐 token 对比。

### 阶段 D：ACLGraph

1. 将 runtime buffer 更新纳入 capture/replay 生命周期。
2. 验证多组真实长度 replay。
3. profiler 证明 D2H 和 synchronize 消失。

### 阶段 E：性能恢复（后续 V3，不属于本次正确性首期）

若需要恢复 S2 跨核，应重新设计 device-side task metadata：

- 根据 `kv_real` 在设备侧生成 S2 split count/workspace index；或
- 固定最大分片数，并为无效分片写 softmax 归约单位元，再让 FD 安全归约；
- 必须证明每个 FD slot 都被初始化，且真实分片数不会导致重复/缺失。

在完成该设计前，禁止在 runtime-KV 模式重新设置 `streamK=true`。

---

## 19. 验收标准

- [ ] 首期 scope 下，新 V6 输出与旧 V5（传真实 CPU list）在容差内一致。
- [ ] `kv_upper > kv_real` 跨 128/256 block 时无越界、无 NaN、无结果偏差。
- [ ] runtime 模式 tiling 永远不生成 FD，所有核边界行对齐。
- [ ] NPU profiler 中无 seqlen D2H、`.tolist()` 或同步等待。
- [ ] ACLGraph 同一张图能 replay 多组 `kv_real`。
- [ ] legacy V1-V5 和非目标场景行为不变。
- [ ] 不支持场景明确 fallback 或报错，不静默使用乐观值作为真值。
- [ ] 单测、算子 ST、vLLM DSpark 回归和真实 NPU 测试全部通过。

---

## 20. 最终建议

V2 应按“**独立 runtime 输入 + 行级静态分核 + kernel 动态 S2**”实现，而不是“Execute 时替换旧输入地址”。该方案利用了当前代码中已经存在的两个事实：

1. PageAttention KV seqlen parser 本来就是 BY_BATCH，可直接读取 `[B]` NPU int64 tensor；
2. `split_core_v2` 在 `streamK=false` 时天然按完整行分配，可以让 CPU KV 上界只影响负载均衡而不影响任务覆盖集合。

这给出了首期可审计的正确性闭环。其性能代价是暂时关闭长 KV 的 S2 跨核和 FlashDecode；该代价应通过性能数据评估，而不能以破坏 host/kernel 一致性为代价规避。
