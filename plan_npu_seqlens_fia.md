# FIA 算子支持 NPU seq_len Tensor 方案计划

> 目标：让 FusedInferAttentionScore 算子支持 tiling 用 CPU 乐观 seq_lens + kernel 用 NPU 精确 seq_lens tensor，
> 消除 DSpark spec-decode 路径中的 D2H 同步瓶颈。

---

## 1. 白盒分析：CPU seq_len_list vs NPU seq_len_tensor 对算子实现的影响

### 1.1 当前数据流（存在 D2H sync）

```
vllm-ascend (Python)
  seq_lens (NPU int32 tensor)
    ↓ .tolist() / ConvertType()  ← 【D2H sync 发生处】
  actual_seq_qlen: List[int] / c10::ArrayRef<c10::SymInt>
    ↓ PTA 层: ConvertType() → aclIntArray (CPU int64 数组)
    ↓ EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV5, ...)
CANN Runtime
  GetWorkspaceSize(): aclIntArray → tiling 代码读取 CPU 内存 → 生成 tiling data
  Execute():        aclIntArray 数据拷贝到 GM buffer → kernel 从 GM 读
NPU Kernel
  actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, size)
  qActSeqLensParser.Init(actualSeqLengthsGmQ, ...)
  qActSeqLensParser.GetActualSeqLength(bIdx)  → 从 GM 读 uint64_t
```

### 1.2 目标数据流（零 D2H sync）

```
vllm-ascend (Python)
  seq_lens_cpu_upper_bound (CPU int64 tensor)  ← 乐观值，纯 CPU 计算，无同步
  seq_lens (NPU int64 tensor)                  ← 精确值，已在 NPU 上
    ↓ PTA 层: 
    ↓   seq_lens_cpu_upper_bound → aclIntArray (CPU，for tiling)
    ↓   seq_lens_npu             → aclTensor  (NPU，for kernel GM address)
    ↓ 新 API: aclnnFusedInferAttentionScoreV6(..., actualSeqLengthsNpu, ...)
CANN Runtime
  GetWorkspaceSize(): aclIntArray(乐观值) → tiling → workspace/core 分配
  Execute():        用 aclTensor 的 GM 地址替代 CPU 拷贝 buffer → kernel 从 NPU tensor 读
NPU Kernel
  actualSeqLengthsGmQ.SetGlobalBuffer(npu_tensor_gm_addr, size)  ← 直接指向 NPU tensor
  qActSeqLensParser.GetActualSeqLength(bIdx) → 从 NPU tensor 读精确值
```

### 1.3 逐层影响分析

#### Tiling 层（GetWorkspaceSize，CPU 侧）

tiling 代码从 `aclIntArray` 读 seq_lens，用于以下决策：

| 决策项 | 使用乐观值的影响 | 安全性 |
|--------|-----------------|--------|
| `s2Size = max(actual_seq_kvlen)` | 得到更大（或相等）的上界 | **安全**：workspace 分配只多不少 |
| `s1Size = max(actual_seq_qlen)` | 同上 | **安全** |
| workspace 大小计算 | 基于 s1Size*s2Size，分配更大 | **安全** |
| 多核切分 / metadata | 可能分配更多任务槽位，但 kernel 会跳过空槽位 | **安全** |
| KV stride 计算（BSND） | stride 用 s2Size_tiling（上界），需 KV 物理维度匹配 | **需确认**（见 1.4） |

**关键结论**：tiling 层使用乐观值完全安全，不需要修改 tiling 代码逻辑。

#### Kernel 层（Execute，NPU 侧）

kernel 从 GM buffer 读 seq_lens（当前是 CPU 拷贝来的，改为 NPU tensor），用于：

| 使用点 | 文件:行号 | 作用 | 精确值的影响 |
|--------|----------|------|-------------|
| `kvActSeqLensParser.GetActualSeqLength(bIdx)` | fia_kernel_noquant_gqa.h:387 | 决定 S2 循环次数 | 更少循环，跳过 rejected token |
| `qActSeqLensParser.GetActualSeqLength(bIdx)` | fia_kernel_noquant_gqa.h:388 | 决定 S1 循环次数 | 更少循环 |
| `CalcCurS2StartEndNoSparse(actSeqLensKv)` | fia_kernel_noquant_gqa.h:451 | 确定 S2 读写范围 | 正确截断 rejected token |
| `CalcParams: actS1Size/actS2Size` | fia_kernel_noquant_gqa.h:600-601 | RunInfoX 传给计算函数 | 正确的有效范围 |
| `CalcPreNextTokens(actSeqLensQ/Kv)` | fia_block_vec_flashdecode.h:340-341 | sparse mask 边界 | 正确的 mask 范围 |
| `offsetCalculator.Init(..., actualSeqLengthsGmQ)` | fia_block_vec_flashdecode.h:282 | 输出地址计算 | 写入正确位置 |
| `actualSeqQlenAddr[bIdx]` (旧路径) | infer_flash_attention_kvcache.h:120 | Q/KV 偏移 | 正确偏移 |

**关键结论**：kernel 所有计算边界决策都基于从 GM 读的 seq_len，改用精确值后计算结果正确。

#### 1.4 需要满足的前提条件

**条件 1：KV/Q tensor 物理步长一致性**

tiling 用乐观 s2Size 计算 stride（如 BSND 布局 `bIdx * n2Size * s2Size_tiling * dSize`），要求 KV tensor 物理 S2 维度 ≥ s2Size_tiling。

- **PageAttention 模式**（vllm 常用）：KV 通过 block_table 访问，stride 由 block_size 决定，不受 s2Size 影响 → **天然安全**
- **Batch continuous 模式**：需确保 KV tensor 的 S2 padding 到乐观值

对 DSpark decode 场景（s1Size=1, PageAttention），**无需额外处理**。

**条件 2：数据类型匹配**

当前 kernel 代码：
```cpp
actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, size);
```

CANN runtime 从 aclIntArray 拷贝时写为 int64，GM buffer 是 uint64_t。

vllm 的 `seq_lens` 通常是 **int32**。直接指针替换会导致每读一个 uint64_t 实际读到两个 int32 → **数据错误**。

**解决方案**：在 PTA 层或 vllm 侧将 seq_lens 转为 int64：
```python
seq_lens_npu = seq_lens.to(torch.int64)
```

**条件 3：TND/NTD 累积格式**

- `BY_BATCH` 模式（BSND/BNSD）：`GetActualSeqLength(bIdx) = gm[bIdx]` → 直接读 per-batch 值
- `ACCUM` 模式（TND/NTD）：`GetActualSeqLength(bIdx) = gm[bIdx] - gm[bIdx-1]` → 需要 cu_seqlens 格式

DSpark 使用 BSND 布局 → BY_BATCH 模式 → **无需额外转换**。

如果未来需要 TND 布局，NPU tensor 需提供 cu_seqlens 格式（cumsum）。

**条件 4：乐观值与精确值的大小关系**

- 乐观值 = num_computed_tokens_cpu + num_scheduled_tokens（上界）
- 精确值 = num_computed_tokens（减去 rejected tokens）
- **乐观值 ≥ 精确值**，恒成立

这保证 tiling 分配的 workspace 和循环上界充足，kernel 用更小的精确值不会越界。

---

## 2. 算子代码修改点

### 2.1 修改策略：新增 V6 API，保持向后兼容

创建 `aclnnFusedInferAttentionScoreV6`，新增两个可选 `aclTensor*` 参数：

```c
// op_api/aclnn_fused_infer_attention_score_v6.h
aclnnStatus aclnnFusedInferAttentionScoreV6GetWorkspaceSize(
    // ... 与 V5 完全相同的参数 ...
    const aclIntArray *actualSeqLengthsOptional,      // CPU 乐观值（for tiling）
    const aclIntArray *actualSeqLengthsKvOptional,    // CPU 乐观值（for tiling）
    // ... 中间参数不变 ...
    // ↓↓↓ 新增参数 ↓↓↓
    const aclTensor *actualSeqLengthsNpuOptional,     // NPU 精确值（for kernel），可选
    const aclTensor *actualSeqLengthsKvNpuOptional,   // NPU 精确值（for kernel），可选
    // ... 其余参数不变 ...
);

aclnnStatus aclnnFusedInferAttentionScoreV6(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream);
```

**语义**：
- 若 `actualSeqLengthsNpuOptional != nullptr`：kernel 用 NPU tensor 的 GM 地址读 seq_lens
- 若 `actualSeqLengthsNpuOptional == nullptr`：退化为 V5 行为，用 CPU 数组拷贝到 GM

### 2.2 修改文件清单

#### 2.2.1 op_api 层（新增 + 修改）

| 文件 | 修改内容 |
|------|---------|
| `op_api/aclnn_fused_infer_attention_score_v6.h` | **新建**。声明 V6 API，含 NPU tensor 参数 |
| `op_api/aclnn_fused_infer_attention_score_v6.cpp` | **新建**。实现 V6 API。核心改动：<br>1. GetWorkspaceSize：tiling 用 aclIntArray（乐观值），与 V5 逻辑相同<br>2. Execute：判断 NPU tensor 是否存在，若存在则将 executor 中的 actual_seq_lens GM 地址替换为 NPU tensor 的 data pointer |
| `op_api/fused_infer_attention_score_inner.h` | **修改**。Inner 结构体新增 `aclTensor* actualSeqLengthsNpu` 和 `aclTensor* actualSeqLengthsKvNpu` 字段 |

**V6 GetWorkspaceSize 实现要点**：
```cpp
aclnnStatus aclnnFusedInferAttentionScoreV6GetWorkspaceSize(
    ..., const aclTensor *actualSeqLengthsNpuOptional, const aclTensor *actualSeqLengthsKvNpuOptional, ...)
{
    auto executor = CREATE_EXECUTOR();
    // tiling 仍用 CPU aclIntArray（乐观值），不读 NPU tensor
    auto ret = FusedInferAttentionScoreV5Tiling(context, ...);  // 复用 V5 tiling
    
    // 保存 NPU tensor 指针到 executor context，Execute 阶段使用
    if (actualSeqLengthsNpuOptional != nullptr) {
        executor->SetInput("actualSeqLengthsNpu", actualSeqLengthsNpuOptional);
    }
    if (actualSeqLengthsKvNpuOptional != nullptr) {
        executor->SetInput("actualSeqLengthsKvNpu", actualSeqLengthsKvNpuOptional);
    }
    return ret;
}
```

**V6 Execute 实现要点**：
```cpp
aclnnStatus aclnnFusedInferAttentionScoreV6(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    // 获取 NPU tensor 的 GM 地址
    auto *seqQNpu = executor->GetInput<aclTensor*>("actualSeqLengthsNpu");
    auto *seqKvNpu = executor->GetInput<aclTensor*>("actualSeqLengthsKvNpu");
    
    if (seqQNpu != nullptr) {
        // 用 NPU tensor 的 data pointer 替代 CPU 拷贝的 GM buffer
        // 将 seqQNpu->data 曫换 kernel 入参中的 actualSeqLengths 指针
        executor->OverrideInputTensorAddr(ACTUAL_SEQ_Q_INDEX, seqQNpu->data);
    }
    // 同理处理 seqKvNpu
    
    return aclnnFusedInferAttentionScoreV5(workspace, workspaceSize, executor, stream);
}
```

> **注**：具体 OverrideInputTensorAddr 的实现方式取决于 CANN executor 的 API。
> 另一种更简单的方案：直接修改 kernel 入参传递逻辑，在 Execute 时判断 NPU tensor 是否存在，
> 若存在则将 kernel 的 actualSeqLengths 参数替换为 NPU tensor 的 GM 地址。

#### 2.2.2 op_host / tiling 层（无需修改）

tiling 层所有代码继续使用 CPU 侧的 aclIntArray（乐观值），**零改动**。

#### 2.2.3 op_kernel 层（无需修改）

kernel 代码已通过 `ActualSeqLensParser` 从 GM 读 seq_lens，GM 地址由 op_api 层传入。
只要 GM 地址指向正确的数据（NPU tensor），kernel 自动使用精确值。**零改动**。

#### 2.2.4 PTA 层（vllm-ascend 侧）

| 文件 | 修改内容 |
|------|---------|
| `csrc/attention/fused_infer_attention_score/fia_torch_adpt.h` | **新建**。PyTorch adapter，调用 V6 API |
| `csrc/torch_binding.cpp` | **修改**。注册 `npu_fused_infer_attention_score_v6` 自定义 op |
| `vllm_ascend/attention/attention_v1.py` | **修改**。DSSpark 路径调用 V6 op，传入 seq_lens_cpu_upper_bound + seq_lens_npu |

**fia_torch_adpt.h 核心逻辑**：
```cpp
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v6(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    at::IntArrayRef actual_seq_qlen,       // CPU 乐观值
    at::IntArrayRef actual_seq_kvlen,      // CPU 乐观值
    const c10::optional<at::Tensor> &actual_seq_qlen_npu,  // NPU 精确值
    const c10::optional<at::Tensor> &actual_seq_kvlen_npu,  // NPU 精确值
    ... /* 其余参数同 FIA v2 */)
{
    // 转换 CPU seq_lens 为 aclIntArray（for tiling）
    auto acl_seq_qlen = ConvertType(actual_seq_qlen);
    auto acl_seq_kvlen = ConvertType(actual_seq_kvlen);
    
    // 转换 NPU seq_lens tensor 为 aclTensor（for kernel）
    // 关键：确保数据类型为 int64
    auto seq_qlen_npu_int64 = actual_seq_qlen_npu->to(at::kLong);
    auto acl_seq_qlen_npu = ConvertType(seq_qlen_npu_int64.value());
    auto seq_kvlen_npu_int64 = actual_seq_kvlen_npu->to(at::kLong);
    auto acl_seq_kvlen_npu = ConvertType(seq_kvlen_npu_int64.value());
    
    EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV6, query, key, value, ...,
                 acl_seq_qlen, acl_seq_kvlen, ...,
                 acl_seq_qlen_npu, acl_seq_kvlen_npu, ...);
}
```

**attention_v1.py DSpark 路径修改**：
```python
# 当前（有 D2H sync）:
seq_lens_list = seq_lens.tolist()  # ← 同步!
fia_params = _get_fia_params(seq_lens_list, ...)

# 修改后（零 D2H sync）:
seq_lens_cpu_upper_bound = attn_metadata.seq_lens_cpu_upper_bound  # CPU 乐观值，已计算好
seq_lens_npu = seq_lens[:num_reqs].to(torch.int64)                # NPU 精确值，零拷贝
fia_params = _get_fia_params_v6(seq_lens_cpu_upper_bound, seq_lens_npu, ...)
```

---

## 3. 编译和迁移至 vllm-ascend/csrc

### 3.1 迁移必要性分析

有两种路径：

| 路径 | 描述 | 优缺点 |
|------|------|--------|
| **A. 完整迁移 FIA 到 csrc/** | 将 FIA 全套源码（op_host + op_kernel + op_api）复制到 csrc/attention/fused_infer_attention_score/ | 优点：完全自主可控<br>缺点：依赖链庞大（~13MB 源码 + 3MB common + 1.2MB incre/prompt_flash_attention） |
| **B. 仅迁移 op_api 层 + 新增 V6** | FIA 的 tiling 和 kernel 用系统已安装的 CANN 内置版本，只新建 op_api V6 wrapper | 优点：改动最小<br>缺点：依赖 CANN 内置 FIA 的 kernel 二进制，无法修改 kernel |

**推荐路径 A**（完整迁移），理由：
1. 完全控制 kernel 行为，未来可做更多优化
2. vllm-ascend 已有完整的 CANN op 构建基础设施，迁移成本可控
3. 系统内置 FIA 可能版本不匹配

### 3.2 需要复制的文件清单

#### 3.2.1 FIA 核心代码

```bash
# 源：/opt/z00830407/zsy_ops/ops-transformer/attention/
# 目标：/vllm-workspace/vllm-ascend/csrc/attention/

# FIA 主目录
cp -r fused_infer_attention_score/  csrc/attention/fused_infer_attention_score/

# FIA 依赖的 common 目录（共享 tiling / kernel 代码）
cp -r common/  csrc/attention/common/

# FIA 依赖的 incre_flash_attention（tiling 引用）
cp -r incre_flash_attention/op_host/ csrc/attention/incre_flash_attention/op_host/

# FIA 依赖的 prompt_flash_attention（tiling 引用）
cp -r prompt_flash_attention/op_host/ csrc/attention/prompt_flash_attention/op_host/
```

**文件量估算**：

| 目录 | 大小 | 说明 |
|------|------|------|
| fused_infer_attention_score/ | ~9.5MB | 含 op_host, op_kernel, op_api, tests |
| common/ | ~2.9MB | 共享 tiling + kernel 代码 |
| incre_flash_attention/op_host/ | ~436KB | tiling 依赖 |
| prompt_flash_attention/op_host/ | ~716KB | tiling 依赖 |
| **合计** | **~13.5MB** | |

#### 3.2.2 精简策略（可选，降低迁移体积）

若需精简，可只保留 arch35 路径（Ascend910B/950），删除 arch22 和 arch38：

```bash
# 删除非 arch35 的 kernel
rm -rf csrc/attention/fused_infer_attention_score/op_kernel/arch22/
rm -rf csrc/attention/fused_infer_attention_score/op_kernel/arch38/

# 删除非 arch35 的 tiling
rm -rf csrc/attention/fused_infer_attention_score/op_host/arch22/
rm -rf csrc/attention/fused_infer_attention_score/op_host/arch38/

# 删除非 arch35 的 common kernel
rm -rf csrc/attention/common/op_kernel/arch22/
```

**精简后约 ~8MB**。

### 3.3 额外需要的头文件

#### 3.3.1 CANN 系统头文件（无需复制，从 CANN 安装目录引用）

这些头文件由 CANN 安装包提供，构建时通过 `ASCEND_HOME_PATH` 定位：

| 头文件路径 | 来源 |
|-----------|------|
| `tiling/tiling_api.h` | CANN tiling 框架 |
| `register/op_def_registry.h` | CANN 注册框架 |
| `register/tilingdata_base.h` | CANN 注册框架 |
| `platform/platform_info.h` | CANN 平台信息 |
| `log/log.h`, `log/error_code.h` | CANN 日志 |
| `err/ops_err.h` | CANN 错误处理 |
| `kernel_operator.h` | CANN AscendC |
| `kernel_vec_intf.h`, `kernel_cube_intf.h` | CANN AscendC (≥9) |
| `aclnn/acl_meta.h` | CANN ACLNN |
| `opdev/*.h` | CANN op-dev |

**vllm-ascend 的 CMakeLists.txt 已正确设置 `ASCEND_HOME_PATH`，无需额外配置。**

#### 3.3.2 项目内部头文件（已在复制范围内）

所有 `#include "../../common/op_host/..."` 和 `#include "../../common/op_kernel/..."` 的引用，
只要保持相对路径不变，复制后即可正常工作。

#### 3.3.3 无需 torch-npu 头文件

FIA 的 op_api/op_host/op_kernel 代码不依赖 `torch_npu/csrc/` 的任何头文件。
torch-npu 桥接代码（`fia_torch_adpt.h`）由 vllm-ascend 自行编写，
使用 vllm-ascend 已有的 `csrc/aclnn_torch_adapter/op_api_common.h` 中的 `EXEC_NPU_CMD` 宏。

### 3.4 构建配置修改

#### 3.4.1 CMakeLists.txt

FIA 已自带 `CMakeLists.txt`（在 `op_host/CMakeLists.txt`），
格式与 vllm-ascend 中 `sparse_flash_attention` 的 CMakeLists.txt 一致，
使用 `add_op_to_compiled_list()` + `add_ops_compile_options()` + `add_modules_sources()` 模式。

**可能需要的修改**：

```cmake
# csrc/attention/fused_infer_attention_score/op_host/CMakeLists.txt
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(fused_infer_attention_score_depends attention/common CACHE INTERNAL "Dependencies")
    target_sources(op_host_aclnnInner PRIVATE
        fused_infer_attention_score_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedInferAttentionScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

# 注意：FIA 使用 aclnn_inner 类型（不是 aclnn_exclude）
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(
        OPTYPE fused_infer_attention_score
        ACLNNTYPE aclnn_inner
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    )
endif()
```

#### 3.4.2 build_aclnn.sh

在 ascend910b 的 `CUSTOM_OPS_ARRAY` 中添加 `fused_infer_attention_score`：

```bash
# csrc/build_aclnn.sh, ascend910b 分支
CUSTOM_OPS_ARRAY=(
    # ... 现有 ops ...
    "fused_infer_attention_score"    # ← 新增
)
```

对 ascend910_93 和 ascend950 的分支也需添加。

#### 3.4.3 torch_binding.cpp

注册 V6 op：

```cpp
// csrc/torch_binding.cpp
#include "attention/fused_infer_attention_score/fia_torch_adpt.h"

// 在 TORCH_LIBRARY_IMPL 块中添加：
ops.def(
    "npu_fused_infer_attention_score_v6("
    "Tensor query, Tensor key, Tensor value, "
    "int[] actual_seq_qlen, int[] actual_seq_kvlen, "
    "Tensor? actual_seq_qlen_npu=None, Tensor? actual_seq_kvlen_npu=None, "
    "Tensor? pse_shift=None, Tensor? atten_mask=None, "
    "float scale_value=1.0, str layout='BSND', "
    "int num_heads=0, int num_kv_heads=0, "
    "int sparse_mode=0, int pre_tokens=2147483647, int next_tokens=2147483647, "
    "int block_size=0, Tensor? block_table=None, "
    "Tensor? query_rope=None, Tensor? key_rope=None, "
    "bool softmax_lse_flag=False) -> (Tensor attention_out, Tensor softmax_lse)"
);
ops.impl("npu_fused_infer_attention_score_v6", torch::kPrivateUse1,
         &vllm_ascend::npu_fused_infer_attention_score_v6);
```

### 3.5 编译命令

```bash
# 完整重新编译（推荐，确保所有自定义 op 重建）
cd /vllm-workspace/vllm-ascend
pip install -e . --no-build-isolation

# 或仅重建 CANN 自定义 op（不重编译 pybind 模块）
python setup.py build_aclnn

# 关键环境变量：
# ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest  (CANN 安装路径)
# SOC_VERSION=ascend910b1  (自动检测，或手动设置)
# MAX_JOBS=32  (并行编译线程数)
# COMPILE_CUSTOM_KERNELS=1  (默认开启)
```

**编译耗时估算**：FIA 是最大的 CANN 算子之一（kernel 代码 ~300KB+），编译时间约 10-20 分钟（取决于核数）。

### 3.6 运行时部署

编译成功后，CANN 自定义 op 安装到：
```
vllm_ascend/_cann_ops_custom/vendors/custom_transformer/
  op_api/lib/libcust_opapi.so   ← 包含 aclnnFusedInferAttentionScoreV6
  op_kernel/binary/             ← kernel 二进制
```

vllm-ascend 启动时自动将此路径加入 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`，
`EXEC_NPU_CMD` 宏会优先从此路径的 `libcust_opapi.so` 查找符号。

---

## 4. 验证方案

### 4.1 验证思路

**核心命题**：tiling 用乐观 seq_lens + kernel 用精确 seq_lens 的结果 = 全程用精确 seq_lens 的结果。

即：`FIA_V6(optimistic_cpu, precise_npu) == FIA_V5(precise_cpu, precise_cpu)`

### 4.2 测试构造

#### 4.2.1 Decode 场景（DSPark 核心场景）

```python
import torch
import torch_npu

batch_size = 4
num_heads = 32
num_kv_heads = 8   # GQA
head_dim = 128
kv_cache_len = [100, 200, 150, 80]  # 各 batch 的 KV cache 长度（精确值）
rejected_tokens = [2, 0, 3, 1]       # 各 batch 被 reject 的 token 数
optimistic_kv_len = [l + r for l, r in zip(kv_cache_len, rejected_tokens)]  # 乐观值

# 构造 Q (decode: s1=1)
query = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16).npu()

# 构造 KV cache (BSND layout, padded to max optimistic length)
max_kv_len = max(optimistic_kv_len)
key = torch.randn(batch_size, num_kv_heads, max_kv_len, head_dim, dtype=torch.float16).npu()
value = torch.randn(batch_size, num_kv_heads, max_kv_len, head_dim, dtype=torch.float16).npu()

# 构造 block_table (PageAttention)
block_size = 16
max_blocks = (max_kv_len + block_size - 1) // block_size
block_table = torch.randint(0, 100, (batch_size, max_blocks), dtype=torch.int32).npu()

# 精确 seq_lens (NPU tensor, int64)
seq_lens_kv_precise = torch.tensor(kv_cache_len, dtype=torch.int64).npu()
seq_lens_q_precise = torch.ones(batch_size, dtype=torch.int64).npu()

# 乐观 seq_lens (CPU list)
seq_lens_kv_optimistic = optimistic_kv_len
seq_lens_q_optimistic = [1] * batch_size
```

#### 4.2.2 对比测试

```python
# 基准：V5 API，全程使用精确 seq_lens
ref_out, ref_lse = npu_fused_infer_attention_score_v5(
    query, key, value,
    actual_seq_qlen=kv_cache_len,           # CPU 精确值
    actual_seq_kvlen=kv_cache_len,          # CPU 精确值
    block_table=block_table,
    scale_value=1.0 / (head_dim ** 0.5),
    layout='BSND',
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    sparse_mode=0,
    block_size=block_size,
)

# 实验组：V6 API，tiling 用乐观值 + kernel 用精确 NPU tensor
test_out, test_lse = npu_fused_infer_attention_score_v6(
    query, key, value,
    actual_seq_qlen=seq_lens_q_optimistic,          # CPU 乐观值
    actual_seq_kvlen=seq_lens_kv_optimistic,         # CPU 乐观值
    actual_seq_qlen_npu=seq_lens_q_precise,          # NPU 精确值
    actual_seq_kvlen_npu=seq_lens_kv_precise,        # NPU 精确值
    block_table=block_table,
    scale_value=1.0 / (head_dim ** 0.5),
    layout='BSND',
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    sparse_mode=0,
    block_size=block_size,
)

# 对比
atol = 1e-3  # fp16 精度
rtol = 1e-3
assert torch.allclose(ref_out, test_out, atol=atol, rtol=rtol), \
    f"Output mismatch! max_diff={torch.abs(ref_out - test_out).max()}"
print("PASSED: V6(optimistic+precise) matches V5(precise)")
```

### 4.3 测试矩阵

| 测试编号 | 场景 | batch_size | GQA | sparse_mode | rejected | 预期 |
|---------|------|-----------|-----|-------------|----------|------|
| T1 | Decode, BSND, no mask | 4 | 32:8 | 0 (noSparse) | [2,0,3,1] | 结果一致 |
| T2 | Decode, BSND, causal | 4 | 32:8 | 3 (rightDownCausal) | [1,1,0,2] | 结果一致 |
| T3 | Decode, BSND, band | 4 | 32:8 | 4 (band) | [0,0,0,0] | 结果一致 |
| T4 | Prefill, BSND | 2 | 8:8 | 0 | [0,0] | 结果一致 |
| T5 | Decode, MHA (32:32) | 4 | 32:32 | 0 | [3,2,1,0] | 结果一致 |
| T6 | 极端：全 rejected | 2 | 32:8 | 0 | [10,10] | 结果一致（空输出） |
| T7 | 极端：0 rejected | 2 | 32:8 | 0 | [0,0] | 结果一致（=V5 退化为 V5） |
| T8 | V6 不传 NPU tensor | 2 | 32:8 | 0 | - | 结果 = V5（向后兼容） |

### 4.4 精度预期

- FP16 attention 的计算本身有 ~1e-3 量级的数值误差
- 同一算子、同一输入、不同 seq_lens 路径的计算顺序完全一致（因为 kernel 代码不变，只是 GM 数据源不同）
- **预期最大误差 < 1e-3**（FP16 精度范围内），多数情况为 0（bit-exact）

### 4.5 性能验证

```python
import time

# 1. 测量 D2H sync 时间（当前 V5 路径）
torch.npu.synchronize()
t0 = time.perf_counter()
seq_lens_list = seq_lens.tolist()  # 触发 D2H sync
torch.npu.synchronize()
t1 = time.perf_counter()
d2h_latency_ms = (t1 - t0) * 1000
print(f"D2H sync latency: {d2h_latency_ms:.2f} ms")

# 2. V6 路径无 D2H sync
# 对比 V5 和 V6 的端到端 attention 执行时间
# 预期：V6 比 V5 快约 d2h_latency_ms（消除了同步等待）
```

### 4.6 端到端 DSpark 验证

在 DSpark spec-decode 流程中替换 attention 调用为 V6，对比：
1. **正确性**：DSpark + V6 的 draft token 验证通过率应与 V5 一致
2. **性能**：DSPark + V6 的端到端吞吐量提升 ≈ 消除的 D2H sync 时间
3. **ACLGraph 兼容性**：确认 V6 op 可被 ACLGraph 正常 capture（NPU tensor 输入对 graph capture 友好）

---

## 5. 风险与注意事项

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| FIA 依赖链过长，编译失败 | 中 | 先用精简版（仅 arch35 non-quant），逐步扩展 |
| CANN executor API 不支持 OverrideInputTensorAddr | 高 | 备选方案：直接修改 inner 层的 kernel 入参传递逻辑，新增 NPU tensor 参数 |
| vllm-ascend csrc 的 CANN cmake 版本与 ops-transformer 不兼容 | 中 | vllm-ascend 的 csrc/CMakeLists.txt 是 ops-transformer 的 fork，已适配；检查 SOC arch 映射（910b→arch32 vs arch22） |
| int64 数据类型转换引入额外拷贝 | 低 | seq_lens tensor 很小（batch_size 个 int64），to(torch.int64) 开销可忽略 |
| ACLGraph capture 时 NPU tensor 输入行为不同 | 中 | NPU tensor 输入对 graph capture 更友好（无需 D2H sync），但需验证 V6 op 的 GetWorkspaceSize 在 capture 模式下行为正确 |

---

## 6. 实施步骤

### Phase 1：验证概念（1-2 天）
1. 在 `/opt/z00830407/zsy_ops/ops-transformer/` 中新建 V6 op_api 文件
2. 修改 inner 层支持 NPU tensor seq_lens 传递
3. 用 ops-transformer 自带的编译框架编译
4. 用 CANN 单算子测试框架验证正确性

### Phase 2：迁移到 vllm-ascend（2-3 天）
1. 复制 FIA + 依赖到 `csrc/attention/`
2. 修改 CMakeLists.txt 和 build_aclnn.sh
3. 编写 `fia_torch_adpt.h` 和 `torch_binding.cpp` 注册
4. `pip install -e .` 编译验证

### Phase 3：集成与测试（1-2 天）
1. 修改 `attention_v1.py` DSpark 路径调用 V6
2. 运行 4.3 的测试矩阵
3. 端到端 DSpark 性能测试
4. ACLGraph capture 兼容性测试
