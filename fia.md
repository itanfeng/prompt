# 一、代码仓信息
这个代码仓是基于华为昇腾NPU的算子代码仓，核心的各种attention算子在attention目录。其中fused_infer_attention_score包含prompt_flash_attention和incre_flash_attention两个算子，分别为profill阶段和decode阶段的attention算子。
# 二、你的任务
仔细分析prompt_flash_attention算子的逻辑，在现有算子基础上，改造该算子，实现attention稀疏化，即
实现BLASST（Blocked Attention Sparsity via Softmax Thresholding）算法，算法的核心在于**动态剪枝**。它通过重用 FlashAttention在在线Softmax过程中已计算的统计信息，实时识别并跳过对最终输出贡献微乎其微的注意力块，从而降低计算和内存负担。
### 以下是BLASST核心算法的具体执行步骤：
1.  **分块与初始化**：将查询（Query）、键（Key）和值（Value）矩阵划分为块。对于每一行查询块 $i$，初始化运行最大值 $m_i$、累加器 $l_i$ 和输出块 $O_i$。
2.  **计算注意力分数**：对于每个键块 $j$，计算注意力分数块 $S_{ij} = Q_i K_j^\top$。
3.  **提取局部与更新运行最大值**：
    *   计算当前块 $S_{ij}$ 的行局部最大值：$\tilde{m}_i^{(j)} = \text{rowmax}(S_{ij})$。
    *   更新该行的全局运行最大值：$m_i^{(j)} = \max(m_i^{(j-1)}, \tilde{m}_i^{(j)})$。
4.  **动态剪枝决策（核心步骤）**：
    *   **判定准则**：检查条件 $\tilde{m}_i^{(j)} - m_i^{(j)} < \ln(\lambda)$，其中 $\lambda$ 是输入的预设阈值。
    *   **执行逻辑**：如果该条件成立，说明当前块的最大注意力分数远小于已知的运行最大值，其指数化后的贡献接近于零（$\approx 0$），算法将直接**跳过（Skip）**该块后续的所有操作。
5.  **选择性计算与累加（非跳过块）**：
    *   计算注意力权重：$P = \exp(S_{ij} - m_i^{(j)})$。
    *   更新累加器 $l_i$ 和输出块 $O_i$（执行 $P_{ij}V_j$ 的矩阵乘法）。
6.  **最终归一化**：处理完所有列块后，通过 $O_i = O_i / l_i$ 进行最终的 Softmax 归一化。
### 算法的关键优化机制
1.  **节省的操作**：一旦触发剪枝，算法会跳过：(1) **指数运算（exp）**；(2) **从显存（HBM）加载值（Value）块**；(3) **注意力与值的矩阵乘法（MMA）**。
2.  **针对性加速**：在**Prefill（预填充）阶段**，主要减少 Tensor Core 的矩阵乘法计算；在**Decode（解码）阶段**，则主要减少 HBM 内存带宽消耗，跳过 $V$ 块加载。
3.  **自动阈值校准**：为了在不同上下文长度 $L$ 下保持稳定的稀疏度，BLASST 使用自动校准程序确定阈值 $\lambda$，其遵循反比关系 $\lambda = a/L$。
# 三、注意事项
在算子改造前，先理清现有算子的运行逻辑，以及对昇腾硬件的使用，以实现最高性能计算，特别是tiling划分、流水排布、cube和vector通信等，再在现有算子基础上进行修改实现。
# 四、昇腾NPU硬件信息
1. 每个昇腾NPU有24个AI Core，计算并行在多个AI Core上进行，每个AI Core包含1个cube单元、2个vector单元和1个scaler单元，cube单元主要用于矩阵计算，vector单元主要用于向量计算，scaler单元主要用于调度。
2. cube和vector之间不能直接通信，需要通过全局内存GM进行数据交换。
3. cube单元的存储包括L1、L0A、L0B和L0C。vector单元的存储为UB，即unified buffer。
4. 更多昇腾NPU的硬件信息，可以查询昇腾官方文档。