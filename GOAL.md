# 目的
fused_infer_attention_score算子skip(Blocked Attention Sparsity via Softmax Thresholding)功能开发

## 一、代码分支
1. 基线分支：8.5.0-base，对应commit为da6cbf83502727d4fdcb086eef8ce8d514dc6a63
2. skip功能发开分支：8.5.0-skip

## 二、功能脚本
1. 与远程服务器交互的功能shell脚本在./scripts/dev目录下面，阅读[README](README.ascend_remote.md)获得各个shell脚本的功能和使用方式，主要实现代码同步、算子编译、执行测试、拉回日志等功能
2. fused_infer_attention_score算子skip功能测试通过torch_npu.npu_fused_infer_attention_score接口测试，测试脚本是[fia_mla_decode.py](./attention/common/tests/fia_mla_decode.py)

## 三、相关Kernel源代码
1. fia_mla_decode.py运行时走的算子源码路径为./attention/common
2. 最直接相关的kernel源码为主模板[fia_kernel_nonquant_mla.h](./attention/common/op_kernel/arch32/fia_kernel_nonquant_mla.h)、vector模板[fia_block_vec_nonquant_mla.h](./attention/common/op_kernel/arch32/fia_block_vec_nonquant_mla.h)和cube模板[fia_block_cube_nonquant_mla.h](./attention/common/op_kernel/arch32/fia_block_cube_nonquant_mla.h)

## 四、硬件信息
1. 昇腾910B2 NPU，有24个aicore，每个aicore有1个cube核和2个vector核
2. cube核执行矩阵乘计算，片上内存为L0A,L0B,L0C,L1等
3. vector核执行向量计算，片上内存为Unified Buffer，即UB，大小为192KB
4. cube核和vector核通过Global Memory，即GM进行数据交换。

## 五、skip算法细节
1. 结合基线分支的相关源代码和硬件信息，通过skip block对fused_infer_attention_score算子进行稀疏，提高性能，降低时延。
2. 算法核心在于动态skip，通过重用FlashAttention在在线Softmax过程中已计算的统计信息，实时识别并跳过对最终输出贡献微乎其微的注意力block，从而减少计算量。算法的具体执行步骤：
    - 分块与初始化：将Query、Key和Value矩阵划分为块。对于每一行查询块 $i$，初始化运行最大值 $m_i$、累加器 $l_i$ 和输出块 $O_i$。
    - 计算注意力分数：对于每个键块 $j$，计算注意力分数块 $S_{ij} = Q_i K_j^\top$。
    - 提取局部与更新运行最大值：
        *   计算当前块 $S_{ij}$ 的行局部最大值：$\tilde{m}_i^{(j)} = \text{rowmax}(S_{ij})$。
        *   更新该行的全局运行最大值：$m_i^{(j)} = \max(m_i^{(j-1)}, \tilde{m}_i^{(j)})$。
    - 动态跳过决策（核心步骤）：
        *   判定准则：检查条件 $\tilde{m}_i^{(j)} - m_i^{(j)} < \ln(\lambda)$，其中 $\lambda$ 是输入的预设阈值。
        *   执行逻辑：如果该条件成立，说明当前块的最大注意力分数远小于已知的运行最大值，其指数化后的贡献接近于零（$\approx 0$），算法将直接跳过（Skip）该块后续的所有操作。
    - 选择性计算与累加（非跳过块）：
        *   计算注意力权重：$P = \exp(S_{ij} - m_i^{(j)})$。
        *   更新累加器 $l_i$ 和输出块 $O_i$（执行 $P_{ij}V_j$ 的矩阵乘法）。
    -  最终归一化：处理完所有列块后，通过 $O_i = O_i / l_i$ 进行最终的 Softmax 归一化。
3. 算法的关键优化：一旦触发skip，算法会跳过：(1) 指数运算（exp）；(2) 从HBM加载Value块；(3) P与Value的矩阵乘法。

## 六、Kernel执行时序建模（必须执行）

在对skip功能发开分支进行任何功能修改（包括skip功能）前，必须严格执行以下步骤：

1. 完整阅读基线分支以下核心文件：
    - fia_kernel_nonquant_mla.h
    - fia_block_vec_nonquant_mla.h
    - fia_block_cube_nonquant_mla.h
    - fia_tiling_nonquant_mla.cpp
    - fia_tiling_nonquant_mla.h

2. 输出基线分支算子的执行模型分析报告，保存为./scripts/dev/REPORT.md，必须包含：

    1) 循环结构建模
        - Q block 循环层级
        - K/V block 循环层级
        - 每层循环对应的 block 维度
        - cube 与 vector 的调用顺序

    2) 执行阶段划分（按时间顺序）
        
        明确列出：
        - Sij 计算阶段
        - rowmax 阶段
        - m_i 更新阶段
        - exp 阶段
        - P*V 阶段
        - l_i 更新阶段
        - 归一化阶段

        并标明每个阶段：
        - 所在文件
        - 所在函数
        - 所在循环

    3) 内存流建模

        - Q / K / V 的来源（GM → L1 → L0 / UB）
        - Sij 存储位置
        - m_i / l_i 存储位置
        - TQue 的申请与释放位置
        - flag / pipe 同步位置

    4) Pipeline 与资源建模

        - 当前 TPipe stage 数
        - TQue 使用数量
        - 是否存在双缓冲
        - cube 与 vector 是否并行

3. 在完成上述分析之前，禁止修改任何 kernel 代码。

4. skip 插入点必须满足以下约束：

    - 位于 rowmax 计算完成之后
    - 位于 exp 计算之前
    - 不新增 TQue
    - 不改变 TPipe stage 数
    - 不破坏 cube/vector pipeline 对齐

5. 只有在分析报告明确通过后，才允许进行 skip 功能代码修改。

6. skip 功能实现过程中：

    - 不允许修改 tiling 策略
    - 不允许修改 block 维度
    - 不允许修改 cube / vector 调度顺序
    - skip 只能通过“条件执行”实现，而不能通过改变计算图结构实现。

## 七、调试规则
1. 基于基线分支分支采用ascend_fused_infer_attention_score_decode_cycle.sh脚本运行完整的基线算子的代码同步、算子编译、执行测试、拉回日志，记录基线分支算子时延
2. 基于skip功能发开分支，添加skip功能，添加printf打印，方便调试，再采用ascend_fused_infer_attention_score_decode_cycle.sh脚本运行代码同步、编译算子、执行测试。这个阶段最难，可能会往复很多次，可能会在编译算子阶段失败，可能会在执行测试阶段失败，注意观察日志修复，使得skip功能正常。
3. skip功能完全正常时，关掉skip功能发开分支中的所有printf打印，再跑一次完整代码同步、算子编译、执行测试、拉回日志，记录skip功能开发分支算子时延
4. 如果skip功能开发分支相比基线分支时延降低，则结束，否则继续优化skip功能开发分支

## 八、错误码debug建议
1. 507046
    - 检查是否有TQue申请的内存未释放
    - 添加printf打印，确认代码执行阻塞位置，再具体分析
    - 构造相同输入，通过孪生调试功能的CPU侧调试进行定位，该方式可以直接显示错误原因

