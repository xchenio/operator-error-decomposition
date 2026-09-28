# 1000 样本调度迁移与训练集 POD 岭回归

## 2026-09-28 锁定协议

本轮只使用现有 train/validation，不加载 test 数组，不新增最终留出集消费。

1. 2 个父 trunk × 5 种子 × 恒定/余弦学习率，共 20 次：零残差初始化，
   1000 样本，batch32，3000 更新，1e-3 恒定或余弦至 1e-5。其余沿用 stability
   协议。恒定组应精确复现 smallfit 的 n=1000、zero-init 指标。
2. 确定性 POD＋岭回归强简单基线：训练目标减去各自输入空间均值，使用其
   **不去列均值**的残差矩阵做 SVD。POD 只接触训练标签。
3. 预先锁定 rank=16/32/64，alpha=1e-4/1e-3/1e-2/1e-1/1/10，共 18 项；
   输入逐传感器均值/标准差只由训练集拟合。截距不惩罚，目标为样本平均
   系数平方误差加 alpha 乘权重 Frobenius 范数平方，正规方程使用 n*alpha。
4. POD 超参数按验证逐样本相对 L2 选择，全部候选都报告。这是验证调参结果，
   不可直接和未调参网络宣称严格公平胜负，更不是新的测试成绩。
5. 同时报告 oracle、实际 train/val、RMSE、相对 Frobenius、基线与学习率配对差值。
   POD 是线性系数映射，不等于完整 POD-DeepONet，也不等于忠实两阶段训练。

神经训练每 50 步断点、250 步评估，学习率由保存的步数重建；闭式岭回归每个
网格项完成后原子保存模型和指标，中断时仅重算小型预处理并跳过已完成项。
所有源码/配置/数据身份均核验；历史脚本、结果与论文主表不覆盖。

```powershell
& D:\miniconda3\envs\pytorch\python.exe -u -B scripts/run_public_scale1000.py
& D:\miniconda3\envs\pytorch\python.exe -u -B scripts/run_public_pod_ridge.py
```

本轮未开展第二 PDE、完整 POD 神经 branch、两阶段方法重现或前瞻诊断选择器。
先完成这些成本低且能改变下一步判断的对照，再决定是否扩大实验范围。
