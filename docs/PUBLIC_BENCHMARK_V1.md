# Matched public-data benchmark v1

## Locked protocol

- Dataset: PDEBench Burgers Nu0.001, revision `97bf5bb5315b9a083b4f9fc03e5c84d35286d52c`.
- SHA-256 independently rechecked against publisher: `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d`.
- HDF5 solution shape: `(10000, 201, 1024)`, float32. Cell-center x grid.
- Time coordinates have 202 entries; solution has 201. The upstream generator allocates
  one extra time coordinate. Index 100 is t=1.0; the unused final coordinate t=2.01 is excluded.
- The generator uses diffusion epsilon/pi, while the file Nu attribute is epsilon=0.001.
  This first experiment uses supervised values only; it makes no claim about physical residuals.
- Initial field (index 0) -> field at t=1.0 (index 100). Both have 256 points, stride 4,
  point subsampling without filtering. No temporal rollout or unseen-resolution evaluation.
- Fixed random split from all 10000 trajectories: 1000 train, 200 validation, 200 test;
  split seed 20260922. Exact IDs and data hash recorded in protocol.json.
- This is a project-specific public-data subset, not a reproduction of official benchmark scores.
- Input/target scalar normalization is fitted only on their respective training arrays.

## Comparisons

DeepONet MLP trunk, DeepONet Fourier-feature trunk, and FNO1d. All receive the same inputs,
coordinates, outputs, sample order for the same seed, batch size 32, Adam lr=1e-3,
1500 updates, and five training seeds 0..4. Use the final checkpoint; validation is logged
every 250 updates and does not select the test checkpoint. Architectures have different
parameter counts and step costs; report parameter count and measured GPU update time.
Equal update counts are NOT equal-compute evidence. No significance claim from this pilot.

DeepONet: 64 features, branch/trunk hidden width 128, three hidden layers.
FNO: channel width 32, 16 spectral modes, four blocks; physical cell-center coordinates.
Complex FNO parameters count as two real degrees of freedom.

Projection diagnostics apply to DeepONet only: same full 256-point grid and same split,
complete target matrix, additional constant column, offline per-sample coefficient relaxation.
Record SVD tolerance 1e-12 and sensitivity at 1e-6. No shared-trunk bound is assigned to FNO.

## Execute and recover

```powershell
& 'D:\miniconda3\envs\pytorch\python.exe' -u -B scripts/run_public_benchmark.py
```

Rerun the identical command after interruption: completed results are identity-checked and
skipped; unfinished models resume from checkpoints every 50 updates. Random state, optimizer,
history and cumulative update time are restored. Source/data/config/environment mismatch rejects
reuse. Checkpoint and status writes are atomic. The latest uncheckpointed updates may be lost.
Results and events are versioned; intermediate model checkpoints stay local under checkpoints/.

Source references (consulted for metadata semantics, not copied implementation):
- https://huggingface.co/datasets/pdebench/Burgers/tree/main
- https://github.com/pdebench/PDEBench/blob/main/pdebench/data_gen/data_gen_NLE/BurgersEq/burgers_multi_solution_Hydra.py

## Next scientific step

Use these same held-out IDs to test whether an early frozen-trunk coefficient probe predicts
the benefit of changing the basis. Physical training and a second PDE remain separate next
experiments; fitting these supervised controls does not establish PINO performance.

## 实测结果：2026-09-22，15 / 15 已完成

下表为测试集逐样本相对 L2 误差的五个训练种子均值 ± 样本标准差，越低越好。
这不是跨数据划分的置信区间。所有模型使用相同的 1500 次更新。

| 模型 / 无训练基线 | 测试误差 | 实参数量 | 平均更新耗时（秒） |
|---|---:|---:|---:|
| 初值空间均值（每条轨迹一个常数） | 0.37577 | 0 | — |
| 普通 DeepONet | 0.39357 ± 0.01014 | 115713 | 6.07 |
| Fourier DeepONet | 0.41231 ± 0.01344 | 131969 | 6.06 |
| FNO | 0.22835 ± 0.00680 | 136481 | 13.98 |

另外，保持初值不变的基线误差为 1.35527；训练集目标均值场基线为 0.99939。
更新耗时不含验证、断点保存和最终诊断，包含首次 CUDA 初始化影响；不同种子的
机器状态也有波动。它不是完整墙钟时间，也不能据此声称严格同算力胜出。

结论：在这个短预算监督实验中，FNO 是最好的已测试模型；两个 DeepONet 都未超过
初值均值基线。不能将此前小规模自生成数据上的 Fourier 优势直接推广到公开数据。

### 表示诊断与研究方向

| DeepONet trunk | 实际测试误差 | 离线投影误差，rcond=1e-12 | 敏感性：rcond=1e-6 |
|---|---:|---:|---:|
| 普通 MLP | 0.39357 | 0.19177 | 0.25818 |
| Fourier | 0.41231 | 0.04770 | 0.05937 |

投影使用测试标签求逐样本最优系数，并加入自由常数列；不是可部署模型的预测结果。
MLP 数值秩从 65 降到 4–5，Fourier 从 65 降到 34–44，说明容差敏感性不可省略。
严格来说，这里诊断的是离散浮点特征矩阵，而不是无限精度解析函数空间。
误差平方分解的表示项占比分别约 21.3% 和 1.5%；在 1e-6 容差下约 39.5% 和 2.2%。
这些是 Frobenius 平方误差的比例，不是表中相对 L2 均值的直接比例。

Fourier 空间可以更好地拟合标签，但端到端 branch 没有实现这种潜力。最有价值的
下一阶段是验证“固定表示下，系数学习 / 数值条件是否构成主要可修复障碍”，不是
简单扩展参数量，也不是把投影差距直接命名为纯优化误差。

建议锁定以下消融，全部只用训练标签拟合，在验证集选方案：

1. 固定已学 trunk，比较原始特征坐标与截断 SVD 正交坐标，保持 branch 容量、
   数据和训练预算一致；同时报告条件数、有效秩和投影误差。
2. 用初值空间均值作显式输出基线，branch 只学习零均值残差；与不做该处理的
   对照匹配预算。这需要单独检查降采样误差，不能宣称离散质量严格守恒。
3. 增加较长训练预算的对照，区分短训不足与结构问题；FNO 保持外部参照。
4. 本次测试集已用于探索性诊断，后续最终确认应锁定未使用的轨迹 ID，避免
   反复根据这 200 条测试样本选择方案。之后再做第二类 PDE 与物理残差训练。

### 产物与恢复验证

- `results/public_fixed_t1_v1/summary.json`：汇总、每个种子的数值、基线和共同身份。
- 每个模型目录包含 manifest、result、progress、events；本地 checkpoints 包含
  模型、优化器、随机状态和累计步数。数据和大权重不写入 Git。
- CPU 与 CUDA 上均通过 DeepONet / FNO 强制中断对照：最终模型逐张量完全一致，
  全部评价指标一致。测试是 4 步合成数据集成测试，不是实际断电硬件实验。
- 原正式命令再次执行后，15 项全部通过身份核验并跳过，没有重复训练。
- 本轮汇总命令：`python -B scripts/summarize_public_benchmark.py`。
