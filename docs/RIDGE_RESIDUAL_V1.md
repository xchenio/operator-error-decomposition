# 冻结岭回归＋受约束残差：分析与预先锁定的实验

## 研究判断（2026-09-28，确认数据揭盲前）

上一轮并非优化完全无效：非线性 branch 的训练相对 L2 从约0.388降到0.234，
但验证从约0.377升到0.401。需要同时解决剩余训练拟合与泛化问题，不能把平方
系数误差直接称为纯优化误差。先扩大网络或 POD 秩不是目前证据最支持的路线。

本轮目标：让非线性分支只修正冻结线性模型的残差，并检验其是否能在新轨迹上
带来可确认收益。即使成功，也不自动构成新方法论文。

### 新颖性边界与可探索的贡献

- POD 与 DeepONet 残差学习已有直接相关工作：
  [Demo, Tezzele & Rozza, 2023](https://arxiv.org/abs/2302.12682)。
  该文将降阶近似与学习残差结合；我们的固定基系数修正并非完全相同实现，
  但不能笼统宣称首次提出 POD＋非线性残差。
- 零初始化残差也有先例，例如
  [ReZero, UAI2021](https://proceedings.mlr.press/v161/bachlechner21a.html)。
  本项目输出层归零不同于该文零门控，仍不能把零残差概念作为独立原创贡献。
- 候选创新方向：**按可学习性而非仅按 POD 能量分配非线性修正**。
  将每模态的系数预测增益及跨训练折稳定性用于收缩或关闭不可靠修正；
  规则仅由开发数据拟合，再跨独立划分、数据量和至少第二种 PDE 验证。
  这是尚未实现的研究假设，未完成穷尽性新颖性检索，不能写成已证实贡献。
- 优先比较诊断驱动规则、统一正则化、随机模态选择和同预算验证搜索。
  若只能在当前反复使用的验证集取胜，应保持诊断案例论文定位。

## Locked protocol: public_ridge_residual_v1

1. Reuse the exact rank16/alpha10 checkpoint from public_pod_ridge_v1. Freeze its Q,
   input normalization, weights and intercept. All 1000 training / 200 development
   trajectories and time/grid semantics are unchanged. No old test arrays are loaded.
2. Coefficients: c(x)=c_ridge(x)+s*f_theta(normalized x), where s is the training target
   standard deviation. f is one tanh hidden layer, width16 or32, with zero final weights
   and bias. The initial prediction equals ridge exactly; no extra mean shift is learned.
3. Eight settings: width in {16,32} × correction penalty lambda in {0,0.1,1,10}.
   Five seeds0–4 per setting, paired minibatches, AdamW weight_decay=1e-4 fixed,
   batch32, at most3000 updates, cosine1e-3→1e-5 over that maximum horizon.
   Training loss: mean_i(||f_i-(c*_i-c_ridge_i)/s||² + lambda||f_i||²)/256.
   The penalty shrinks the predicted correction, not individual POD modes. The field
   projection residual is a constant omitted from this training loss.
4. Evaluate every100 steps; early stopping uses development mean relative L2,
   min_delta=1e-4, patience5 evaluations, earliest stop500 updates. Step0 is an eligible
   checkpoint. Persist both latest full training state and selected weights. The best
   step is updated only for improvement greater than min_delta; slight improvements
   below that threshold deliberately retain the earlier checkpoint.
5. Select one width/lambda by mean development error across all five selected seeds.
   Ties follow the declared ascending grid order, never best-seed selection. If aggregate
   gain versus frozen ridge is <=1e-4, the deployable procedure falls back to ridge.
6. Stronger linear control: rank16 ridge over alphas {1e-4,1e-3,1e-2,0.1,1,10,100,1000},
   fitted with the exact same stored Q and normalization, selected only on development.
   Eight hyperparameter settings per family does NOT equate total tuning budgets:
   the residual arm additionally selects checkpoints and trains five seeds.
7. Before training, choose500 confirmation IDs using RNG seed2026092801 from the sorted
   complement of all historical train/validation/test IDs, also conservatively excluding
   IDs0–127 used by the generic smoke-run path. Repository records show the public study
   used one1400-ID split; no other public-data split was found. This is a local provenance
   audit, not a claim about unpublished work outside this repository.
8. Stage separation: `plan` reserves IDs only; `train` cannot load confirmation labels.
   It writes an immutable selection lock containing all selected checkpoint hashes.
   Commit that lock to Git before `confirm`; the confirmation command enforces this gate.
   Confirmation extracts only the reserved500 pairs after full source SHA256 verification.
   Each50-pair chunk is cached atomically for interruption recovery. Do not inspect all
   hyperparameter configurations on confirmation data or adjust the protocol afterward.
9. Primary endpoint: average the five fixed models' per-trajectory relative L2 errors
   (NOT predictions), subtract frozen-ridge errors, and bootstrap the500 paired trajectories
   5000 times with seed20260928. Success requires both a95% percentile interval upper<0
   and at least1% relative mean improvement. Five seeds×500 points are NOT2500 independent
   observations. Intervals condition on the trained models, not training-split uncertainty.
   Expanded-ridge and mean-baseline comparisons, RMSE and seed dispersion are secondary
   descriptive checks. No multiplicity-adjusted superiority claim is made for them.
10. Confirmation is one new same-dataset trajectory split, not a second PDE, official test
    benchmark or broad robustness proof. After viewing it, any new method changes require
    another prospectively reserved confirmation set. Preserve failures as well as gains.

Full-state checkpoints every50 updates include optimizer, RNG, step, patience counter,
selected state, history and frozen anchor. Atomic progress and flushed event logs persist
while running. Checkpoints and extracted data remain local/ignored; protocol, IDs,
selection, history, statistics and per-trajectory errors are versioned.

```powershell
D:/miniconda3/envs/pytorch/python.exe scripts/run_public_ridge_residual.py plan
# Commit protocol and code, then:
D:/miniconda3/envs/pytorch/python.exe scripts/run_public_ridge_residual.py train
# Commit selection_lock.json and development results before:
D:/miniconda3/envs/pytorch/python.exe scripts/run_public_ridge_residual.py confirm
```

Rerunning a stage resumes full state or skips identity-matched completed work. No formal
results are available at protocol-lock time; any outcomes are appended below afterward.

## Development completed; confirmation still sealed

- Source commit:4e2e3e2. All40 neural runs and8 deterministic rank16 ridge candidates
  completed. Total60200 AdamW updates; all runs terminated by declared patience rules.
- Selected width32/lambda1, development mean relative L2=0.3663995661,
  sample SD=0.0032784147. Seed0–4 selected steps:2200/1400/2300/2300/1100.
  No seeds were discarded. The aggregate fallback is not activated.
- Frozen alpha10 ridge:0.3736562256. Expanded grid selects alpha100:0.3735998498.
  Development gain versus frozen ridge is approximately1.94%, still tuning evidence.
- Across the full residual grid,17/40 runs selected step0 (unmodified ridge).
  Width16/lambda0 selects step0 in5/5 seeds; width32/lambda0 in4/5. These retained
  failures are part of the result, not missing or rerun experiments.
- Independent audit replays all early-stopping decisions, reproduces group selection,
  checks checkpoint/result hashes and frozen arrays exactly. Maximum squared-error
  closure discrepancy is1.32e-16. Confirmation data have not yet been extracted.
