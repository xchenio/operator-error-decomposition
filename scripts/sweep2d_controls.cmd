@echo off
rem ============================================================
rem P0-1：2D Burgers 的「四类让基变多样 / 逃离退化解」对照
rem
rem 要回答的问题：2D 上 "alignment is necessary"（baseline 确定性塌到 u≡0，
rem relL2=1.003 → 0.821）到底是在说
rem   (a) 谱先验特殊？
rem   (b) 任何只作用于 trunk 的正则都能把 baseline 推出退化点？
rem   (c) 还只是 IC 是软约束、权重不够？
rem 1D 的教训正是 (b)：fourier trunk / ortho 几乎免费地拿到了对齐的收益。
rem
rem 已有（results/expV，5 seed）：pi_baseline、pi_align(prior)
rem 本脚本补 4 组，各 5 seed（0-4），width 32，6000 步，输出 results/expW：
rem   [1] --trunk-type fourier         零成本架构对照（1D 上它全场最好）
rem   [2] pi_ortho --ortho-weight 1.0  无目标子空间的去相关正则（同样只作用于 trunk）
rem   [3] pi_align random              同光滑类的随机子空间（检验先验是否特殊）
rem   [4] --ic-weight 10               检验塌缩是否只是软 IC 权重不足
rem   [5] （可选，默认注释掉）fourier + align：检验两者是否互补
rem
rem 不纳入 hard_ic：1D 上 u=u0+t*NN 的参数化使残差量级从 ~0.1 抬到 ~1000
rem （u0_xx 主导），loss≈1157、relL2≈2.26，不是干净的对照（实现本身已通过
rem tests/test_hard_ic_2d.py）。
rem
rem 单 run 约 6.3 分钟（width 32，6000 步）；20 个 run 约 2.1 小时。
rem --skip-existing 使重跑幂等，中断后直接再执行一次即可。
rem ============================================================
set PYTHONIOENCODING=utf-8
cd /d f:\Desktop\temp\pinode_align
set PY=D:\miniconda3\envs\pytorch\python.exe
set BASE=--pde burgers2d --n-data 0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2,3,4 --save-model --skip-existing --out results/expW

echo [1/4] fourier trunk baseline  seeds 0-4
%PY% scripts/run.py %BASE% --methods pi_baseline --trunk-type fourier

echo [2/4] ortho 去相关正则 (weight 1.0)  seeds 0-4
%PY% scripts/run.py %BASE% --methods pi_ortho --ortho-weight 1.0

echo [3/4] align random (同光滑类随机子空间)  seeds 0-4
%PY% scripts/run.py %BASE% --methods pi_align --align-kind random --align-weight 1.0

echo [4/4] baseline + ic/bc 权重 x10  seeds 0-4
%PY% scripts/run.py %BASE% --methods pi_baseline --ic-weight 10

rem echo [5/5] fourier + align prior（检验是否互补）  seeds 0-4
rem %PY% scripts/run.py %BASE% --methods pi_align --align-kind prior --align-weight 1.0 --trunk-type fourier

echo.
echo [analysis] 汇总 2D 全部 arms（含 expV 的 baseline / align prior）
%PY% scripts/analyze_attribution.py --dirs expV,expW --pde burgers2d --out results/attribution_2d.json

echo [done] 2D controls finished
