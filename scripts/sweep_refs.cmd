@echo off
rem ============================================================
rem 三个缺失的参照系（2026-09-21）
rem
rem  A) data_only 监督上界：全项目 0 次。没有它，"relL2 0.69 是优化受限"没有尺度参照。
rem  B) 步数收敛扫描 6k/20k/50k：证明 0.69 不是"步数不够"。
rem  C) 同 seed 重跑 5 次测 sigma_repeat：已发现一组重复运行 Delta=0.0124，
rem     与"方差地板 ~0.01"同量级；这个对照没做，地板结论不能进论文。
rem
rem ⚠️ 注意：ExpCfg.tag() 不含 steps，所以不同步数的 run 必须写到不同目录，
rem    否则同名 JSON 会互相覆盖（与 README §6.5 同类的静默问题）。
rem ============================================================
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not defined PY set PY=python

echo ########## A) data_only 监督上界 ##########
rem 2D：纯监督能到多少？这是 2D 上"这个架构到底能不能学会这个算子"的答案
%PY% scripts/run.py --pde burgers2d --methods data_only --n-data 0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2 --skip-existing --out results/expDO
rem 1D 参照（nu=0.02, w64）
%PY% scripts/run.py --pde burgers --methods data_only --n-data 0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0,1,2 --skip-existing --out results/expDO

echo ########## C) 同 seed 重跑 5 次：sigma_repeat ##########
rem 1D nu=0.02 w64 baseline：该格 seed 间 sigma=0.0122，正好落在"地板"上——
rem 若重跑 sigma 也是 ~0.01，则地板主要是数值噪声，不是对齐的效果
for %%i in (1 2 3 4 5) do (
  %PY% scripts/run.py --pde burgers --methods pi_baseline --n-data 0 --steps 6000 --eval-every 6000 --diag-every 6000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0 --out results/rep1d%%i
)
rem 2D baseline：论文称其"确定性塌缩"(sigma=0.0024)，需要重跑噪声来验证
for %%i in (1 2 3 4 5) do (
  %PY% scripts/run.py --pde burgers2d --methods pi_baseline --n-data 0 --steps 6000 --eval-every 6000 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0 --out results/rep2d%%i
)

echo ########## B) 步数收敛扫描 ##########
rem 2D fourier（目前最好 0.693）继续加步数能否再降？
%PY% scripts/run.py --pde burgers2d --methods pi_baseline --trunk-type fourier --n-data 0 --steps 20000 --eval-every 5000 --diag-every 5000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0 --skip-existing --out results/expS20k
%PY% scripts/run.py --pde burgers2d --methods pi_baseline --trunk-type fourier --n-data 0 --steps 50000 --eval-every 5000 --diag-every 5000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0 --skip-existing --out results/expS50k
rem 1D nu=0.02 w64 baseline：opt_gap_train~0.45 会不会只是步数不够？
%PY% scripts/run.py --pde burgers --methods pi_baseline --n-data 0 --steps 20000 --eval-every 5000 --diag-every 5000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0 --skip-existing --out results/expS20k
%PY% scripts/run.py --pde burgers --methods pi_baseline --n-data 0 --steps 50000 --eval-every 5000 --diag-every 5000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0 --skip-existing --out results/expS50k

echo ########## 汇总 ##########
%PY% scripts/analyze_attribution.py --dirs expV,expW --pde burgers2d --out results/attribution_2d.json
%PY% scripts/analyze_attribution.py --dirs expS20k,expS50k --out results/attribution_steps.json

echo [done] reference sweeps finished
