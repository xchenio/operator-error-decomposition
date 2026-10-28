@echo off
rem ============================================================
rem 最终相图批量：目标 n>=10 个设置，每个设置 5 个 seed。
rem 只补缺失的 seed，避免重复浪费算力。全部写入 results/expV。
rem 预计总时长 3~5 小时，建议后台跑完再用 scripts/analyze_phase.py 汇总。
rem ============================================================
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not defined PY set PY=python
set BASE=--pde burgers --methods pi_baseline,pi_align --n-data 0 --align-kind prior --align-weight 1.0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --out results/expV

echo [1/10] w16 nu=0.02  seeds 3,4
%PY% scripts/run.py %BASE% --width 16 --nu 0.02 --nx 256 --seeds 3,4

echo [2/10] w32 nu=0.02  seeds 3,4
%PY% scripts/run.py %BASE% --width 32 --nu 0.02 --nx 256 --seeds 3,4

echo [3/10] w64 nu=0.02  seeds 3,4
%PY% scripts/run.py %BASE% --width 64 --nu 0.02 --nx 256 --seeds 3,4

echo [4/10] w16 nu=0.005 seeds 3,4
%PY% scripts/run.py %BASE% --width 16 --nu 0.005 --nx 1024 --nx-store 128 --seeds 3,4

echo [5/10] w32 nu=0.005 seeds 3,4
%PY% scripts/run.py %BASE% --width 32 --nu 0.005 --nx 1024 --nx-store 128 --seeds 3,4

echo [6/10] w64 nu=0.005 seeds 3,4
%PY% scripts/run.py %BASE% --width 64 --nu 0.005 --nx 1024 --nx-store 128 --seeds 3,4

echo [7/10] w32 nu=0.01  seeds 0,1,2,3,4
%PY% scripts/run.py %BASE% --width 32 --nu 0.01 --nx 512 --nx-store 128 --seeds 0,1,2,3,4

echo [8/10] w64 nu=0.01  seeds 0,1,2,3,4   (首次运行需生成数据 ~2min)
%PY% scripts/run.py %BASE% --width 64 --nu 0.01 --nx 512 --nx-store 128 --seeds 0,1,2,3,4

echo [9/10] w24 nu=0.02  seeds 0,1,2,3,4   (新增设置)
%PY% scripts/run.py %BASE% --width 24 --nu 0.02 --nx 256 --seeds 0,1,2,3,4

echo [10/10] 2D burgers2d w32  seeds 0,1,2,3,4
%PY% scripts/run.py --pde burgers2d --methods pi_baseline,pi_align --n-data 0 --align-kind prior --align-weight 1.0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2,3,4 --save-model --out results/expV

echo [done] final sweep finished
