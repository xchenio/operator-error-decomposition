@echo off
rem ============================================================
rem 续跑：final_sweep.cmd 的第 8/9/10 组未完成部分。
rem --skip-existing 使重跑幂等，已完成的 run 会被跳过，不会浪费算力。
rem ============================================================
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not defined PY set PY=python
set BASE=--pde burgers --methods pi_baseline,pi_align --n-data 0 --align-kind prior --align-weight 1.0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --skip-existing --out results/expV

echo [8/10] w64 nu=0.01  seeds 0,1,2,3,4
%PY% scripts/run.py %BASE% --width 64 --nu 0.01 --nx 512 --nx-store 128 --seeds 0,1,2,3,4

echo [9/10] w24 nu=0.02  seeds 0,1,2,3,4
%PY% scripts/run.py %BASE% --width 24 --nu 0.02 --nx 256 --seeds 0,1,2,3,4

echo [10/10] 2D burgers2d w32  seeds 0,1,2,3,4
%PY% scripts/run.py --pde burgers2d --methods pi_baseline,pi_align --n-data 0 --align-kind prior --align-weight 1.0 --steps 6000 --eval-every 1500 --diag-every 6000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2,3,4 --save-model --skip-existing --out results/expV

echo [done] final sweep (part 2) finished
