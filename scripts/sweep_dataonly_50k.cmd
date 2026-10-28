@echo off
rem 预算对等检查：物理侧已跑到 50k，监督侧只到 30k。
rem 若不补，"物理 > 监督"就犯了本文自己批评的预算不对等错误。
rem data_only 很便宜（30k 约 195s），50k 约 330s/seed。
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not defined PY set PY=python

%PY% scripts/run.py --pde burgers2d --methods data_only --n-data 0 --steps 50000 --eval-every 5000 --diag-every 5000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2 --skip-existing --out results/expDO50k
%PY% scripts/run.py --pde burgers --methods data_only --n-data 0 --steps 50000 --eval-every 5000 --diag-every 5000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0,1,2 --skip-existing --out results/expDO50k
echo [done] data_only 50k finished
