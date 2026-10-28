@echo off
rem data_only（纯监督）步数充分性检查：它比物理训练便宜 20 倍（30s vs 6min / 6k 步），
rem 所以必须给它 5~8 倍的步数，否则"物理 > 监督"可能只是监督没训够。
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not defined PY set PY=python

%PY% scripts/run.py --pde burgers2d --methods data_only --n-data 0 --steps 30000 --eval-every 5000 --diag-every 5000 --device cuda --steps-col 64 --bs 16 --width 32 --seeds 0,1,2 --skip-existing --out results/expDOlong
%PY% scripts/run.py --pde burgers --methods data_only --n-data 0 --steps 30000 --eval-every 5000 --diag-every 5000 --device cuda --nu 0.02 --nx 256 --nx-store 128 --width 64 --seeds 0,1,2 --skip-existing --out results/expDOlong
echo [done] data_only long finished
