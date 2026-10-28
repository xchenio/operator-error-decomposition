# Recompute the mean/fluctuation diagnostic for every split-2 family. Re-run to resume.
#
#   powershell -File scripts/rs2_diagnose.ps1
#
# A family is done when its diagnostic JSON exists, so re-running skips finished work.
#
# NOTE: pass the interpreter explicitly. A bare `python` resolves to the Windows Store alias
# stub on this machine, which exits immediately; note also that a child `powershell -File` does
# not inherit the caller's directory, hence every path below is derived from $PSScriptRoot.
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Output "[cwd] $root"

$tags = @('010', '025', '050', '075', '100', '150', '200')

$families = @()
foreach ($tag in $tags) {
  $families += @{
    d = Join-Path $root "external_data/pdebench/rs2/burgers_nu0001/fixed_t$($tag)_v1"
    o = Join-Path $root "results/rs2_fluctuation/burgers_nu0001/t$($tag)"
    n = "rs2_burgers_nu0001_t$($tag)_v1"
  }
}
foreach ($tag in $tags) {
  $families += @{
    d = Join-Path $root "external_data/pdebench/rs2/advection/fixed_t$($tag)_v1"
    o = Join-Path $root "results/rs2_fluctuation/advection/t$($tag)"
    n = "rs2_advection_t$($tag)_v1"
  }
}
foreach ($tag in $tags) {
  $families += @{
    d = Join-Path $root "external_data/pdebench/rs2/burgers_nu001/fixed_t$($tag)_v1"
    o = Join-Path $root "results/rs2_fluctuation/burgers_nu001/t$($tag)"
    n = "rs2_burgers_nu001_t$($tag)_v1"
  }
}

foreach ($f in $families) {
  $target = Join-Path $f.o "$($f.n).json"
  if (Test-Path -LiteralPath $target) {
    Write-Output "[skip] $($f.n)"
    continue
  }
  Write-Output "[run] $($f.n)  $(Get-Date -Format 'HH:mm:ss')"
  & $Python -u scripts/diagnose_fluctuation.py `
      --data $f.d `
      --families "$($f.n)=endtoend:test,val" `
      --out $f.o --device cuda
  if ($LASTEXITCODE -ne 0) {
    Write-Output "[FAILED] $($f.n) exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
}

Write-Output "[diag done]"
