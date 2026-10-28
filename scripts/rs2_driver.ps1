# Driver for the disjoint-split replication. Re-run this exact command to resume.
#
#   powershell -File scripts/rs2_driver.ps1
#   powershell -File scripts/rs2_driver.ps1 -Only burgers_nu0001_t150,burgers_nu0001_t200
#
# A family counts as done only when all 15 runs (3 architectures x 5 seeds) have a result.json,
# so an interruption loses at most the family in progress and re-running does not re-read the
# source HDF5 files for work already completed.
#
# Every path is resolved from $PSScriptRoot. This matters: a child `powershell -File` does not
# inherit the caller's working directory, so relative paths silently pointed one level above the
# repository and made training "fail" instantly while actually writing nothing.
# You must pass the interpreter explicitly if bare `python` does not resolve to the conda
# environment: on Windows a bare `python` often hits the Store alias stub, which exits
# immediately having done nothing. Both failure modes look like an instant training failure.
param([string[]]$Only = @(), [string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$ExpectedRuns = 15

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Output "[cwd] $root"

# `powershell -File` passes `-Only a,b,c` as one comma-joined string, so normalise either form.
if ($Only.Count -eq 1 -and $Only[0] -like '*,*') { $Only = $Only[0].Split(',') }
$Only = @($Only | Where-Object { $_ } | ForEach-Object { $_.Trim() })

$families = @(
  # fitting set: Burgers Nu = 0.001, seven horizons
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t010_v1'; o='results/rs2_burgers_nu0001_t010_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t025_v1'; o='results/rs2_burgers_nu0001_t025_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t050_v1'; o='results/rs2_burgers_nu0001_t050_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t075_v1'; o='results/rs2_burgers_nu0001_t075_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t100_v1'; o='results/rs2_burgers_nu0001_t100_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t150_v1'; o='results/rs2_burgers_nu0001_t150_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu0001/fixed_t200_v1'; o='results/rs2_burgers_nu0001_t200_v1'}
  # fitting set: linear advection, seven horizons
  @{d='external_data/pdebench/rs2/advection/fixed_t010_v1'; o='results/rs2_advection_t010_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t025_v1'; o='results/rs2_advection_t025_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t050_v1'; o='results/rs2_advection_t050_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t075_v1'; o='results/rs2_advection_t075_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t100_v1'; o='results/rs2_advection_t100_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t150_v1'; o='results/rs2_advection_t150_v1'}
  @{d='external_data/pdebench/rs2/advection/fixed_t200_v1'; o='results/rs2_advection_t200_v1'}
  # held out: Burgers Nu = 0.01, seven horizons
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t010_v1'; o='results/rs2_burgers_nu001_t010_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t025_v1'; o='results/rs2_burgers_nu001_t025_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t050_v1'; o='results/rs2_burgers_nu001_t050_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t075_v1'; o='results/rs2_burgers_nu001_t075_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t100_v1'; o='results/rs2_burgers_nu001_t100_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t150_v1'; o='results/rs2_burgers_nu001_t150_v1'}
  @{d='external_data/pdebench/rs2/burgers_nu001/fixed_t200_v1'; o='results/rs2_burgers_nu001_t200_v1'}
)

function Get-RunCount([string]$outDir) {
  $full = Join-Path $root $outDir
  if (-not (Test-Path -LiteralPath $full)) { return 0 }
  return (Get-ChildItem -LiteralPath $full -Recurse -Filter result.json |
          Measure-Object).Count
}

$done = 0
foreach ($f in $families) {
  $tag = ($f.o -replace '^results/rs2_', '' -replace '_v1$', '')
  if ($Only.Count -gt 0 -and $tag -notin $Only) { continue }

  $n = Get-RunCount $f.o
  if ($n -ge $ExpectedRuns) {
    Write-Output "[skip] $tag complete ($n runs)"
    $done++
    continue
  }
  if ($n -gt 0) { Write-Output "[resume] $tag has $n of $ExpectedRuns" }

  Write-Output "[run] $tag  $(Get-Date -Format 'HH:mm:ss')"
  & $Python -u -B (Join-Path $root 'scripts/run_public_benchmark.py') `
      --data-dir (Join-Path $root $f.d) --out (Join-Path $root $f.o) --steps 1500 --device cuda

  $n = Get-RunCount $f.o
  if ($n -lt $ExpectedRuns) {
    Write-Output "[FAILED] $tag has $n of $ExpectedRuns runs"
    exit 1
  }
  Write-Output "[done] $tag  $n runs  $(Get-Date -Format 'HH:mm:ss')"
  $done++
}

Write-Output "[all done] $done of $($families.Count) families complete"
