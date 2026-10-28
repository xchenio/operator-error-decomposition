# Driver for the disjoint-split replication. Re-run this exact command to resume.
#
# A family counts as done only when all 15 runs (3 architectures x 5 seeds) have a result.json,
# so an interruption loses at most the family in progress and re-running does not re-read the
# source HDF5 files for work already completed.
#
#   powershell -File scripts/rs2_driver.ps1
#   powershell -File scripts/rs2_driver.ps1 -Only t010,t025          # resume a subset
param([string[]]$Only = @())

$ErrorActionPreference = 'Stop'
$ExpectedRuns = 15

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

$done = 0
foreach ($f in $families) {
  $tag = ($f.o -replace '^results/rs2_', '' -replace '_v1$', '')
  if ($Only.Count -gt 0 -and $tag -notin $Only) { continue }

  if (Test-Path $f.o) {
    $n = (Get-ChildItem -Recurse -Filter result.json $f.o -ErrorAction SilentlyContinue |
          Measure-Object).Count
    if ($n -ge $ExpectedRuns) {
      Write-Output "[skip] $tag already complete ($n runs)"
      $done++
      continue
    }
    Write-Output "[resume] $tag has $n of $ExpectedRuns runs"
  }

  Write-Output "[run] $tag  $(Get-Date -Format 'HH:mm:ss')"
  python -u -B scripts/run_public_benchmark.py `
      --data-dir $f.d --out $f.o --steps 1500 --device cuda
  if ($LASTEXITCODE -ne 0) {
    Write-Output "[FAILED] $tag exit=$LASTEXITCODE"
    exit $LASTEXITCODE
  }
  $done++
}

Write-Output "[done] $done of $($families.Count) families complete"
