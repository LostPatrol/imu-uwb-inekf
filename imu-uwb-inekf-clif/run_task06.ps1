[CmdletBinding()]
param(
    [ValidateSet('download', 'test', 'quick', 'synthetic', 'diagnose', 'full', 'improve', 'author')]
    [string]$Mode = 'quick',
    [string]$Dataset = 'E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset',
    [string]$Sequence = '2021-02-12_09_56_09_twr_random.csv',
    [double]$Duration = 30.0,
    [int]$AuthorSeeds = 10,
    [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$condaExe = 'D:\miniconda\Scripts\conda.exe'
$environmentPath = 'D:\conda-envs\imu-uwb-inekf'
$experiment = Join-Path $projectRoot 'scripts\estimation\twr_experiment.py'
$authorExperiment = Join-Path $projectRoot 'scripts\estimation\pfeiffer_fair_compare.py'

if ($Mode -eq 'download') {
    & (Join-Path $projectRoot 'download_dataset.ps1')
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $condaExe)) {
    throw "Conda executable not found: $condaExe"
}
if (-not (Test-Path -LiteralPath $environmentPath)) {
    throw "Required Conda environment not found: $environmentPath"
}

function Invoke-TaskPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $condaExe run --no-capture-output -p $environmentPath python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    if ($Mode -eq 'test') {
        Invoke-TaskPython -m compileall -q scripts tests
        Invoke-TaskPython -m unittest discover -s tests -v
        exit 0
    }

    if ($Mode -eq 'synthetic') {
        Invoke-TaskPython $experiment `
            --mode synthetic `
            --output 'results\task06_synthetic_representative'
        exit 0
    }

    if (-not (Test-Path -LiteralPath (Join-Path $Dataset 'anchor_positions.yaml'))) {
        throw "Dataset not found at '$Dataset'. Run '.\run_task06.ps1 -Mode download' first."
    }

    if ($Mode -eq 'quick') {
        Invoke-TaskPython $experiment `
            --mode real `
            --dataset $Dataset `
            --sequence $Sequence `
            --duration $Duration `
            --offset 6 -6 3 `
            --output 'results\task06_quick\real'
        Invoke-TaskPython $experiment `
            --mode synthetic `
            --output 'results\task06_quick\synthetic'
        Write-Host "Quick results: $(Join-Path $projectRoot 'results\task06_quick')"
        exit 0
    }

    if ($Mode -eq 'diagnose') {
        Invoke-TaskPython $experiment `
            --mode diagnostic `
            --dataset $Dataset `
            --output 'results\task06_tail_diagnostic'
        Write-Host "Diagnostic results: $(Join-Path $projectRoot 'results\task06_tail_diagnostic')"
        exit 0
    }

    if ($Mode -eq 'improve') {
        Invoke-TaskPython $experiment `
            --mode improved `
            --dataset $Dataset `
            --output 'results\task06_per_anchor_robust'
        Write-Host "Improved Task 06 results: $(Join-Path $projectRoot 'results\task06_per_anchor_robust')"
        exit 0
    }

    if ($Mode -eq 'author') {
        Invoke-TaskPython $authorExperiment `
            --dataset $Dataset `
            --seeds $AuthorSeeds `
            --workers $Workers `
            --output 'results\task06_author_fair_compare'
        Write-Host "Author fair-comparison results: $(Join-Path $projectRoot 'results\task06_author_fair_compare')"
        exit 0
    }

    Invoke-TaskPython $experiment `
        --mode full `
        --dataset $Dataset `
        --output 'results\task06_full'
    Write-Host "Full results: $(Join-Path $projectRoot 'results\task06_full')"
}
finally {
    Pop-Location
}
