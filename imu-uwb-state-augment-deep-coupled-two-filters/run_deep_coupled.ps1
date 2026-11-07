[CmdletBinding()]
param(
    [ValidateSet('download', 'test', 'calibrate', 'quick', 'real', 'synthetic', 'full')]
    [string]$Mode = 'quick',
    [string]$Dataset = 'E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset',
    [string]$Sequence = '2021-02-12_09_56_09_twr_random.csv',
    [double]$Duration = 30.0,
    [double]$SensorDelaySeconds = 0.20,
    [double]$SyntheticDuration = 45.0,
    [int]$Trials = 20
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$condaExe = 'D:\miniconda\Scripts\conda.exe'
$environmentPath = 'D:\conda-envs\imu-uwb-inekf'
$experiment = Join-Path $projectRoot 'scripts\estimation\deep_coupled_experiment.py'
$delayDiagnostic = Join-Path $projectRoot 'scripts\estimation\time_alignment_diagnostic.py'

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

function Invoke-ProjectPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $condaExe run --no-capture-output -p $environmentPath python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    if ($Mode -eq 'test') {
        Invoke-ProjectPython -m compileall -q scripts tests
        Invoke-ProjectPython -m unittest discover -s tests -v
        exit 0
    }

    if ($Mode -ne 'synthetic' -and -not (
        Test-Path -LiteralPath (Join-Path $Dataset 'anchor_positions.yaml')
    )) {
        throw "Dataset not found at '$Dataset'. Run '.\run_deep_coupled.ps1 -Mode download' first."
    }

    if ($Mode -eq 'calibrate') {
        Invoke-ProjectPython $delayDiagnostic `
            --dataset $Dataset `
            --selected-delay $SensorDelaySeconds `
            --output 'results\time_alignment_calibration'
        Write-Host "Calibration results: $(Join-Path $projectRoot 'results\time_alignment_calibration')"
        exit 0
    }

    if ($Mode -eq 'quick') {
        Invoke-ProjectPython $experiment `
            --mode quick `
            --dataset $Dataset `
            --sequence $Sequence `
            --duration $Duration `
            --synthetic-duration $SyntheticDuration `
            --sensor-delay $SensorDelaySeconds `
            --offset 6 -6 3 `
            --output 'results\deep_coupled_delay_quick'
        Write-Host "Quick results: $(Join-Path $projectRoot 'results\deep_coupled_delay_quick')"
        exit 0
    }

    if ($Mode -eq 'real') {
        Invoke-ProjectPython $experiment `
            --mode real `
            --dataset $Dataset `
            --sequence $Sequence `
            --sensor-delay $SensorDelaySeconds `
            --offset 6 -6 3 `
            --output 'results\deep_coupled_delay_real'
        exit 0
    }

    if ($Mode -eq 'synthetic') {
        Invoke-ProjectPython $experiment `
            --mode synthetic `
            --trials $Trials `
            --synthetic-duration $SyntheticDuration `
            --output 'results\deep_coupled_synthetic'
        exit 0
    }

    Invoke-ProjectPython $experiment `
        --mode full `
        --dataset $Dataset `
        --sensor-delay $SensorDelaySeconds `
        --trials $Trials `
        --synthetic-duration $SyntheticDuration `
        --output 'results\deep_coupled_delay_full'
    Write-Host "Full results: $(Join-Path $projectRoot 'results\deep_coupled_delay_full')"
}
finally {
    Pop-Location
}
