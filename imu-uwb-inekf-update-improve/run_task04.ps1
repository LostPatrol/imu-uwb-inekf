param(
    [ValidateSet("test", "quick", "full", "overhead")]
    [string]$Mode = "quick",

    [string]$AnchorPath = "E:\datasets\dataset\flight-dataset\survey-results\anchor_const1.npz",

    [string]$CsvPath = "E:\datasets\dataset\flight-dataset\csv-data\const1\const1-trial1-tdoa2.csv",

    [string]$OutputDir = "",

    [double]$DurationSeconds = 30.0,

    [double]$UwbRateHz = 0.0
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$condaExecutable = "D:\miniconda\Scripts\conda.exe"
$environmentPath = "D:\conda-envs\imu-uwb-inekf"
$testScript = Join-Path $repositoryRoot "tests\test_pose_update.py"
$comparisonScript = Join-Path $repositoryRoot "experiments\compare_pose_update.py"
$overheadScript = Join-Path $repositoryRoot "experiments\benchmark_pose_update_overhead.py"

if (-not (Test-Path -LiteralPath $condaExecutable)) {
    throw "Conda executable not found: $condaExecutable"
}
if (-not (Test-Path -LiteralPath $environmentPath)) {
    throw "Conda environment not found: $environmentPath"
}

Write-Output "[task04] Running focused tests"
& $condaExecutable run --no-capture-output -p $environmentPath python $testScript
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Mode -eq "test") {
    Write-Output "[task04] Tests completed"
    exit 0
}

if ($Mode -eq "overhead") {
    if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        $OutputDir = Join-Path $repositoryRoot "results\task04_compute_overhead"
    }
    Write-Output "[task04] Running AB/BA computational overhead benchmark"
    & $condaExecutable run --no-capture-output -p $environmentPath python `
        $overheadScript --output-dir $OutputDir
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $AnchorPath)) {
    throw "Anchor file not found: $AnchorPath"
}
if (-not (Test-Path -LiteralPath $CsvPath)) {
    throw "Dataset CSV not found: $CsvPath"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repositoryRoot "results\task04_$Mode"
}

$comparisonArguments = @(
    "run",
    "--no-capture-output",
    "-p",
    $environmentPath,
    "python",
    $comparisonScript,
    "-i",
    $AnchorPath,
    $CsvPath,
    "--output-dir",
    $OutputDir
)

if ($Mode -eq "quick") {
    if ($DurationSeconds -le 0.0) {
        throw "DurationSeconds must be positive in quick mode"
    }
    $comparisonArguments += @("--max-duration", $DurationSeconds.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    ))
}

if ($UwbRateHz -gt 0.0) {
    $comparisonArguments += @("--uwb-rate-hz", $UwbRateHz.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    ))
}

Write-Output "[task04] Running $Mode dataset comparison"
& $condaExecutable @comparisonArguments
exit $LASTEXITCODE
