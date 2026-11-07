[CmdletBinding()]
param(
    [string]$Destination = 'E:\datasets\tu-delft-uwb-mhe'
)

$ErrorActionPreference = 'Stop'
$expectedMd5 = '7f9f2f81e1b028db8f03ccc4ab0da346'
$downloadUri = 'https://data.4tu.nl/file/f572e49f-ed95-4778-808e-e72c645b9777/4300a573-e342-4daf-b62b-610f126a8013'
$zipPath = Join-Path $Destination 'UWB dataset.zip'
$extractRoot = Join-Path $Destination 'data_v1'
$datasetRoot = Join-Path $extractRoot 'UWB dataset'

New-Item -ItemType Directory -Force $Destination | Out-Null

if (-not (Test-Path -LiteralPath $zipPath)) {
    Write-Host "Downloading the TU Delft TWR/TDoA dataset..."
    Invoke-WebRequest -Uri $downloadUri -OutFile $zipPath
}

$actualMd5 = (Get-FileHash -LiteralPath $zipPath -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "Dataset MD5 mismatch. Expected $expectedMd5, got $actualMd5."
}
Write-Host "Dataset archive MD5 verified: $actualMd5"

if (-not (Test-Path -LiteralPath (Join-Path $datasetRoot 'anchor_positions.yaml'))) {
    New-Item -ItemType Directory -Force $extractRoot | Out-Null
    tar.exe -xf $zipPath -C $extractRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset extraction failed with exit code $LASTEXITCODE."
    }
}

$twrFiles = @(Get-ChildItem -LiteralPath (Join-Path $datasetRoot 'twr') -Filter '*.csv' -File)
if ($twrFiles.Count -ne 12) {
    throw "Expected 12 TWR logs after extraction, found $($twrFiles.Count)."
}

Write-Host "Dataset ready: $datasetRoot"
Write-Host "TWR logs: $($twrFiles.Count)"
