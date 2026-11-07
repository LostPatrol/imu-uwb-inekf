# Build the English IEEE paper and Chinese mathematical companion with local TeX Live.
param([string]$TexBin = 'D:/Texlive/2026/bin/windows')

$ErrorActionPreference = 'Stop'
# Keep auxiliary files out of the deliverable source directory.
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot 'agent/codex/research_20261107/repro_build'
New-Item -ItemType Directory -Force $buildRoot | Out-Null

function Invoke-CheckedTex {
    # Fail immediately if a compiler or bibliography process returns an error.
    param([string]$Executable, [string[]]$CompilerArguments)
    & $Executable @CompilerArguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with code $LASTEXITCODE" }
}

$previousBibInputs = $env:BIBINPUTS
Push-Location $PSScriptRoot
try {
    $english = 'recovery_bootstrap_paper'
    $chinese = 'recovery_bootstrap_math_zh'
    $common = @('-interaction=nonstopmode', '-halt-on-error', "-output-directory=$buildRoot")
    Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
    $env:BIBINPUTS = "$PSScriptRoot;"
    Push-Location $buildRoot
    try { Invoke-CheckedTex (Join-Path $TexBin 'bibtex.exe') @($english) }
    finally { Pop-Location }
    Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
    Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
    Invoke-CheckedTex (Join-Path $TexBin 'xelatex.exe') ($common + "$chinese.tex")
    Invoke-CheckedTex (Join-Path $TexBin 'xelatex.exe') ($common + "$chinese.tex")
    Copy-Item -LiteralPath (Join-Path $buildRoot "$english.pdf") -Destination (Join-Path $PSScriptRoot "$english.pdf")
    Copy-Item -LiteralPath (Join-Path $buildRoot "$chinese.pdf") -Destination (Join-Path $PSScriptRoot "$chinese.pdf")
}
finally {
    $env:BIBINPUTS = $previousBibInputs
    Pop-Location
}
