# Build the IEEE manuscript, supplement, and Chinese mathematics with local TeX Live.
param([string]$TexBin = 'D:/Texlive/2026/bin/windows')

$ErrorActionPreference = 'Stop'
# Keep auxiliary files out of the deliverable source directory.
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot 'agent/codex/recovery_paper_build'
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
    $englishDocuments = @('recovery_bootstrap_paper', 'recovery_bootstrap_supplement')
    $chinese = 'recovery_bootstrap_math_zh'
    $common = @('-interaction=nonstopmode', '-halt-on-error', "-output-directory=$buildRoot")
    foreach ($english in $englishDocuments) {
        Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
        $env:BIBINPUTS = "$PSScriptRoot;"
        Push-Location $buildRoot
        try { Invoke-CheckedTex (Join-Path $TexBin 'bibtex.exe') @($english) }
        finally { Pop-Location }
        Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
        Invoke-CheckedTex (Join-Path $TexBin 'pdflatex.exe') ($common + "$english.tex")
        Copy-Item -LiteralPath (Join-Path $buildRoot "$english.pdf") -Destination (Join-Path $PSScriptRoot "$english.pdf")
    }
    Invoke-CheckedTex (Join-Path $TexBin 'xelatex.exe') ($common + "$chinese.tex")
    Invoke-CheckedTex (Join-Path $TexBin 'xelatex.exe') ($common + "$chinese.tex")
    Copy-Item -LiteralPath (Join-Path $buildRoot "$chinese.pdf") -Destination (Join-Path $PSScriptRoot "$chinese.pdf")
}
finally {
    $env:BIBINPUTS = $previousBibInputs
    Pop-Location
}
