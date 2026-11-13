# Build the pinned author numerical core with local MSVC and isolated dependencies.
param(
    [string]$ProjectRoot = 'E:/0syncthing/Code/InEKF',
    [string]$VisualStudio = 'C:/Program Files/Microsoft Visual Studio/18/Community',
    [switch]$DebugBuild
)
$ErrorActionPreference = 'Stop'
$buildRoot = Join-Path $ProjectRoot 'agent/codex/toronto_sfuise'
$dependencyRoot = Join-Path $ProjectRoot 'agent/codex/sfuise_challenge/deps/Library'
$python = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
# Generated core is copied unchanged from the audited sibling portability build.
& (Join-Path $VisualStudio 'Common7/Tools/Launch-VsDevShell.ps1') -SkipAutomaticLocation -Arch amd64 -HostArch amd64 | Out-Null
$optimization = if ($DebugBuild) { '/Od' } else { '/O2' }
$compileArgs = @('/nologo', '/utf-8', '/std:c++17', $optimization, '/MD', '/EHsc', '/bigobj', '/D_USE_MATH_DEFINES',
    ('/I' + (Join-Path $buildRoot 'generated')), ('/I' + (Join-Path $dependencyRoot 'include')),
    ('/I' + (Join-Path $dependencyRoot 'include/suitesparse')),
    (Join-Path $PSScriptRoot 'native_cli.cpp'), ('/Fo' + (Join-Path $buildRoot 'sfuise_native.obj')),
    ('/Fe' + (Join-Path $buildRoot 'sfuise_native.exe')), '/link',
    ('/LIBPATH:' + (Join-Path $dependencyRoot 'lib')), 'cholmod.lib')
& cl.exe @compileArgs
if ($LASTEXITCODE -ne 0) { throw 'Native SFUISE compilation failed.' }
