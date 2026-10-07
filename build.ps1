<#
  build.ps1 -- build nut_lenet.exe on Windows without MSVC.

  Host code (main.cpp + launch_nvrtc.cpp) is compiled by mingw g++; the device
  code (kernels.cu) is compiled at run time by NVRTC.  No Visual Studio, no
  nvcc, no cl.exe required.

  usage:
      powershell -ExecutionPolicy Bypass -File build.ps1
      powershell -ExecutionPolicy Bypass -File build.ps1 -Cuda "C:\...\CUDA\v12.9"
#>
param(
    [string]$Cuda = $env:CUDA_PATH,
    [string]$Gxx = "",
    [string]$Out = "build",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ---- locate the CUDA toolkit -------------------------------------------------
function Test-Cuda([string]$p) {
    return ($p -and (Test-Path (Join-Path $p "include\cuda_runtime.h")) -and
            (Test-Path (Join-Path $p "include\nvrtc.h")))
}
$candidates = @($Cuda,
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    "D:\Data_cuda\Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
$cudaRoot = $null
foreach ($c in $candidates) { if (Test-Cuda $c) { $cudaRoot = $c; break } }
if (-not $cudaRoot) { throw "CUDA toolkit with nvrtc not found; pass -Cuda <path>" }
Write-Host "CUDA      : $cudaRoot"

# ---- locate g++ --------------------------------------------------------------
if (-not $Gxx) {
    $cmd = Get-Command g++ -ErrorAction SilentlyContinue
    if ($cmd) { $Gxx = $cmd.Source }
    elseif (Test-Path "D:\Program Files\mingw64\bin\g++.exe") { $Gxx = "D:\Program Files\mingw64\bin\g++.exe" }
    elseif (Test-Path "C:\mingw64\bin\g++.exe") { $Gxx = "C:\mingw64\bin\g++.exe" }
}
if (-not $Gxx -or -not (Test-Path $Gxx)) { throw "g++ not found; pass -Gxx <path>" }
Write-Host "host g++  : $Gxx"
$gxxDir = Split-Path -Parent $Gxx

if ($Clean -and (Test-Path $Out)) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# ---- compile -----------------------------------------------------------------
$inc = Join-Path $cudaRoot "include"
$bin = Join-Path $cudaRoot "bin"
$exe = Join-Path $Out "nut_lenet.exe"

$argsList = @(
    "-O2", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter",
    "-Wno-builtin-macro-redefined",
    "-static-libgcc", "-static-libstdc++",
    "main.cpp", "launch_nvrtc.cpp",
    "-I$inc",
    (Join-Path $bin "cudart64_12.dll"),
    (Join-Path $bin "nvrtc64_120_0.dll"),
    "-o", $exe
)

Write-Host "compiling ..."
$env:PATH = "$gxxDir;$bin;$env:PATH"
& $Gxx @argsList
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "built: $exe"
Write-Host "run  : powershell -File run.ps1 --impl shared"
