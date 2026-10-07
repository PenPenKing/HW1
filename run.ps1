<#
  run.ps1 -- run nut_lenet.exe with the CUDA runtime DLLs on PATH.

  Deliberately has no param() block so that the program's own --options are
  passed through untouched (otherwise something like --out collides with
  PowerShell's common parameters).

  usage:
      powershell -File run.ps1 --impl shared
      powershell -File run.ps1 --impl naive --iters 5 --n 2000
      powershell -File run.ps1 --cuda "C:\path\to\CUDA\v12.9" --impl shared
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "" }
$rest = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--cuda") { $cuda = $args[++$i] } else { $rest += $args[$i] }
}

$candidates = @($cuda,
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    "D:\Data_cuda\Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
$cudaRoot = $null
foreach ($c in $candidates) {
    if ($c -and (Test-Path (Join-Path $c "bin\nvrtc64_120_0.dll"))) { $cudaRoot = $c; break }
}
if (-not $cudaRoot) { throw "CUDA toolkit not found; pass --cuda <path>" }

$env:PATH = "$(Join-Path $cudaRoot 'bin');$env:PATH"
$exe = Join-Path $root "build\nut_lenet.exe"
if (-not (Test-Path $exe)) { throw "build/nut_lenet.exe missing -- run build.ps1 first" }

& $exe @rest
exit $LASTEXITCODE
