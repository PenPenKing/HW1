<#
  bench.ps1 -- reproduce the performance table.

  Runs every implementation over the full 10000-image test set, once with the
  inputs resident on the device (pure kernel time) and once with the usual
  host->device copy per batch (end to end), then writes
      results/bench_<impl>.json
      results/summary.md
      results/bench_all.csv

  usage:  powershell -ExecutionPolicy Bypass -File bench.ps1 [--iters N]
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$iters = 20
for ($i = 0; $i -lt $args.Count; $i++) { if ($args[$i] -eq "--iters") { $iters = [int]$args[$i + 1] } }

$cuda = $env:CUDA_PATH
if (-not $cuda -or -not (Test-Path (Join-Path $cuda "bin\nvrtc64_120_0.dll"))) {
    foreach ($c in @("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
                     "D:\Data_cuda\Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")) {
        if (Test-Path (Join-Path $c "bin\nvrtc64_120_0.dll")) { $cuda = $c; break }
    }
}
$env:PATH = "$(Join-Path $cuda 'bin');$env:PATH"
$exe = Join-Path $root "build\nut_lenet.exe"
if (-not (Test-Path $exe)) { throw "build/nut_lenet.exe missing -- run build.ps1 first" }
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null

$impls = @("naive", "shared", "row", "best")
$rows = @()

foreach ($mode in @("resident", "streamed")) {
    foreach ($impl in $impls) {
        $json = "results/bench_$($impl)_$mode.json"
        $a = @("--impl", $impl, "--batch", "2000", "--iters", "$iters", "--warmup", "5",
               "--out", $json)
        if ($mode -eq "resident") { $a += "--resident" }
        Write-Host ("running impl={0,-6} mode={1}" -f $impl, $mode)
        & $exe @a | Out-Null
        $o = Get-Content $json -Raw | ConvertFrom-Json
        $rows += [pscustomobject]@{
            impl     = $impl
            mode     = $mode
            accuracy = [math]::Round($o.accuracy_percent, 2)
            pass_ms  = [math]::Round($o.pass_best_ms, 3)
            per_img  = [math]::Round($o.ms_per_image, 5)
            img_s    = [math]::Round($o.images_per_second, 0)
            conv1    = [math]::Round($o.layers_ms.conv1, 3)
            conv2    = [math]::Round($o.layers_ms.conv2, 3)
            dense    = [math]::Round($o.layers_ms.dense, 3)
        }
    }
}

$rows | Export-Csv -NoTypeInformation -Path "results/bench_all.csv"

$md = @()
$md += "# TinyLeNet on RTX 4090 D - full 10000-image MNIST test set"
$md += ""
$md += "batch = 2000, warm-up 5 passes, best of $iters timed passes."
$md += "resident = inputs already in device memory (pure kernel time);"
$md += "streamed = host->device copy of every batch included."
$md += ""
$md += "| impl | mode | accuracy | pass (ms) | per image (ms) | images/s | conv1 (ms) | conv2 (ms) | dense (ms) |"
$md += "|---|---|---:|---:|---:|---:|---:|---:|---:|"
foreach ($r in $rows) {
    $md += ("| {0} | {1} | {2}% | {3} | {4} | {5} | {6} | {7} | {8} |" -f `
        $r.impl, $r.mode, $r.accuracy, $r.pass_ms, $r.per_img, $r.img_s, $r.conv1, $r.conv2, $r.dense)
}
$md -join "`n" | Set-Content "results/summary.md"
Write-Host ""
Get-Content "results/summary.md"
