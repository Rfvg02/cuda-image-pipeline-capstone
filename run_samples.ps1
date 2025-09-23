$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Carpetas necesarias
@("out","out\results","out\logs","out\submission") | ForEach-Object {
  if (-not (Test-Path $_)) { New-Item -ItemType Directory $_ | Out-Null }
}

# Info de entorno
nvcc --version | Out-File -Encoding utf8 out\submission\env_nvcc.txt
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Out-File -Encoding utf8 out\submission\env_gpu.txt
} else {
  "nvidia-smi not found" | Out-File -Encoding utf8 out\submission\env_gpu.txt
}

# CSV de timings
$csv = "out\logs\timings.csv"
if (-not (Test-Path $csv)) {
  "date,cmd,W,H,N,streams,sigma,H2D_ms,BLUR_ms,SOBEL_ms,D2H_ms,TOTAL_ms" | Out-File -Encoding utf8 $csv
}

function Run-Case {
  param([int]$W,[int]$H,[int]$N,[int]$Streams,[double]$Sigma)
  $cmd = ".\bin\image_pipeline.exe --n $N --w $W --h $H --streams $Streams --sigma $Sigma"
  Write-Host ">>> $cmd"
  $out = & .\bin\image_pipeline.exe --n $N --w $W --h $H --streams $Streams --sigma $Sigma

  # Guardar salida raw
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $raw = "out\submission\run_${stamp}_w${W}_h${H}_n${N}_s${Streams}_sg${Sigma}.txt"
  $out | Out-File -Encoding utf8 $raw

  # Parsear línea de timings
  $line = $out | Select-String -Pattern 'Timing\(ms\):'
  if ($line) {
    if ($line.Line -match 'H2D=([\d\.]+)\s+BLUR=([\d\.]+)\s+SOBEL=([\d\.]+)\s+D2H=([\d\.]+)\s+TOTAL=([\d\.]+)') {
      $h2d=$Matches[1]; $blur=$Matches[2]; $sobel=$Matches[3]; $d2h=$Matches[4]; $total=$Matches[5]
      $date = Get-Date -Format s
      "$date,""$cmd"",$W,$H,$N,$Streams,$Sigma,$h2d,$blur,$sobel,$d2h,$total" | Add-Content -Encoding utf8 $csv
    }
  }
}

# Limpiar resultados anteriores
Remove-Item -Force -ErrorAction SilentlyContinue out\results\*

# Tres corridas a escala (decenas de imágenes / resoluciones grandes)
Run-Case -W 2560 -H 1440 -N 30 -Streams 8 -Sigma 1.6
Run-Case -W 1920 -H 1080 -N 40 -Streams 8 -Sigma 1.4
Run-Case -W 3840 -H 2160 -N 12 -Streams 8 -Sigma 2.0

# Copiar un subconjunto de imágenes como muestra
Copy-Item out\results\img_00000_* out\submission -Force -ErrorAction SilentlyContinue
Copy-Item out\results\img_00001_* out\submission -Force -ErrorAction SilentlyContinue
Copy-Item out\results\img_00002_* out\submission -Force -ErrorAction SilentlyContinue

# Incluir CSV y logs
Copy-Item out\logs\timings.csv out\submission -Force

# ZIP final
if (Test-Path .\submission_evidence.zip) { Remove-Item .\submission_evidence.zip -Force }
Compress-Archive -Path out\submission\* -DestinationPath submission_evidence.zip -Force

Write-Host "`nOK -> $(Resolve-Path .\submission_evidence.zip)"
