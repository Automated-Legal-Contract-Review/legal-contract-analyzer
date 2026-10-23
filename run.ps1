<#
  LexiGuard Local - start script for Windows PowerShell 5.1 or PowerShell 7.
  Usage:   .\run.ps1            (or double-click run.bat)
  Options: -SkipDocker  don't try to start Qdrant
           -Model NAME  AI model to use (default: llama3.2)
           -Port N      web port (default: 8000)
#>
param(
  [switch]$SkipDocker,
  [string]$Model = "",
  [int]$Port = 0
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Say($text)  { Write-Host $text -ForegroundColor Cyan }
function Good($text) { Write-Host ("  [ok]   " + $text) -ForegroundColor Green }
function Warn($text) { Write-Host ("  [note] " + $text) -ForegroundColor Yellow }
function Fail($text) {
  Write-Host ""
  Write-Host ("  [stop] " + $text) -ForegroundColor Red
  Write-Host ""
  exit 1
}

# ---- read settings from .env (only the few the script needs) ----
$envValues = @{}
if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$' -and -not $_.TrimStart().StartsWith("#")) {
      $envValues[$Matches[1]] = $Matches[2].Trim('"').Trim("'")
    }
  }
}
if (-not $Model) { $Model = if ($envValues["OLLAMA_MODEL"]) { $envValues["OLLAMA_MODEL"] } else { "llama3.2" } }
$EmbedModel = if ($envValues["OLLAMA_EMBED_MODEL"]) { $envValues["OLLAMA_EMBED_MODEL"] } else { "nomic-embed-text" }
if ($Port -le 0) { $Port = if ($envValues["LEXIGUARD_PORT"]) { [int]$envValues["LEXIGUARD_PORT"] } else { 8000 } }
$useQdrant = -not $SkipDocker -and ($envValues["USE_QDRANT"] -ne "false")
$Url = "http://127.0.0.1:$Port"

Say "LexiGuard Local"
Write-Host ""

# ---- 1. Python ----
Say "1/5  Checking Python"
$pythonExe = $null
$pythonArgs = @()
$candidates = @(
  @{ Exe = "py";      Args = @("-3") },
  @{ Exe = "python";  Args = @() },
  @{ Exe = "python3"; Args = @() }
)
foreach ($c in $candidates) {
  if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
  try {
    $cargs = $c.Args
    $out = & $c.Exe @cargs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out -match '^(\d+)\.(\d+)$') {
      $major = [int]$Matches[1]; $minor = [int]$Matches[2]
      if ($major -eq 3 -and $minor -ge 10) { $pythonExe = $c.Exe; $pythonArgs = $c.Args; Good ("Python " + $out); break }
    }
  } catch { }
}
if (-not $pythonExe) {
  Fail "Python 3.10 or newer was not found.`n         Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH',`n         then open a NEW PowerShell window and run this script again."
}

# ---- 2. virtual environment and packages ----
Say "2/5  Preparing the Python environment (first run takes a few minutes)"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  & $pythonExe @pythonArgs -m venv .venv
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "Could not create the virtual environment (.venv)." }
}
$reqHash = (Get-FileHash "requirements.txt" -Algorithm SHA256).Hash
$stamp = Join-Path $PSScriptRoot ".venv\requirements.sha256"
$installed = (Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $reqHash)
if (-not $installed) {
  & $venvPython -m pip install --disable-pip-version-check -r requirements.txt
  if ($LASTEXITCODE -ne 0) { Fail "Installing the Python packages failed. Check your internet connection and run the script again." }
  Set-Content -Path $stamp -Value $reqHash
}
Good "Packages ready"

# ---- 3. Ollama ----
Say "3/5  Checking Ollama and the AI model"
function Get-OllamaModels {
  try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
    return @($r.models | ForEach-Object { $_.name })
  } catch { return $null }
}
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
$models = Get-OllamaModels
if ($null -eq $models -and $ollamaCmd) {
  Warn "Ollama is not running. Starting it..."
  Start-Process -FilePath $ollamaCmd.Source -ArgumentList "serve" -WindowStyle Hidden
  for ($i = 0; $i -lt 20 -and $null -eq $models; $i++) { Start-Sleep -Seconds 1; $models = Get-OllamaModels }
}
if ($null -eq $models) {
  Fail "Ollama is not installed or did not start.`n         Install it from https://ollama.com/download, open it once, then run this script again."
}
function Has-Model($name) {
  foreach ($m in $models) { if ($m -eq $name -or $m -eq ($name + ":latest") -or ($name -notmatch ":" -and $m.Split(":")[0] -eq $name)) { return $true } }
  return $false
}
if (-not (Has-Model $Model)) {
  Warn "Downloading the AI model '$Model' (one time, a few GB)..."
  & ollama pull $Model
  $models = Get-OllamaModels
  if ($null -eq $models -or -not (Has-Model $Model)) { Fail "The model '$Model' could not be downloaded. Check your internet connection." }
}
Good "Model '$Model' is ready"
if ($useQdrant -and -not (Has-Model $EmbedModel)) {
  Warn "Downloading the small search model '$EmbedModel' (optional, ~270 MB)..."
  & ollama pull $EmbedModel
  $models = Get-OllamaModels
  if ($null -eq $models -or -not (Has-Model $EmbedModel)) { Warn "Search model not installed. Question answering will use keyword search." }
}

# ---- 4. Qdrant (optional) ----
Say "4/5  Checking Qdrant (optional smarter search)"
function Test-Qdrant {
  try { Invoke-WebRequest -Uri "http://127.0.0.1:6333/healthz" -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}
if (-not $useQdrant) {
  Warn "Skipped. Questions will use keyword search."
} elseif (Test-Qdrant) {
  Good "Qdrant is already running"
} elseif (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Warn "Docker is not installed. That's fine: reviews work, questions use keyword search."
} else {
  & docker info *> $null
  if ($LASTEXITCODE -ne 0) {
    Warn "Docker Desktop is not running. That's fine: reviews work, questions use keyword search."
    Warn "Start Docker Desktop and re-run this script if you want semantic search."
  } else {
    & docker compose up -d qdrant
    if ($LASTEXITCODE -ne 0) { & docker-compose up -d qdrant }
    $ready = $false
    for ($i = 0; $i -lt 30 -and -not $ready; $i++) { if (Test-Qdrant) { $ready = $true } else { Start-Sleep -Seconds 1 } }
    if ($ready) { Good "Qdrant started" } else { Warn "Qdrant did not respond. Continuing without it." }
  }
}

# ---- 5. start the app ----
Say "5/5  Starting LexiGuard"
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  try {
    $h = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 3
    Good "LexiGuard is already running. Opening it."
    Start-Process $Url
    exit 0
  } catch {
    Fail "Port $Port is used by another program. Close it or run:  .\run.ps1 -Port 8010"
  }
}

# open the browser only after the server answers (the old script opened it too early)
$null = Start-Job -ArgumentList $Url -ScriptBlock {
  param($u)
  for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod -Uri "$u/api/health" -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep -Seconds 1 }
  }
}

$env:OLLAMA_MODEL = $Model
$env:LEXIGUARD_PORT = "$Port"
Write-Host ""
Write-Host "  LexiGuard is starting at $Url" -ForegroundColor Green
Write-Host "  Keep this window open while you use it. Press Ctrl+C to stop." -ForegroundColor Green
Write-Host ""
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
