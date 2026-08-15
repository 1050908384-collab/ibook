#requires -Version 5.1
param()
$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AI Agent Trainer - Launcher (PowerShell)"  -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Resolve project root (one level up from the folder holding this .ps1)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Write-Host "  Project  : $ProjectDir"
Write-Host ""

# ---------- Load config defaults + overrides from backend/.env ----------
$BackendPort  = 8500
$FrontendPort = 5173
$EnvFile = Join-Path $ProjectDir "backend\.env"
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        $t = $line.Trim()
        if ($t.Length -eq 0 -or $t.StartsWith("#")) { continue }
        $eq = $t.IndexOf("=")
        if ($eq -le 0) { continue }
        $k = $t.Substring(0,$eq).Trim()
        $v = $t.Substring($eq+1).Trim().Trim('"').Trim("'")
        if ($k -ieq "APP_PORT") { [int]$BackendPort = $v }
    }
}

# ---------- Pick python interpreter ----------
$PythonExe = $null
$VenvPy = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (Test-Path $VenvPy) { $PythonExe = $VenvPy }
if (-not $PythonExe) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { $PythonExe = $py.Source }
}
Write-Host "  Backend  : http://localhost:$BackendPort"
Write-Host "  Frontend : http://localhost:$FrontendPort"
if ($PythonExe) { Write-Host "  Python   : $PythonExe" } else { Write-Host "  Python   : [NOT FOUND]" -ForegroundColor Yellow }
Write-Host ""

# ---------- Helpers ----------
function Test-PortListening([int]$Port) {
    try {
        $conns = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        foreach ($c in $conns) { if ($c.Port -eq $Port) { return $true } }
    } catch {}
    return $false
}
function Wait-Http200([string]$Url, [int]$TimeoutSeconds, [string]$Name) {
    for ($i=0; $i -lt $TimeoutSeconds; $i++) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $Url -ErrorAction Stop
            if ($r.StatusCode -eq 200) { Write-Host "  $Name ready." -ForegroundColor Green; return $true }
        } catch {}
        Start-Sleep -Milliseconds 950
    }
    Write-Host "[ERROR] $Name timeout - no HTTP 200 at $Url" -ForegroundColor Red
    Write-Host "        Troubleshoot:" -ForegroundColor Yellow
    Write-Host "          * Read the $Name cmd window for startup errors"
    Write-Host "          * If a zombie process holds the port, kill it then retry"
    return $false
}
function Start-CmdWindow([string]$Title, [string]$WorkingDir, [string]$CommandLine) {
    # Assemble one cmd.exe command that cmd.exe's /k will run exactly as-is:
    #   1) set title (visible in taskbar and window frame)
    #   2) cd to working dir
    #   3) run the user-provided command line (preserves ampersands/quotes in it)
    $inner = "title $Title & cd /d `"$WorkingDir`" & $CommandLine"
    # cmd.exe rule: if the first and last char of the argument after /C or /K is ",
    # cmd strips them and parses what's inside. We want our $inner parsed literally.
    $arg = 'cmd /c "' + $inner + '"'
    Start-Process -FilePath "cmd.exe" -ArgumentList ("/k",$inner) -WorkingDirectory $WorkingDir
}

# ---------- 1) Ollama ----------
if (-not (Test-PortListening 11434)) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollama) {
        Write-Host "[ERROR] ollama.exe not found in PATH." -ForegroundColor Red
        Write-Host "        Install from https://ollama.com/download and try again."
        throw "Ollama missing"
    }
    Write-Host "  Ollama not on :11434 - launching service window." -ForegroundColor Yellow
    Start-CmdWindow "[1/3] Ollama Serve" $ProjectDir "ollama serve"
} else {
    Write-Host "  Ollama already on :11434. Reusing." -ForegroundColor DarkGreen
}
if (-not (Wait-Http200 "http://127.0.0.1:11434/api/tags" 25 "Ollama")) { throw "Ollama not ready" }

# ---------- 2) Backend (python import check FIRST, then start uvicorn) ----------
if (-not (Test-PortListening $BackendPort)) {
    if (-not $PythonExe) {
        Write-Host "[ERROR] No usable python interpreter." -ForegroundColor Red
        Write-Host "        Create venv under $ProjectDir\venv OR install Python to PATH."
        throw "Python missing"
    }
    Write-Host "  Running backend import sanity check." -ForegroundColor Yellow
    $importLog = Join-Path $env:TEMP "aiat_backend_import_$(Get-Random).log"
    $probePy = Join-Path $env:TEMP "aiat_backend_probe_$(Get-Random).py"
    # Always write the probe as a .py file - this sidesteps every possible quoting /
    # token-splitting / escape hell between PS -> cmd -> python -c "...".
    $backendDir = Join-Path $ProjectDir "backend"
    $probeContent = @"
import sys
sys.path.insert(0, r'$backendDir')
import main  # noqa: F401  (let any import error surface)
print('backend import OK; app=', type(main.app).__name__)
sys.exit(0)
"@
    [System.IO.File]::WriteAllText($probePy, $probeContent, [System.Text.UTF8Encoding]::new($false))
    $tmpOut = $importLog + ".out"
    $tmpErr = $importLog + ".err"
    $proc = Start-Process -FilePath $PythonExe -ArgumentList $probePy `
        -WorkingDirectory $backendDir `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    $sb = New-Object System.Text.StringBuilder
    if (Test-Path $tmpOut) { [void]$sb.AppendLine((Get-Content -Raw $tmpOut)) }
    if (Test-Path $tmpErr) { [void]$sb.AppendLine((Get-Content -Raw $tmpErr)) }
    [System.IO.File]::WriteAllText($importLog, $sb.ToString())
    Remove-Item $tmpOut -ErrorAction SilentlyContinue
    Remove-Item $tmpErr -ErrorAction SilentlyContinue
    Remove-Item $probePy -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        Write-Host "[ERROR] Backend import FAILED (python exit $($proc.ExitCode))." -ForegroundColor Red
        Write-Host "        Usually a missing pip dependency or a syntax error."
        Write-Host "        --- diagnostic log ---" -ForegroundColor Magenta
        if (Test-Path $importLog) { Get-Content $importLog }
        Write-Host "        --- end of log ---"        -ForegroundColor Magenta
        Write-Host ""
        Write-Host "        Fix: activate venv (if any), then run:  pip install -r requirements.txt"
        throw "Backend import failed"
    } else {
        Write-Host "  Backend import OK." -ForegroundColor Green
    }
    Write-Host "  Starting uvicorn on :$BackendPort ..." -ForegroundColor Yellow
    $cmdLine = "set PYTHONUNBUFFERED=1 & ""$PythonExe"" -m uvicorn main:app --host 0.0.0.0 --port $BackendPort"
    Start-CmdWindow "[2/3] Backend :$BackendPort" (Join-Path $ProjectDir "backend") $cmdLine
} else {
    Write-Host "  Backend already on :$BackendPort. Reusing." -ForegroundColor DarkGreen
}
if (-not (Wait-Http200 "http://127.0.0.1:$BackendPort/health" 40 "Backend")) { throw "Backend not ready" }

# ---------- 3) Frontend (auto `npm install` when vite.cmd missing) ----------
if (-not (Test-PortListening $FrontendPort)) {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        Write-Host "[ERROR] npm.cmd not in PATH. Install Node.js 18+ LTS." -ForegroundColor Red
        throw "npm.cmd missing"
    }
    $FrontendDir = Join-Path $ProjectDir "frontend"
    $ViteCmd     = Join-Path $FrontendDir "node_modules\.bin\vite.cmd"
    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        New-Item -ItemType Directory -Path (Join-Path $FrontendDir "node_modules") -Force | Out-Null
    }
    if (-not (Test-Path $ViteCmd)) {
        Write-Host ""
        Write-Host "  NOTICE: node_modules\.bin\vite.cmd missing." -ForegroundColor Yellow
        Write-Host "  Launcher will run 'npm install --no-audit --no-fund' under frontend now."
        Write-Host "  First run may take 1-5 minutes. Be online."
        Write-Host ""
        Push-Location $FrontendDir
        try {
            & $npm.Source install --no-audit --no-fund
            $rc = $LASTEXITCODE
        } finally { Pop-Location }
        if ($rc -ne 0) {
            Write-Host "[ERROR] npm install returned non-zero ($rc)." -ForegroundColor Red
            Write-Host "        Common fixes:"
            Write-Host "          * npm config set registry https://registry.npmmirror.com"
            Write-Host "          * Delete frontend\node_modules and frontend\package-lock.json; then retry"
            Write-Host "          * Make sure you can reach the configured npm registry."
            throw "npm install failed"
        }
        if (-not (Test-Path $ViteCmd)) {
            Write-Host "[ERROR] npm install succeeded but vite.cmd still missing:" -ForegroundColor Red
            Write-Host "        $ViteCmd"
            Write-Host "        Check frontend\package.json -> devDependencies contains 'vite'."
            throw "vite.cmd still missing after install"
        }
        Write-Host "  npm install completed. Frontend deps ready." -ForegroundColor Green
    }
    Write-Host "  Starting Vite dev server on :$FrontendPort ..." -ForegroundColor Yellow
    # Double-quoting the npm.cmd path keeps it safe when %PATH% has spaces;
    # the rest of the line is the npm "run" subcommand and flags to Vite.
    $cmdLine = """$($npm.Source)"" run dev -- --host 127.0.0.1 --port $FrontendPort"
    Start-CmdWindow "[3/3] Frontend :$FrontendPort" $FrontendDir $cmdLine
} else {
    Write-Host "  Frontend already on :$FrontendPort. Reusing." -ForegroundColor DarkGreen
}
if (-not (Wait-Http200 "http://127.0.0.1:$FrontendPort/" 45 "Frontend")) { throw "Frontend not ready" }

# ---------- 4) Browser in --app (desktop-PWA) mode. ----------
$BrowserExe = $null
$ms = Get-Command msedge -ErrorAction SilentlyContinue
if ($ms) { $BrowserExe = $ms.Source }
if (-not $BrowserExe) {
    $cr = Get-Command chrome -ErrorAction SilentlyContinue
    if ($cr) { $BrowserExe = $cr.Source }
}
$candidates = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
)
foreach ($c in $candidates) {
    if (-not $BrowserExe -and (Test-Path $c)) { $BrowserExe = $c; break }
}

$AppUrl = "http://localhost:$FrontendPort/"
if ($BrowserExe) {
    Write-Host "  Opening desktop-app window via:"
    Write-Host "     $BrowserExe"
    Start-Process -FilePath $BrowserExe -ArgumentList "--app=`"$AppUrl`""
} else {
    Write-Host "  Neither Edge nor Chrome detected. Opening system default browser..."
    Start-Process $AppUrl
}

Write-Host ""
Write-Host "  DONE. All services up. Close the 3 cmd windows to stop." -ForegroundColor Green
Write-Host ""
exit 0