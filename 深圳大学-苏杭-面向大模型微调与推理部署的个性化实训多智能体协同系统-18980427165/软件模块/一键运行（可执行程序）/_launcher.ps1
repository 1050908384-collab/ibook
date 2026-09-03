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
$ProjectDir = (Resolve-Path (Join-Path $ScriptDir "..\源代码")).Path
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

# ---------- 依赖完整性自检（评委复现第一步） ----------
$DepPyCheck = @('pydantic_settings', 'sqlalchemy', 'fastapi', 'chromadb')
$MissingDeps = @()
foreach ($mod in $DepPyCheck) {
    $pyCode = "import sys`ntry:`n    import $mod`n    sys.stdout.write('OK')`nexcept Exception:`n    sys.stdout.write('MISSING')`n"
    $out = (& $PythonExe -c $pyCode 2>$null | Out-String).Trim()
    if ($out -ne 'OK') { $MissingDeps += $mod }
}
if ($MissingDeps.Count -gt 0) {
    Write-Host ""
    Write-Host "[DEP] 发现缺失依赖：$($MissingDeps -join ', ')，正在为当前 Python 自动安装 requirements.txt ..." -ForegroundColor Yellow
    $ReqFile = Join-Path $ProjectDir "requirements.txt"
    & $PythonExe -m pip install --disable-pip-version-check -r $ReqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("[DEP] pip install 返回非零码，请手动执行: & ""{0}"" -m pip install -r ""{1}""" -f $PythonExe, $ReqFile) -ForegroundColor Red
        Read-Host "按回车键继续（或 Ctrl+C 退出）" | Out-Null
    } else {
        Write-Host "[DEP] requirements.txt 安装完成" -ForegroundColor Green
    }
}
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
    #   0) propagate any env vars the launcher injected (HTTPS_PROXY / DEEPSEEK_* / HTTP_PROXY)
    #      so backend/frontend cmd windows inherit them too; otherwise the user's
    #      Windows system proxy bridge goes only to the launcher PowerShell process.
    #   1) set title (visible in taskbar and window frame)
    #   2) cd to working dir
    #   3) run the user-provided command line (preserves ampersands/quotes in it)
    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($k in @("HTTPS_PROXY","HTTP_PROXY","DEEPSEEK_API_KEY","DEEPSEEK_API_BASE","DEEPSEEK_MODEL","LLM_PROVIDER","LLM_FALLBACK_PROVIDER","OLLAMA_BASE_URL","OLLAMA_MODEL","DATABASE_URL","OLLAMA_MODELS")) {
        $v = [Environment]::GetEnvironmentVariable($k)
        if (-not [string]::IsNullOrWhiteSpace($v)) {
            $escaped = $v.Replace('%', '^%')
            $parts.Add(('set "{0}={1}"' -f $k, $escaped))
        }
    }
    $titlePart = 'title ' + $Title
    $cdPart = 'cd /d "' + $WorkingDir + '"'
    $parts.Add($titlePart) | Out-Null
    $parts.Add($cdPart) | Out-Null
    $parts.Add($CommandLine) | Out-Null
    # Use a single-character separator (' & ') so PowerShell never treats the
    # ampersand as a PS operator on PS 5.1 (Chinese GBK codepage-safe).
    $inner = $parts -join ' & '
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $inner) -WorkingDirectory $WorkingDir -WindowStyle Normal
}
function Stop-ProjectFrontendProcesses([string]$FrontendPath) {
    # npm ci removes node_modules before reinstalling.  A stale Vite process
    # can keep esbuild.exe locked on Windows, so stop only Node processes whose
    # command line explicitly belongs to this project's frontend directory.
    try {
        $escapedPath = [regex]::Escape($FrontendPath)
        $processes = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" |
            Where-Object { $_.CommandLine -match $escapedPath }
        foreach ($process in $processes) {
            Write-Host "  Stopping stale project frontend process (PID $($process.ProcessId))..." -ForegroundColor Yellow
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        if ($processes) { Start-Sleep -Milliseconds 800 }
    } catch {
        Write-Host "  [WARN] Could not inspect stale frontend processes: $($_.Exception.Message)" -ForegroundColor DarkGray
    }
}
function Stop-ProjectBackendProcesses([string]$BackendPath) {
    try {
        $escapedPath = [regex]::Escape($BackendPath)
        $processes = Get-CimInstance Win32_Process |
            Where-Object {
                ($_.Name -in @('python.exe','pythonw.exe','uvicorn.exe')) -and
                (($_.CommandLine -match $escapedPath) -or ($_.CommandLine -match 'uvicorn.*main:app'))
            }
        foreach ($process in $processes) {
            Write-Host "  Stopping stale project backend process (PID $($process.ProcessId))..." -ForegroundColor Yellow
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        }
        if ($processes) { Start-Sleep -Milliseconds 800 }
    } catch {
        Write-Host "  [WARN] Could not inspect stale backend processes: $($_.Exception.Message)" -ForegroundColor DarkGray
    }
}

# ---------- 1.5) 声明路径 + 自动复制缺失的 .env 文件 ----------
$DbPath       = Join-Path $ProjectDir "data\sqlite_data\project.db"
$BackendEnv   = Join-Path $ProjectDir "backend\.env"
$FrontendEnv  = Join-Path $ProjectDir "frontend\.env.local"

# Auto-copy .env files if missing (git clone won't have them)
if (-not (Test-Path $BackendEnv)) {
    $envExample = Join-Path $ProjectDir "backend\.env.example"
    if (Test-Path $envExample) {
        Copy-Item $envExample $BackendEnv -Force
        Write-Host "  Created backend\.env from .env.example" -ForegroundColor Green
    }
}
if (-not (Test-Path $FrontendEnv)) {
    $envExample = Join-Path $ProjectDir "frontend\.env.example"
    if (Test-Path $envExample) {
        Copy-Item $envExample $FrontendEnv -Force
        Write-Host "  Created frontend\.env.local from .env.example" -ForegroundColor Green
    }
}

# ---------- 0.25) Windows 系统代理 → HTTPS_PROXY / HTTP_PROXY 进程环境变量自动注入 ----------
# 用户常见场景：在 Clash / v2rayN 里开启了「系统代理」开关（即写进了 Windows 注册表
# HKCU\...\Internet Settings 的 ProxyEnable/ProxyServer），但没把 HTTPS_PROXY 显式导出为
# 进程环境变量。而 httpx 的 trust_env=True 只会读取环境变量，不会读 Windows 注册表。
# 结果：backend/.env 里 DEEPSEEK_USE_SYSTEM_PROXY=true 仍然直连 443，ConnectTimeout。
# 这里**仅在启动脚本层**做桥接：把注册表的 ProxyServer 解析成协议前缀正确的 HTTP(S)_PROXY，
# 注入到当前 PowerShell 进程的 env 变量 → 后续启动的 Python backend 子进程自动继承。
# 完全不改任何业务代码，进程退出后无副作用，用户手动设置的 env 变量优先不覆盖。
if (
    [string]::IsNullOrWhiteSpace($env:HTTPS_PROXY) -and
    [string]::IsNullOrWhiteSpace($env:HTTP_PROXY)
) {
    try {
        $ieKey = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction SilentlyContinue
        if ($ieKey -and [int]$ieKey.ProxyEnable -eq 1 -and -not [string]::IsNullOrWhiteSpace($ieKey.ProxyServer)) {
            $proxyStr = "$($ieKey.ProxyServer)".Trim()
            $https = ""
            $http  = ""
            # ProxyServer 有两种格式：
            #   A) "127.0.0.1:7890"                       单一端口，所有协议共用
            #   B) "http=127.0.0.1:7890;https=127.0.0.1:7890;ftp=127.0.0.1:7890"   按协议拆分
            if ($proxyStr -match '=') {
                # 格式 B
                foreach ($seg in $proxyStr -split ';') {
                    if ($seg -match '^\s*https?\s*=\s*(.+?)\s*$') {
                        $hostPort = $Matches[1].Trim()
                        if (-not $hostPort.StartsWith("http")) { $hostPort = "http://" + $hostPort }
                        if ($seg -match '^\s*https\s*=') { $https = $hostPort } else { $http = $hostPort }
                    }
                }
                # 如果某协议缺失，用另一边兜底；所有都缺失则退回默认格式
                if ([string]::IsNullOrWhiteSpace($https) -and -not [string]::IsNullOrWhiteSpace($http))  { $https = $http }
                if ([string]::IsNullOrWhiteSpace($http)  -and -not [string]::IsNullOrWhiteSpace($https)) { $http  = $https }
            } else {
                # 格式 A: 单一 host:port → 所有协议共用
                if (-not $proxyStr.StartsWith("http")) { $proxyStr = "http://" + $proxyStr }
                $https = $proxyStr
                $http  = $proxyStr
            }
            if (-not [string]::IsNullOrWhiteSpace($https)) {
                $env:HTTPS_PROXY = $https
                $env:HTTP_PROXY  = $http
                Write-Host "  Windows 系统代理已桥接 → HTTPS_PROXY=$https" -ForegroundColor DarkGreen
                if (-not [string]::IsNullOrWhiteSpace($http) -and $http -ne $https) {
                    Write-Host "                                   HTTP_PROXY =$http" -ForegroundColor DarkGreen
                }
            }
        }
    } catch {
        Write-Host "  [WARN] 读取 Windows 系统代理失败: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  进程已存在 HTTPS_PROXY / HTTP_PROXY → 保留用户设置（不覆盖）。" -ForegroundColor DarkGray
}

# ---------- 0.5) Always sync seed-derived data (fast, idempotent) ----------
# 修改种子文件（knowledge_subtopics.json / knowledge_graph_relations.json）
# 后，现有 sqlite 表和 graph pickle 会静默漂移。这两步脚本每次启动都
# 跑一遍——INSERT ON CONFLICT DO UPDATE 安全幂等，只改种子派生数据，
# 永不触碰 user_subtopic_status / sessions / attempts 等学习记录。总耗时 ~2s。
if ($PythonExe) {
    $SeedScriptsDir = Join-Path $ProjectDir "scripts"
    $SeedBackendDir = Join-Path $ProjectDir "backend"
    $seedSyncOk = $true
    foreach ($seedScript in @(
        @{name="insert_knowledge_data";           desc="Seeds -> sqlite tables (blocks/subtopics/questions/resources)"},
        @{name="import_local_learning_materials"; desc="Local .md lessons -> knowledge_docs"},
        @{name="build_knowledge_graph";          desc="Seeds -> graph pickle (nodes/edges/closure)"}
    )) {
        $seedPath = Join-Path $SeedScriptsDir "$($seedScript.name).py"
        if (-not (Test-Path $seedPath)) { continue }
        $seedOut = Join-Path $env:TEMP "aiat_seed_$($seedScript.name)_out.log"
        $seedErr = Join-Path $env:TEMP "aiat_seed_$($seedScript.name)_err.log"
        Push-Location $SeedBackendDir
        try {
            # Use Start-Process so a Python traceback on stderr is captured as
            # a normal non-zero exit code instead of terminating this launcher
            # under $ErrorActionPreference = Stop.
            $seedProc = Start-Process -FilePath $PythonExe -ArgumentList @($seedPath) `
                -WorkingDirectory $SeedBackendDir -RedirectStandardOutput $seedOut `
                -RedirectStandardError $seedErr -Wait -PassThru -WindowStyle Hidden `
                -UseNewEnvironment:$false
            $rc = $seedProc.ExitCode
        } finally { Pop-Location }
        if ($rc -ne 0) {
            Write-Host "  [WARN] $($seedScript.desc) failed (exit $rc) - continuing with existing data." -ForegroundColor Yellow
            if (Test-Path $seedErr) {
                Get-Content $seedErr | Select-Object -Last 12 | ForEach-Object {
                    Write-Host "        $_" -ForegroundColor DarkYellow
                }
            }
            $seedSyncOk = $false
        }
    }
    if ($seedSyncOk) { Write-Host "  Seed data synced (sqlite + graph)." -ForegroundColor DarkGreen }
    Write-Host ""
}

# ---------- 1) Ollama (按需启用：只有当 DeepSeek 未配置或需要首次构建嵌入索引时才要求) ----------
# 先保证上面的 env-copy 跑完，再读取 backend/.env 中的 DeepSeek 配置，判断是否可以完全跳过 Ollama
$RequireOllama = $false
$DeepSeekConfigured = $false
$DbExists = Test-Path $DbPath
if (Test-Path $BackendEnv) {
    # Start with inherited environment variables so a deployment can keep
    # secrets outside backend/.env. Values in backend/.env override them when
    # explicitly non-empty.
    $dsKey   = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY")
    $dsProv  = [Environment]::GetEnvironmentVariable("LLM_PROVIDER")
    $dsFb    = [Environment]::GetEnvironmentVariable("LLM_FALLBACK_PROVIDER")
    if ($null -eq $dsKey)  { $dsKey = "" }
    if ($null -eq $dsProv) { $dsProv = "" }
    if ($null -eq $dsFb)   { $dsFb = "" }
    foreach ($line in Get-Content $BackendEnv) {
        $t = $line.Trim()
        if ($t.Length -eq 0 -or $t.StartsWith("#")) { continue }
        $eq = $t.IndexOf("=")
        if ($eq -le 0) { continue }
        $k = $t.Substring(0,$eq).Trim()
        $v = $t.Substring($eq+1).Trim().Trim('"').Trim("'")
        if ($k -ieq "APP_PORT")            { [int]$BackendPort = $v }   # 也在这里刷新端口号（env 刚复制完的情况下）
        if ($k -ieq "DEEPSEEK_API_KEY" -and -not [string]::IsNullOrWhiteSpace($v)) { $dsKey  = $v }
        if ($k -ieq "LLM_PROVIDER" -and -not [string]::IsNullOrWhiteSpace($v)) { $dsProv = $v }
        if ($k -ieq "LLM_FALLBACK_PROVIDER" -and -not [string]::IsNullOrWhiteSpace($v)) { $dsFb = $v }
    }
    # 判定 DeepSeek 是否真正配置完成（LLM_PROVIDER=deepseek 且 key 非占位符/非空）
    $placeholderPattern = '^(your-|your_|sk-your|xxx|xxxx|placeholder)'
    $keyNormalized = "$dsKey".Trim().ToLower()
    $providerOk = ($dsProv -ieq "deepseek") -or [string]::IsNullOrWhiteSpace($dsProv)  # 默认值即 deepseek
    $keyOk = -not [string]::IsNullOrWhiteSpace($dsKey) -and ($keyNormalized -notmatch $placeholderPattern)
    $fallbackNeedsOllama = ($dsFb -ieq "ollama")
    $DeepSeekConfigured = $providerOk -and $keyOk
    if ($DeepSeekConfigured) {
        $keySource = if ((Get-Content $BackendEnv) -match '^\s*DEEPSEEK_API_KEY\s*=\s*\S+') { "backend/.env" } else { "环境变量" }
        Write-Host "  DeepSeek 云端模型已检测到配置（LLM_PROVIDER=$dsProv，Key来源：$keySource）。" -ForegroundColor DarkGreen
    } else {
        Write-Host "  DeepSeek 云端模型未正确配置：将依赖本地 Ollama。" -ForegroundColor Yellow
        $RequireOllama = $true
    }
    if ($fallbackNeedsOllama) {
        Write-Host "  LLM_FALLBACK_PROVIDER=ollama 已设置：保留 Ollama 作为回退。" -ForegroundColor Yellow
        $RequireOllama = $true
    }
} else {
    Write-Host "  backend\.env 暂不存在（复制失败），默认依赖本地 Ollama。" -ForegroundColor Yellow
    $RequireOllama = $true
}
# 首次初始化且 DB 不存在时，init_knowledge_base.py 需要 Ollama 构建 Chroma 嵌入索引（EMBEDDING_MODEL）
if (-not $DbExists) {
    Write-Host "  数据库不存在，首次启动将构建向量嵌入索引（需 Ollama embedding）。" -ForegroundColor Yellow
    $RequireOllama = $true
}

if ($RequireOllama) {
    if (-not (Test-PortListening 11434)) {
        $ollama = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $ollama) {
            Write-Host "[ERROR] ollama.exe not found in PATH, but this run requires Ollama." -ForegroundColor Red
            Write-Host "        Install from https://ollama.com/download and try again."
            Write-Host "        或者：正确配置 backend/.env 中的 DEEPSEEK_API_KEY + LLM_PROVIDER=deepseek。" -ForegroundColor Yellow
            throw "Ollama missing"
        }
        Write-Host "  Ollama not on :11434 - launching service window." -ForegroundColor Yellow
        Start-CmdWindow "[1/3] Ollama Serve" $ProjectDir "ollama serve"
    } else {
        Write-Host "  Ollama already on :11434. Reusing." -ForegroundColor DarkGreen
    }
    if (-not (Wait-Http200 "http://127.0.0.1:11434/api/tags" 25 "Ollama")) { throw "Ollama not ready" }
} else {
    Write-Host "  本次运行无需本地 Ollama（跳过 :11434 检查）。" -ForegroundColor DarkGreen
}

if (-not (Test-Path $DbPath)) {
    Write-Host ""
    Write-Host "  ==================================================" -ForegroundColor Yellow
    Write-Host "    FIRST RUN: Database not found. Auto-initializing." -ForegroundColor Yellow
    Write-Host "    This takes 3-10 minutes on first run. Be patient." -ForegroundColor Yellow
    Write-Host "  ==================================================" -ForegroundColor Yellow
    Write-Host ""

    # Re-read .env now that we copied it
    if (Test-Path $BackendEnv) {
        foreach ($line in Get-Content $BackendEnv) {
            $t = $line.Trim()
            if ($t.Length -eq 0 -or $t.StartsWith("#")) { continue }
            $eq = $t.IndexOf("=")
            if ($eq -le 0) { continue }
            $k = $t.Substring(0,$eq).Trim()
            $v = $t.Substring($eq+1).Trim().Trim('"').Trim("'")
            if ($k -ieq "APP_PORT") { [int]$BackendPort = $v }
        }
    }

    if (-not $PythonExe) {
        Write-Host "[ERROR] Python is required for database initialization." -ForegroundColor Red
        Write-Host "        Install Python 3.10+ from https://python.org and retry." -ForegroundColor Yellow
        throw "Python missing"
    }

    $ScriptsDir = Join-Path $ProjectDir "scripts"
    $BackendDir = Join-Path $ProjectDir "backend"

    # Ensure data directories exist
    $dataDirs = @("data\sqlite_data", "data\chroma_db", "data\graph_data\runtime")
    foreach ($d in $dataDirs) {
        $fullPath = Join-Path $ProjectDir $d
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        }
    }

    # Step 0: pip install (skip if already installed)
    Write-Host "  [0/6] Checking Python dependencies..." -ForegroundColor Cyan
    $pipCheck = Start-Process -FilePath $PythonExe -ArgumentList "-c", "import fastapi, uvicorn, pydantic, chromadb, fastembed" `
        -Wait -PassThru -NoNewWindow -UseNewEnvironment:$false `
        -RedirectStandardOutput "$env:TEMP\aiat_pipcheck.log" -RedirectStandardError "$env:TEMP\aiat_pipcheck_err.log"
    if ($pipCheck.ExitCode -ne 0) {
        Write-Host "  Installing Python dependencies (pip install)..." -ForegroundColor Yellow
        Write-Host "  (This may take 2-5 minutes on first run)" -ForegroundColor DarkGray
        $pipProc = Start-Process -FilePath $PythonExe -ArgumentList "-m", "pip", "install", "-r", "`"$ProjectDir\requirements.txt`"" `
            -WorkingDirectory $ProjectDir -Wait -PassThru -NoNewWindow -UseNewEnvironment:$false
        if ($pipProc.ExitCode -ne 0) {
            Write-Host "[ERROR] pip install failed (exit $($pipProc.ExitCode))." -ForegroundColor Red
            Write-Host "        Try: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple" -ForegroundColor Yellow
            throw "pip install failed"
        }
        Write-Host "  pip install OK." -ForegroundColor Green
    } else {
        Write-Host "  Python dependencies already installed." -ForegroundColor DarkGreen
    }

    # Step 1-6: Run initialization scripts in dependency order
    $initScripts = @(
        @{name="init_db";                         desc="Building database schema";           needOllama=$false},
        @{name="insert_knowledge_data";           desc="Loading knowledge seeds (blocks, subtopics, docs, videos, questions)"; needOllama=$false},
        @{name="import_local_learning_materials"; desc="Importing .md lesson files into knowledge_docs"; needOllama=$false},
        @{name="build_knowledge_graph";          desc="Building knowledge graph (nodes, edges, prerequisites)"; needOllama=$false},
        @{name="init_knowledge_base";             desc="Building Chroma vector index (uses Ollama embedding model)"; needOllama=$true},
        @{name="init_agent1_semantic_index";     desc="Building Agent1 semantic index (downloads ~90MB FastEmbed model)"; needOllama=$false}
    )

    for ($i = 0; $i -lt $initScripts.Count; $i++) {
        $script = $initScripts[$i]
        $stepNum = $i + 1
        $scriptPath = Join-Path $ScriptsDir "$($script.name).py"

        if (-not (Test-Path $scriptPath)) {
            Write-Host "  [SKIP] $($script.name).py not found - skipping." -ForegroundColor DarkGray
            continue
        }

        Write-Host ""
        Write-Host "  [$stepNum/6] $($script.desc)..." -ForegroundColor Cyan

        # For scripts that need Ollama embedding, check model availability first
        if ($script.needOllama) {
            try {
                $tagsResp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
                $modelNames = $tagsResp.models | ForEach-Object { $_.name }
                $embeddingModel = "qwen3-embedding:0.6b"
                if ($modelNames -notcontains $embeddingModel) {
                    Write-Host "  Ollama embedding model '$embeddingModel' not found. Pulling now..." -ForegroundColor Yellow
                    Write-Host "  (This is a one-time ~1GB download)" -ForegroundColor DarkGray
                    $pullProc = Start-Process -FilePath "ollama" -ArgumentList "pull", $embeddingModel -Wait -PassThru -NoNewWindow
                    if ($pullProc.ExitCode -ne 0) {
                        Write-Host "[ERROR] Failed to pull $embeddingModel." -ForegroundColor Red
                        Write-Host "        Run manually: ollama pull $embeddingModel" -ForegroundColor Yellow
                        throw "Ollama model pull failed"
                    }
                    Write-Host "  Model $embeddingModel pulled OK." -ForegroundColor Green
                }
            } catch {
                Write-Host "  [WARN] Could not check Ollama models - proceeding anyway." -ForegroundColor DarkGray
            }
        }

        $outLog = Join-Path $env:TEMP "aiat_$($script.name)_out.log"
        $errLog = Join-Path $env:TEMP "aiat_$($script.name)_err.log"
        $proc = Start-Process -FilePath $PythonExe -ArgumentList $scriptPath `
            -WorkingDirectory $BackendDir -Wait -PassThru -NoNewWindow -UseNewEnvironment:$false `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog

        if ($proc.ExitCode -ne 0) {
            Write-Host "[ERROR] $($script.name) failed (exit $($proc.ExitCode))." -ForegroundColor Red
            Write-Host "  --- last 15 lines of output ---" -ForegroundColor Magenta
            if (Test-Path $outLog) { Get-Content $outLog | Select-Object -Last 15 }
            if (Test-Path $errLog) { Get-Content $errLog | Select-Object -Last 15 }
            Write-Host "  --- end ---" -ForegroundColor Magenta
            throw "$($script.name) failed"
        }
        # Show last line as success indicator
        if (Test-Path $outLog) {
            $lastLine = (Get-Content $outLog | Where-Object { $_.Trim() } | Select-Object -Last 1)
            if ($lastLine) { Write-Host "  -> $lastLine" -ForegroundColor DarkGray }
        }
        Write-Host "  $($script.name) OK." -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "  ==================================================" -ForegroundColor Green
    Write-Host "    Database initialized! Starting services..." -ForegroundColor Green
    Write-Host "  ==================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "  Database found at data\sqlite_data\project.db - skipping init." -ForegroundColor DarkGreen
}

# ---------- 1.6) Back up/migrate and verify the existing runtime database ----------
# The runtime SQLite database is intentionally excluded from Git.  A teammate
# can therefore retain an older database after pulling new code.  Detect a
# missing video catalogue here instead of letting the UI silently show no
# videos after the application has already started.
if (-not $PythonExe) {
    Write-Host "[ERROR] Python is required to verify the runtime database." -ForegroundColor Red
    throw "Python missing"
}
$DatabaseUpgradeScript = Join-Path $ProjectDir "scripts\upgrade_runtime_database.py"
$DatabaseVerifyScript = Join-Path $ProjectDir "scripts\verify_runtime_database.py"
$CatalogVerifyScript = Join-Path $ProjectDir "scripts\verify_catalog_runtime.py"
foreach ($verifyScript in @($DatabaseUpgradeScript, $DatabaseVerifyScript, $CatalogVerifyScript)) {
    if (-not (Test-Path $verifyScript)) {
        Write-Host "[ERROR] Missing database verification script:" -ForegroundColor Red
        Write-Host "        $verifyScript"
        throw "database verification script missing"
    }
}
Write-Host "  Checking whether the runtime database needs an upgrade..." -ForegroundColor Yellow
Push-Location $ProjectDir
try {
    & $PythonExe $DatabaseUpgradeScript
    $upgradeRc = $LASTEXITCODE
} finally { Pop-Location }
if ($upgradeRc -ne 0) {
    Write-Host "[ERROR] Runtime database upgrade failed. The backup (if created) is under data\sqlite_data\backups." -ForegroundColor Red
    throw "runtime database upgrade failed"
}
Write-Host "  Verifying SQLite integrity, foreign keys, and core learning data..." -ForegroundColor Yellow
$databaseOut = Join-Path $env:TEMP "aiat_database_verify_out.log"
$databaseErr = Join-Path $env:TEMP "aiat_database_verify_err.log"
Push-Location $ProjectDir
try {
    & $PythonExe $DatabaseVerifyScript 1> $databaseOut 2> $databaseErr
    $databaseRc = $LASTEXITCODE
} finally { Pop-Location }
if ($databaseRc -ne 0) {
    Write-Host "[ERROR] Runtime database health verification failed." -ForegroundColor Red
    Write-Host "        Do not delete the database. Review the output below and restore a backup if needed." -ForegroundColor Yellow
    if (Test-Path $databaseOut) { Get-Content $databaseOut | Select-Object -Last 20 }
    if (Test-Path $databaseErr) { Get-Content $databaseErr | Select-Object -Last 20 }
    throw "runtime database verification failed"
}
Write-Host "  Runtime database verified." -ForegroundColor Green
Write-Host "  Verifying knowledge graph and video catalogue..." -ForegroundColor Yellow
$catalogOut = Join-Path $env:TEMP "aiat_catalog_verify_out.log"
$catalogErr = Join-Path $env:TEMP "aiat_catalog_verify_err.log"
Push-Location $ProjectDir
try {
    & $PythonExe $CatalogVerifyScript 1> $catalogOut 2> $catalogErr
    $catalogRc = $LASTEXITCODE
} finally { Pop-Location }
if ($catalogRc -ne 0) {
    Write-Host "[ERROR] Knowledge graph or video catalogue verification failed." -ForegroundColor Red
    Write-Host "        The existing data\sqlite_data\project.db may be old or incomplete." -ForegroundColor Yellow
    Write-Host "        Stop here, then run: python scripts\insert_knowledge_data.py" -ForegroundColor Yellow
    Write-Host "        Back up data\sqlite_data\project.db first if it contains learning records."
    Write-Host "  --- verification output ---" -ForegroundColor Magenta
    if (Test-Path $catalogOut) { Get-Content $catalogOut | Select-Object -Last 20 }
    if (Test-Path $catalogErr) { Get-Content $catalogErr | Select-Object -Last 20 }
    Write-Host "  --- end ---" -ForegroundColor Magenta
    throw "catalogue verification failed"
}
Write-Host "  Knowledge graph and video catalogue verified." -ForegroundColor Green

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
    Push-Location $backendDir
    try {
        & $PythonExe $probePy 1> $tmpOut 2> $tmpErr
        $importExitCode = $LASTEXITCODE
    } finally { Pop-Location }
    $sb = New-Object System.Text.StringBuilder
    if (Test-Path $tmpOut) { [void]$sb.AppendLine((Get-Content -Raw $tmpOut)) }
    if (Test-Path $tmpErr) { [void]$sb.AppendLine((Get-Content -Raw $tmpErr)) }
    [System.IO.File]::WriteAllText($importLog, $sb.ToString())
    Remove-Item $tmpOut -ErrorAction SilentlyContinue
    Remove-Item $tmpErr -ErrorAction SilentlyContinue
    Remove-Item $probePy -ErrorAction SilentlyContinue
    if ($importExitCode -ne 0) {
        Write-Host "[ERROR] Backend import FAILED (python exit $importExitCode)." -ForegroundColor Red
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
    $cmdLine = "set PYTHONUNBUFFERED=1 & ""$PythonExe"" -m uvicorn main:app --host 0.0.0.0 --port $BackendPort --access-log"
    Start-CmdWindow "[2/3] Backend :$BackendPort" (Join-Path $ProjectDir "backend") $cmdLine
} else {
    $BackendDir = Join-Path $ProjectDir "backend"
    Write-Host "  Backend already on :$BackendPort. Restarting this project's backend to restore its log window..." -ForegroundColor Yellow
    Stop-ProjectBackendProcesses $BackendDir
    if (Test-PortListening $BackendPort) {
        Write-Host "[ERROR] Port $BackendPort is still occupied by another process; refusing to stop it." -ForegroundColor Red
        throw "backend port occupied by another process"
    }
    $cmdLine = "set PYTHONUNBUFFERED=1 & ""$PythonExe"" -m uvicorn main:app --host 0.0.0.0 --port $BackendPort --access-log"
    Start-CmdWindow "[2/3] Backend :$BackendPort" $BackendDir $cmdLine
}
if (-not (Wait-Http200 "http://127.0.0.1:$BackendPort/health" 40 "Backend")) { throw "Backend not ready" }

# ---------- 3) Frontend dependencies + Vite ----------
# A Vite executable alone does not prove that all dependencies are current.
# Record the package-lock hash after a successful install so a teammate who
# pulls a new dependency gets a deterministic `npm ci` on the next launch.
if (-not (Test-PortListening $FrontendPort)) {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        Write-Host "[ERROR] npm.cmd not in PATH. Install Node.js 18+ LTS." -ForegroundColor Red
        throw "npm.cmd missing"
    }
    $FrontendDir = Join-Path $ProjectDir "frontend"
    $ViteCmd     = Join-Path $FrontendDir "node_modules\.bin\vite.cmd"
    $PackageLock = Join-Path $FrontendDir "package-lock.json"
    $DepsStamp   = Join-Path $FrontendDir "node_modules\.aiat-package-lock.sha256"
    $NpmCacheDir = Join-Path $FrontendDir ".npm-cache"
    if (-not (Test-Path $PackageLock)) {
        Write-Host "[ERROR] frontend\package-lock.json is missing." -ForegroundColor Red
        Write-Host "        This launcher requires the committed lock file for reproducible installs."
        throw "package-lock.json missing"
    }

    $lockHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackageLock).Hash
    $installedHash = ""
    if (Test-Path $DepsStamp) {
        $installedHash = (Get-Content -Raw -LiteralPath $DepsStamp).Trim()
    }
    $NeedFrontendInstall = (-not (Test-Path $ViteCmd)) -or ($installedHash -ne $lockHash)

    # A lock-file stamp proves only that npm ci completed once. A later
    # interrupted install or manual edit can leave a missing transitive package
    # (for example rehype-katex's HTML parser) while vite.cmd still exists.
    # npm ls validates the full installed dependency tree before we reuse it.
    if (-not $NeedFrontendInstall) {
        $npmTreeLog = Join-Path $env:TEMP "aiat_npm_tree_check.log"
        Remove-Item $npmTreeLog -ErrorAction SilentlyContinue
        Write-Host "  Verifying frontend dependency tree..." -ForegroundColor Yellow
        Push-Location $FrontendDir
        try {
            & $npm.Source ls --all --omit=optional 1> $npmTreeLog 2>&1
            $npmTreeRc = $LASTEXITCODE
        } finally { Pop-Location }
        if ($npmTreeRc -ne 0) {
            $NeedFrontendInstall = $true
            Write-Host "  NOTICE: Frontend dependency tree is incomplete; rebuilding from package-lock.json." -ForegroundColor Yellow
            if (Test-Path $npmTreeLog) { Get-Content $npmTreeLog | Select-Object -Last 12 }
        }
    }

    if ($NeedFrontendInstall) {
        Write-Host ""
        if (-not (Test-Path $ViteCmd)) {
            Write-Host "  NOTICE: Frontend dependencies are not installed." -ForegroundColor Yellow
        } else {
            Write-Host "  NOTICE: frontend\package-lock.json changed; updating frontend dependencies." -ForegroundColor Yellow
        }
        Write-Host "  Launcher will run 'npm ci --no-audit --no-fund' under frontend now."
        Write-Host "  This can take 1-5 minutes and requires network access."
        Write-Host ""
        Stop-ProjectFrontendProcesses $FrontendDir
        Push-Location $FrontendDir
        try {
            & $npm.Source ci --no-audit --no-fund --cache $NpmCacheDir
            $rc = $LASTEXITCODE
        } finally { Pop-Location }
        if ($rc -ne 0) {
            Write-Host "[ERROR] npm ci returned non-zero ($rc)." -ForegroundColor Red
            Write-Host "        Common fixes:"
            Write-Host "          * npm config set registry https://registry.npmmirror.com"
            Write-Host "          * Delete frontend\node_modules; then retry"
            Write-Host "          * Make sure you can reach the configured npm registry."
            throw "npm ci failed"
        }
        if (-not (Test-Path $ViteCmd)) {
            Write-Host "[ERROR] npm ci succeeded but vite.cmd is still missing:" -ForegroundColor Red
            Write-Host "        $ViteCmd"
            Write-Host "        Check frontend\package.json -> devDependencies contains 'vite'."
            throw "vite.cmd still missing after npm ci"
        }
        [System.IO.File]::WriteAllText($DepsStamp, $lockHash + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
        Write-Host "  npm ci completed. Frontend dependencies match package-lock.json." -ForegroundColor Green
    } else {
        Write-Host "  Frontend dependencies already match package-lock.json." -ForegroundColor DarkGreen
    }
    Write-Host "  Starting Vite dev server on :$FrontendPort ..." -ForegroundColor Yellow
    # Double-quoting the npm.cmd path keeps it safe when %PATH% has spaces;
    # the rest of the line is the npm "run" subcommand and flags to Vite.
    # Keep the configured port stable. Without --strictPort Vite silently
    # falls back to 5174/5175 when 5173 is occupied, while this launcher still
    # checks 5173 and misleadingly reports a timeout.
    $FrontendLog = Join-Path $env:TEMP "aiat_frontend_vite.log"
    Remove-Item $FrontendLog -ErrorAction SilentlyContinue
    $cmdLine = """$($npm.Source)"" run dev -- --host 127.0.0.1 --port $FrontendPort --strictPort 1> ""$FrontendLog"" 2>&1"
    Start-CmdWindow "[3/3] Frontend :$FrontendPort" $FrontendDir $cmdLine
} else {
    $FrontendDir = Join-Path $ProjectDir "frontend"
    Write-Host "  Frontend already on :$FrontendPort. Restarting this project's frontend to restore its log window..." -ForegroundColor Yellow
    Stop-ProjectFrontendProcesses $FrontendDir
    if (Test-PortListening $FrontendPort) {
        Write-Host "[ERROR] Port $FrontendPort is still occupied by another process; refusing to stop it." -ForegroundColor Red
        throw "frontend port occupied by another process"
    }
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { throw "npm.cmd missing" }
    $FrontendLog = Join-Path $env:TEMP "aiat_frontend_vite.log"
    Remove-Item $FrontendLog -ErrorAction SilentlyContinue
    $cmdLine = """$($npm.Source)"" run dev -- --host 127.0.0.1 --port $FrontendPort --strictPort 1> ""$FrontendLog"" 2>&1"
    Start-CmdWindow "[3/3] Frontend :$FrontendPort" $FrontendDir $cmdLine
}
if (-not (Wait-Http200 "http://127.0.0.1:$FrontendPort/" 45 "Frontend")) {
    if ($FrontendLog -and (Test-Path $FrontendLog)) {
        Write-Host "  --- Vite startup log ---" -ForegroundColor Magenta
        Get-Content $FrontendLog | Select-Object -Last 40
        Write-Host "  --- end ---" -ForegroundColor Magenta
    }
    throw "Frontend not ready"
}

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
