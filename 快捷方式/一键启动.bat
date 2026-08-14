@echo off
setlocal
REM ============================================================
REM  AI Agent Trainer - One-click Launcher
REM  Place: <project-root>\快捷方式\一键启动.bat
REM  Double-click to start: Ollama + Backend + Frontend,
REM  then open the UI as a desktop-style window (Edge/Chrome --app).
REM ============================================================
title AI Agent Trainer

REM --- Project root is the parent dir (because this .bat lives in 快捷方式\) ---
cd /d "%~dp0\.."
set "ROOT=%cd%"

REM --- Fixed ports per project defaults ---
set "B_PORT=8500"
set "F_PORT=5173"

REM --- Python: prefer venv, otherwise PATH python ---
set "PY=python"
if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"

echo.
echo ============================================
echo   AI Agent Trainer - Starting
echo ============================================
echo   Project  : %ROOT%
echo   Backend  : http://localhost:%B_PORT%
echo   Frontend : http://localhost:%F_PORT%
echo   Python   : %PY%
echo.

REM --- [1/3] Ollama serve ---
start "[1] Ollama" cmd /k "title [1/3] Ollama Serve && cd /d "%ROOT%" && ollama serve"
timeout /t 1 /nobreak >nul

REM --- [2/3] Backend ---
start "[2] Backend" cmd /k "title [2/3] Backend port %B_PORT% && cd /d "%ROOT%\backend" && set PYTHONUNBUFFERED=1 && "%PY%" main.py"
timeout /t 2 /nobreak >nul

REM --- [3/3] Frontend ---
start "[3] Frontend" cmd /k "title [3/3] Frontend port %F_PORT% && cd /d "%ROOT%\frontend" && npm run dev"

REM --- Wait for services, then open desktop-style App window ---
echo   Waiting for services...
timeout /t 6 /nobreak >nul

set "BROWSER="
where msedge >nul 2>nul
if %ERRORLEVEL%==0 set "BROWSER=msedge"
if not defined BROWSER (
    where chrome >nul 2>nul
    if %ERRORLEVEL%==0 set "BROWSER=chrome"
)

if defined BROWSER (
    echo   Opening desktop-app window via %BROWSER%...
    start "" "%BROWSER%" --app="http://localhost:%F_PORT%"
) else (
    echo   Opening default browser...
    start "" "http://localhost:%F_PORT%"
)

echo.
echo   DONE. Close the three cmd windows to stop all services.
echo.
timeout /t 3 /nobreak >nul
endlocal
exit /b 0
