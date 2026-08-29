@echo off
setlocal
title AI Agent Trainer Launcher
cd /d "%~dp0"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
    echo [FATAL] PowerShell not found at: %PS%
    pause
    exit /b 1
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0_launcher.ps1"
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" (
echo.
echo   Launcher completed. Backend and frontend logs are in their separate cmd windows.
pause
exit /b 0
)
echo.
echo   Launcher returned error code %RC%.
echo   Scroll UP to see the [ERROR] lines above, fix the issue, then double-click again.
echo.
pause
exit /b %RC%
