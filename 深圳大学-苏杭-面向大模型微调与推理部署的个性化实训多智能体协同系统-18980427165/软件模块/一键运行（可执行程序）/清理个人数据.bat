@echo off
setlocal
title Clean Personal Data
cd /d "%~dp0"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" (
    echo [FATAL] PowerShell not found at: %PS%
    pause
    exit /b 1
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0_clean_personal.ps1"
set "RC=%ERRORLEVEL%"
exit /b %RC%
