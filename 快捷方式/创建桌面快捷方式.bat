@echo off
setlocal
title Create Desktop Shortcut - AI Agent Trainer

REM --- go to the folder this .bat lives in ---
cd /d "%~dp0"

echo.
echo ============================================
echo   Create Desktop / Start-Menu Shortcut
echo ============================================
echo.

if not exist "_create_shortcut.vbs" (
    echo [ERROR] Missing companion file "_create_shortcut.vbs"
    echo         It should be in the same folder as this .bat.
    echo.
    pause
    exit /b 2
)

REM --- Use cscript so all output prints in this cmd window ---
cscript //nologo "%~dp0_create_shortcut.vbs"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Finished successfully (exit code 0).
) else (
    echo   FAILED with exit code %RC%. See messages above.
)
echo.
pause
endlocal
exit /b %RC%
