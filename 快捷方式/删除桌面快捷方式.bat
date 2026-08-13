@echo off
setlocal
title Remove Desktop Shortcut - AI Agent Trainer

cd /d "%~dp0"

echo.
echo ============================================
echo   Remove Desktop / Start-Menu Shortcut
echo ============================================
echo.

if not exist "_remove_shortcut.vbs" (
    echo [ERROR] Missing companion file "_remove_shortcut.vbs"
    echo         It should be in the same folder as this .bat.
    echo.
    pause
    exit /b 2
)

cscript //nologo "%~dp0_remove_shortcut.vbs"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Finished successfully.
    echo   TIP: If icon still appears on desktop, press F5 to refresh.
) else (
    echo   FAILED with exit code %RC%. See messages above.
)
echo.
pause
endlocal
exit /b %RC%
