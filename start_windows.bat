@echo off
rem Double-click to start the Solar Battery Simulator in your browser.
rem First run: installs uv (and through it Python + dependencies) - needs internet, a few minutes.
setlocal
title Solar Battery Simulator
cd /d "%~dp0"

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv

echo uv not found - installing it (one time only)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
where uv >nul 2>nul
if not errorlevel 1 goto have_uv
echo.
echo Could not install uv. Install it manually from https://docs.astral.sh/uv/ and run this file again.
pause
exit /b 1

:have_uv
set "GUI_HOST=127.0.0.1"
set "PYTHONUTF8=1"
echo.
echo Starting the simulator - your browser will open in a moment.
echo Keep this window open while you use it; close it to stop the simulator.
echo.
uv run --frozen gui.py
if errorlevel 1 (
    echo.
    echo The simulator stopped with an error ^(see above^).
    pause
)
endlocal
