@echo off
cd /d "%~dp0"
where py >nul 2>&1
if errorlevel 1 (
    python -B launch.py
) else (
    py -3 -B launch.py
)
pause
