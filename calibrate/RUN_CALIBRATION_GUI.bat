@echo off
echo Starting Mouse Calibration GUI for LLM Training...
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Launching calibration GUI...
python src\gui\mouse_calibration_gui.py

pause