@echo off
title AI Vision - Reliable Service
echo Starting Reliable AI Vision Service...
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting vision service in standalone mode...
echo Press Ctrl+C to stop
echo.

python vision_service.py standalone

echo.
echo Vision service stopped.
pause