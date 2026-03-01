@echo off
title AI Vision - Install Windows Service
echo Installing AI Vision as Windows Service...
cd /d "%~dp0"

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing Windows service...
python vision_service.py install

echo.
echo Service installed successfully!
echo.
echo Commands:
echo   net start AIVisionService    - Start service
echo   net stop AIVisionService     - Stop service
echo   sc delete AIVisionService    - Remove service
echo.
pause