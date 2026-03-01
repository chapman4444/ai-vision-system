@echo off
echo Starting Claude Vision Service...
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Start the vision service in daemon mode
echo Starting vision service daemon...
python claude_service.py daemon

echo Vision service started. Check status with:
echo   python claude_service.py status
echo.
echo To stop the service:
echo   python claude_service.py stop
echo.
pause