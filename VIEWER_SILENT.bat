@echo off
title AI Vision System - Silent Viewer
REM AI Vision System - Silent Viewer with Auto Service Start
REM Automatically starts the vision service and launches silent viewer

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Start the vision service in daemon mode
echo Starting AI Vision Service...
python claude_service.py daemon

REM Wait a moment for service to initialize
echo Waiting for service to initialize...
ping 127.0.0.1 -n 3 > nul

REM Launch the silent viewer
echo Starting Silent Viewer...
if exist "stream_viewer_silent.pyw" (
    echo Launching viewer window...
    pythonw stream_viewer_silent.pyw
) else (
    echo Warning: stream_viewer_silent.pyw not found
    echo Displaying current_view.png directly...
    if exist "claude_session\current_view.png" (
        start claude_session\current_view.png
    ) else (
        echo Error: No capture files found
        pause
        exit /b 1
    )
)

echo.
echo AI Vision System started successfully!
echo - Vision service running in background
echo - Viewer window launched
echo - Current view: claude_session\current_view.png
echo.
echo To stop the service, run: STOP_VISION.bat