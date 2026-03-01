@echo off
title 🔍 AI VISION CLAUDE - Show Screen
color 0D

echo.
echo ========================================================
echo                CLAUDE VISION SERVICE                  
echo                     Show Claude                          
echo ========================================================
echo.

cd /d "%~dp0"

if "%~1"=="" (
    set /p MESSAGE="What do you want to show Claude?: "
) else (
    set "MESSAGE=%*"
)

if "%MESSAGE%"=="" (
    set "MESSAGE=Look at this!"
)

echo.
echo Showing Claude your screen...
echo Message: %MESSAGE%
echo Location: %CD%
echo.

python simple_claude_vision.py show "%MESSAGE%"

echo.
echo [OK] Screen captured for Claude!
echo Claude can now see your screen with context: "%MESSAGE%"
echo.

if exist "claude_workspace" (
    echo Image saved to:
    for /f %%f in ('dir /b /od "claude_workspace\claude_vision_*.png" 2^>nul') do (
        echo    * claude_workspace\%%f
        set LATEST_FILE=%%f
    )
    
    echo.
    echo Claude Instructions:
    echo    1. Claude will see your screen image
    echo    2. Claude will understand the context: "%MESSAGE%"
    echo    3. Claude can analyze and respond to what you're showing
)

echo.
pause