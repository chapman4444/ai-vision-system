@echo off
title 📷 AI VISION - Quick Capture
color 0E

echo.
echo ========================================================
echo                CLAUDE VISION SERVICE                  
echo                     Quick Capture                        
echo ========================================================
echo.

cd /d "%~dp0"

if "%1"=="" (
    set /p MESSAGE="Message for Claude (optional): "
) else (
    set MESSAGE=%*
)

if "%MESSAGE%"=="" (
    set MESSAGE=Quick screen capture
)

echo.
echo Taking screenshot for Claude...
echo Message: %MESSAGE%
echo Location: %CD%
echo.

python simple_claude_vision.py capture --message "%MESSAGE%"

echo.
echo [OK] Capture complete!
echo Check: claude_workspace\ folder for the image
echo.

if exist "claude_workspace" (
    echo Recent captures:
    dir /b /od "claude_workspace\claude_vision_*.png" 2>nul | findstr /r ".*" && (
        for /f %%f in ('dir /b /od "claude_workspace\claude_vision_*.png" 2^>nul') do (
            echo    * %%f
        )
    ) || (
        echo    * No captures found
    )
)

echo.
pause