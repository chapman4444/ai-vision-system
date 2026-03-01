@echo off
echo 🎯 Quick Mouse Calibration for LLMs
echo ================================
echo.

cd /d "%~dp0"

echo Step 1: Starting fullscreen calibration GUI...
start "Calibration GUI" python fullscreen_calibration.py

echo Waiting 5 seconds for GUI to initialize...
timeout /t 5 /nobreak >nul

echo Step 2: Running automated calibration test...
python smart_auto_calibrate.py

echo.
echo ✓ Calibration complete! 
echo Check the results file: fullscreen_calibration_*.json
echo.
echo Press any key to view instructions...
pause >nul

type MOUSE_CALIBRATION_INSTRUCTIONS.md
pause