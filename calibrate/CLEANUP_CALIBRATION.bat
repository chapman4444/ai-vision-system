@echo off
echo Cleaning up calibration folder...

REM Create backup folder
if not exist "calibration_backup" mkdir calibration_backup

REM Move old/obsolete files to backup
move mouse_calibration_gui.py calibration_backup\ 2>nul
move run_calibration_test.py calibration_backup\ 2>nul  
move simple_calibration_test.py calibration_backup\ 2>nul
del QUICK_CALIBRATE.bat 2>nul

REM Move old calibration results to backup
move fullscreen_calibration_*.json calibration_backup\ 2>nul
move calibration_results_screenshot.png calibration_backup\ 2>nul

echo.
echo ✓ Cleanup complete!
echo.
echo ACTIVE CALIBRATION FILES:
echo - fullscreen_calibration.py (GUI)
echo - smart_auto_calibrate.py (Main entry point - run this!)
echo - MOUSE_CALIBRATION_INSTRUCTIONS.md (Documentation)
echo.
echo OLD FILES moved to: calibration_backup\
echo.
pause