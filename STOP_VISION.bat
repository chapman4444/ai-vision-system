@echo off
echo Stopping Claude Vision Service...
cd /d "%~dp0"

python claude_service.py stop

echo Vision service stop command sent.
pause