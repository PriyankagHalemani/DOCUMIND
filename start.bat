@echo off
title DocuChat AI Launcher
echo ========================================================
echo   Starting DocuChat AI Full-Stack Application...
echo ========================================================
cd /d "%~dp0backend"
echo Serving API and Frontend at http://127.0.0.1:8000
python run.py
pause
