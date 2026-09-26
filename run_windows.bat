@echo off
REM Starts the API and the dashboard in two windows
cd /d "%~dp0"
start "Engine API" cmd /k python -m uvicorn api.main:app --port 8000
timeout /t 5 >nul
python -m streamlit run dashboard/app.py
