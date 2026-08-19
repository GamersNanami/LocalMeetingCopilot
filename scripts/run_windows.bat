@echo off
setlocal

cd /d "%~dp0\.."

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.11 x64 from python.org.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
python run.py --live %*
