@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Visitor Tracking System

echo ============================================================
echo Visitor Tracking System - setup and launch
echo ============================================================

where py >nul 2>nul
if not errorlevel 1 (
  py -3.10 -c "import sys; assert sys.version_info[:2] == (3,10)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3.10"
)

if not defined PYTHON_CMD (
  python -c "import sys; assert sys.version_info[:2] == (3,10)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
  echo ERROR: Python 3.10 was not found.
  echo Install Python 3.10 64-bit, tick "Add Python to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python 3.10 virtual environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 goto :venv_error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :venv_error

python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :install_error

REM Remove the old drawable-canvas packages. This build uses a simpler
REM click-point editor that does not depend on Fabric.js canvas JSON.
python -m pip uninstall -y streamlit-drawable-canvas streamlit-drawable-canvas-fix >nul 2>nul
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :install_error

echo Checking/downloading model files...
python download_models.py
if errorlevel 1 goto :model_error

echo Running deterministic setup checks...
python verify_setup.py
if errorlevel 1 goto :verify_error

echo YouTube URL support: enabled through yt-dlp.
echo Starting Streamlit...
python -m streamlit run app.py
pause
exit /b 0

:venv_error
echo.
echo ERROR: The Python virtual environment could not be created or activated.
pause
exit /b 1

:install_error
echo.
echo ERROR: Required Python packages could not be installed.
echo Check the internet connection and company firewall, then run this file again.
pause
exit /b 1

:model_error
echo.
echo ERROR: YOLO or OSNet model download/validation failed.
echo Check the internet connection and run this file again.
pause
exit /b 1

:verify_error
echo.
echo ERROR: A deterministic project check failed. Copy the complete error shown above.
pause
exit /b 1
