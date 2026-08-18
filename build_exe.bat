@echo off
rem  Build a standalone Windows exe into dist\TwSaifuri\
rem  NOTE: keep this file ASCII-only. cmd.exe seeks the batch file by byte
rem  offset, so non-ASCII text desyncs parsing under a different codepage.
setlocal
cd /d "%~dp0"

set "PYC="
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
    if not defined PYC set "PYC=%%P"
)
if not defined PYC (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

"%PYC%" -m PyInstaller --noconfirm --clean --windowed --name TwSaifuri ^
  --collect-all numba --collect-all llvmlite ^
  --hidden-import character_parameter --hidden-import xien_availability ^
  --hidden-import search --hidden-import solver_fast ^
  --hidden-import format_plan --hidden-import replay_plan ^
  --exclude-module config --exclude-module matplotlib --exclude-module IPython ^
  --exclude-module PIL --exclude-module pytest --exclude-module setuptools ^
  gui.py
if errorlevel 1 ( pause & exit /b 1 )

echo.
echo Running self-test...
"dist\TwSaifuri\TwSaifuri.exe" --selftest
type "%LOCALAPPDATA%\tw_saifuri_selftest.txt"
echo.
pause
