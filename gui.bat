@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Saifuri Optimizer

rem  NOTE: keep this file ASCII-only. cmd.exe seeks the batch file by byte
rem  offset, so non-ASCII text desyncs parsing under a different codepage.
rem
rem  Usage:  gui.bat          launch the GUI (no console)
rem          gui.bat debug    launch in console mode and keep the log

set "DBG=%~1"

rem ---- locate python -------------------------------------------------
rem  NOTE: never write  if not defined X cmd && set ...
rem  cmd evaluates && outside the if, so the set always runs. Use parens.
set "PYC="
for /f "delims=" %%P in ('where python.exe 2^>nul') do (
    if not defined PYC set "PYC=%%P"
)
if not defined PYC (
    for /f "delims=" %%P in ('where py.exe 2^>nul') do (
        if not defined PYC set "PYC=%%P"
    )
)
if not defined PYC goto :nopython

rem  console-less launcher must come from the SAME install, otherwise it
rem  may be a different python without numpy/numba
for %%P in ("!PYC!") do set "PYDIR=%%~dpP"
set "PYW=!PYC!"
if exist "!PYDIR!pyw.exe" set "PYW=!PYDIR!pyw.exe"
if exist "!PYDIR!pythonw.exe" set "PYW=!PYDIR!pythonw.exe"

rem ---- required files ------------------------------------------------
set "MISS="
for %%F in (gui.py search.py solver_fast.py character_parameter.py xien_availability.py) do (
    if not exist "%%~F" set "MISS=!MISS! %%~F"
)
if defined MISS goto :nofile

rem ---- required modules ----------------------------------------------
"!PYC!" -c "import tkinter, numpy" >nul 2>&1
if errorlevel 1 goto :nodeps

"!PYC!" -c "import numba" >nul 2>&1
if errorlevel 1 (
    echo [WARN] numba not found - it still runs on numpy alone, but ~80x slower.
    echo        To speed up:  "!PYC!" -m pip install numba
    echo.
    timeout /t 4 >nul
)

rem ---- launch --------------------------------------------------------
if defined DBG (
    echo Console mode.  python = !PYC!
    echo.
    "!PYC!" gui.py
    echo.
    echo exit code = %errorlevel%
    pause
    exit /b
)
start "" "!PYW!" gui.py
exit /b


:nopython
echo.
echo [ERROR] Python not found on PATH.
echo         Install from https://www.python.org/downloads/windows/
echo         and tick "Add python.exe to PATH".
echo.
pause
exit /b 1

:nofile
echo.
echo [ERROR] Missing file(s):!MISS!
echo         Put gui.bat in the same folder as gui.py.
echo         Current folder: %CD%
echo.
pause
exit /b 1

:nodeps
echo.
echo [ERROR] Missing modules (tkinter / numpy).
echo         Install:  "!PYC!" -m pip install numpy numba
echo         If tkinter is missing, reinstall Python with
echo         "tcl/tk and IDLE" checked.
echo.
pause
exit /b 1
