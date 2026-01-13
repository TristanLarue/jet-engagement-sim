REM Main entrypoint to safely setup the simulation
REM The simulation will run a console and open up a website with localhost using Vpython
REM ==================== PYTHON 3.12 IS REQUIRED ====================
REM ==================== PYTHON 3.12 IS REQUIRED ====================
REM ==================== PYTHON 3.12 IS REQUIRED ====================
@echo off
setlocal
cd /d "%~dp0"
set V=.venv
set PY=py -3.12
%PY% -V >nul 2>nul || set PY=python
%PY% -c "import sys;exit(0 if sys.version_info[:2]==(3,12) else 1)" || (echo Python 3.12 required&pause&exit /b 1)
if not exist "%V%\Scripts\python.exe" %PY% -m venv "%V%" || (echo venv failed&pause&exit /b 1)
if not exist "%V%\installed.ok" ("%V%\Scripts\python.exe" -m pip install -r requirements.txt && type nul>"%V%\installed.ok") || (pause&exit /b 1)
"%V%\Scripts\python.exe" "src\main\main.py"
endlocal
