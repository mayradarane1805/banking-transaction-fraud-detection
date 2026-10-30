@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.11 or newer and try again.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto failed
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo Installing dashboard dependencies. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)

echo Starting the Banking Transaction Fraud Detection dashboard...
echo Your browser will open when the app is ready.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo The dashboard could not be started. Review the error message above.
pause
exit /b 1
