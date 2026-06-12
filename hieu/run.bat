@echo off
REM RAG IR Student Server - One-Click Setup & Submit for Windows

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo === RAG IR Student Server - One-Click Setup ^& Submit ===
echo.

REM Check if uv is installed
python -m uv --version >nul 2>&1
if errorlevel 1 (
    echo [*] Installing uv package manager...
    python -m pip install uv
    echo [OK] uv installed
) else (
    echo [OK] uv found
)

REM Create virtual environment
echo [*] Creating virtual environment...
python -m uv venv .venv
echo [OK] Virtual environment created
echo.

REM Activate venv
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo [*] Installing dependencies...
python -m uv pip install -r requirements.txt
echo [OK] Dependencies installed
echo.

REM Download model
echo [*] Downloading embedding model...
python download_model.py
echo [OK] Model downloaded
echo.

REM Start server
echo [*] Starting student server...
start "RAG Server" python main.py
echo [OK] Server started
echo.

REM Wait for server
echo [*] Waiting for server to be ready...
timeout /t 3 /nobreak
for /L %%i in (1,1,30) do (
    python -c "import httpx; httpx.get('http://127.0.0.1:5004/docs', timeout=2)" >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Server is ready
        goto SERVER_READY
    )
    timeout /t 1 /nobreak
)
echo [ERROR] Server failed to start
pause
exit /b 1

:SERVER_READY
echo.

REM Register
echo [*] Registering with teacher server...
python client.py register
echo.

REM Evaluate
echo [*] Running evaluation (submitting answers)...
echo This may take 10-15 minutes. Please wait...
echo.
python client.py evaluate
echo.

REM Get result
echo [*] Fetching results...
timeout /t 2 /nobreak
python client.py result
echo.

echo === Submission Complete ===
echo.
echo Server log saved in: server.log
echo Configuration: .env
echo.
pause
