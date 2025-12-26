@echo off
echo ==========================================
echo   RAG API - Windows Setup Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo [1/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
)

REM Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [4/5] Installing dependencies (this may take a few minutes)...
pip install fastapi uvicorn[standard] python-multipart
pip install PyMuPDF pdfplumber python-docx pandas
pip install pytesseract Pillow
pip install sentence-transformers
pip install faiss-cpu
pip install groq google-generativeai
pip install numpy pydantic python-dotenv
pip install httpx

REM Create .env file if not exists
echo [5/5] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo.
    echo [IMPORTANT] Please edit .env file and add your API key!
    echo.
    echo Recommended: Get a FREE Groq API key from:
    echo https://console.groq.com/keys
    echo.
)

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your GROQ_API_KEY or GEMINI_API_KEY
echo 2. Run: run_server.bat
echo.
pause
