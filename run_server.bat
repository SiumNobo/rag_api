@echo off
echo ==========================================
echo   Starting RAG API Server
echo ==========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Load environment variables
if exist .env (
    for /f "tokens=*" %%a in (.env) do (
        set %%a
    )
)

echo Starting server at http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
