@echo off
echo ==========================================
echo   Testing RAG API
echo ==========================================
echo.

call venv\Scripts\activate.bat

python test_api_local.py

pause
