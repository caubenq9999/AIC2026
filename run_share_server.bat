@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay .venv. Hay tao virtual environment va cai requirements truoc.
    pause
    exit /b 1
)

echo Dang khoi dong Team Chat tai port 5050...
".venv\Scripts\python.exe" share_server.py
pause
