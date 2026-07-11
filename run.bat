@echo off
setlocal

cd /d "%~dp0"

set "APP_URL=http://127.0.0.1:5000/dashboard"
set "PYTHON_EXE=C:\Users\1\AppData\Local\Programs\Python\Python311\python.exe"

if exist "%PYTHON_EXE%" (
    start "" "%APP_URL%"
    "%PYTHON_EXE%" app.py
) else (
    echo 未找到 Python 解释器：
    echo %PYTHON_EXE%
    echo.
    echo 请修改 run.bat 中的 PYTHON_EXE 路径后重试。
    pause
)
