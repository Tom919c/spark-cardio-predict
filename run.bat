@echo off
setlocal

cd /d "%~dp0"
call conda run -n bigdata python app.py
