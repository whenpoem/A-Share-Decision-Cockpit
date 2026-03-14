@echo off
set SCRIPT_DIR=%~dp0
powershell -NoExit -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_all.ps1"
