@echo off
rem Double-click to start LexiGuard. Bypasses PowerShell's script-blocking policy for this one run only.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 pause
