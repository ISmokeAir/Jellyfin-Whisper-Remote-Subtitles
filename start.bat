@echo off
title Jellyfin Whisper Worker
echo ============================================
echo   Jellyfin Whisper Worker
echo ============================================
echo.
echo Starting daemon... (press Ctrl+C to stop)
echo Log file: logs\whisper_worker.log
echo.

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3 "%~dp0main.py" %*
) else (
    python "%~dp0main.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Worker exited with an error.
    pause
)

