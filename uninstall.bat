@echo off
cd /d "%~dp0"

echo ============================================
echo  OVA Uninstall
echo ============================================
echo.

echo Stopping running services...
"C:\Program Files\Git\bin\bash.exe" ova.sh stop

echo.
echo Removing Python virtual environment...
if exist ".venv" (
    rmdir /s /q ".venv"
    echo Virtual environment removed.
) else (
    echo No virtual environment found.
)

echo.
echo Removing runtime data (.ova)...
if exist ".ova" (
    rmdir /s /q ".ova"
    echo Runtime data removed.
) else (
    echo No runtime data found.
)

echo.
echo ============================================
echo  Uninstall complete.
echo  Note: HuggingFace model cache and Ollama
echo  models are stored globally and were not
echo  removed. To free disk space, run:
echo    uvx hf cache delete
echo    ollama rm ministral-3:3b-instruct-2512-q4_K_M
echo ============================================

pause
