@echo off
cd /d "%~dp0"

echo ============================================
echo  OVA Reinstall
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
echo Running fresh install...
"C:\Program Files\Git\bin\bash.exe" ova.sh install

echo.
echo ============================================
echo  Reinstall complete.
echo ============================================

pause
