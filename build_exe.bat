@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Construyendo Drone People Counter Pro (.exe)
echo ============================================================
echo.

python -m PyInstaller --clean --noconfirm DronePeopleCounterPro.spec
if errorlevel 1 (
    echo.
    echo [ERROR] La construccion fallo. Revisa la salida de PyInstaller.
    exit /b 1
)

echo.
echo ============================================================
echo  Compilacion finalizada
echo  Ejecutable:
echo    %cd%\dist\DronePeopleCounterPro\DronePeopleCounterPro.exe
echo.
echo  La primera vez se copiaran config.json y se crearan:
echo    logs\    exports\
echo  junto al ejecutable.
echo ============================================================
pause
