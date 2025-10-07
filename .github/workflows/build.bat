@echo off
REM Lokales Build-Script für os-materialmaker (Windows)
REM Erstellt ausführbare Dateien für Windows

echo.
echo 🔧 os-materialmaker Build Script (Windows)
echo ==========================================

REM Finde die neueste Version
for /f "delims=" %%i in ('dir /b os-materialmaker_*.py ^| sort /r') do (
    set LATEST_FILE=%%i
    goto :found
)
:found

REM Extrahiere Version
for /f "tokens=2 delims=_." %%a in ("%LATEST_FILE%") do set VERSION=%%a

echo 📄 Neueste Version: %VERSION%
echo 📄 Datei: %LATEST_FILE%

REM Überprüfe Python
echo.
echo 🐍 Überprüfe Python-Installation...
python --version
if errorlevel 1 (
    echo ❌ Python nicht gefunden! Bitte Python 3.11+ installieren.
    pause
    exit /b 1
)

REM Installiere Dependencies
echo.
echo 📦 Installiere Dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️ requirements.txt nicht gefunden, installiere manuell...
    pip install pillow pyinstaller requests
)

REM Syntax-Check
echo.
echo ✅ Syntax-Check...
python -m py_compile %LATEST_FILE%
if errorlevel 1 (
    echo ❌ Syntax-Fehler in %LATEST_FILE%
    pause
    exit /b 1
)

REM Erstelle Build-Verzeichnis
echo.
echo 📁 Erstelle Build-Verzeichnis...
if not exist dist mkdir dist
if not exist build mkdir build

set OUTPUT_NAME=os-materialmaker-%VERSION%-windows.exe
set PORTABLE_DIR=portable-windows

echo.
echo 🔨 Erstelle Executable für Windows...
echo 🎯 Output: %OUTPUT_NAME%

REM PyInstaller Build
pyinstaller --onefile ^
  --windowed ^
  --name "os-materialmaker-%VERSION%-windows" ^
  --icon="Resources/icons/app-icon.ico" ^
  --add-data "Resources;Resources" ^
  --hidden-import="PIL._tkinter_finder" ^
  --hidden-import="tkinter" ^
  --hidden-import="tkinter.ttk" ^
  --hidden-import="tkinter.filedialog" ^
  --hidden-import="tkinter.messagebox" ^
  --collect-submodules="PIL" ^
  --clean ^
  %LATEST_FILE%

if errorlevel 1 (
    echo ❌ PyInstaller Build fehlgeschlagen!
    pause
    exit /b 1
)

REM Erstelle Portable Package
echo.
echo 📦 Erstelle Portable Package...
if exist %PORTABLE_DIR% rmdir /s /q %PORTABLE_DIR%
mkdir %PORTABLE_DIR%

copy dist\%OUTPUT_NAME% %PORTABLE_DIR%\
if exist Resources xcopy /e /i Resources %PORTABLE_DIR%\Resources
if exist README.md copy README.md %PORTABLE_DIR%\
if exist LICENSE copy LICENSE %PORTABLE_DIR%\

REM Erstelle Start-Script
echo @echo off > %PORTABLE_DIR%\start.bat
echo echo Starting os-materialmaker v%VERSION%... >> %PORTABLE_DIR%\start.bat
echo echo. >> %PORTABLE_DIR%\start.bat
echo %OUTPUT_NAME% >> %PORTABLE_DIR%\start.bat
echo pause >> %PORTABLE_DIR%\start.bat

REM Erstelle ZIP
echo.
echo 🗜️ Erstelle ZIP-Archiv...
powershell Compress-Archive -Path %PORTABLE_DIR%\* -DestinationPath os-materialmaker-%VERSION%-windows-portable.zip -Force

echo.
echo ✅ Build erfolgreich abgeschlossen!
echo.
echo 📁 Generierte Dateien:
echo    • dist\%OUTPUT_NAME%
echo    • os-materialmaker-%VERSION%-windows-portable.zip
echo.
echo 🚀 Zum Testen: cd %PORTABLE_DIR% ^&^& start.bat
echo.
pause