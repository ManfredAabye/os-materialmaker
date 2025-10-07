#!/bin/bash
# Lokales Build-Script für os-materialmaker
# Erstellt ausführbare Dateien für das aktuelle System

set -e

echo "🔧 os-materialmaker Build Script"
echo "================================"

# Finde die neueste Version
LATEST_FILE=$(ls os-materialmaker_*.py | sort -V | tail -1)
VERSION=$(echo $LATEST_FILE | sed 's/os-materialmaker_\(.*\)\.py/\1/')

echo "📄 Neueste Version: $VERSION"
echo "📄 Datei: $LATEST_FILE"

# Überprüfe Python und Dependencies
echo ""
echo "🐍 Überprüfe Python-Installation..."
python3 --version || python --version

echo ""
echo "📦 Installiere Dependencies..."
pip install -r requirements.txt || {
    echo "⚠️ requirements.txt nicht gefunden, installiere manuell..."
    pip install pillow pyinstaller requests
}

# Syntax-Check
echo ""
echo "✅ Syntax-Check..."
python3 -m py_compile $LATEST_FILE || python -m py_compile $LATEST_FILE

# Erstelle Build-Verzeichnis
echo ""
echo "📁 Erstelle Build-Verzeichnis..."
mkdir -p dist
mkdir -p build

# Bestimme Plattform
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
    EXE_EXT=".exe"
    ICON_OPTION="--icon=Resources/icons/app-icon.ico"
else
    PLATFORM="linux"
    EXE_EXT=""
    ICON_OPTION=""
fi

OUTPUT_NAME="os-materialmaker-$VERSION-$PLATFORM$EXE_EXT"

echo ""
echo "🔨 Erstelle Executable für $PLATFORM..."
echo "🎯 Output: $OUTPUT_NAME"

# PyInstaller Build
pyinstaller --onefile \
  --name "os-materialmaker-$VERSION-$PLATFORM" \
  $ICON_OPTION \
  --add-data "Resources${PATH_SEP}Resources" \
  --hidden-import="PIL._tkinter_finder" \
  --hidden-import="tkinter" \
  --hidden-import="tkinter.ttk" \
  --hidden-import="tkinter.filedialog" \
  --hidden-import="tkinter.messagebox" \
  --collect-submodules="PIL" \
  --clean \
  $LATEST_FILE

# Erstelle Portable Package
echo ""
echo "📦 Erstelle Portable Package..."
PORTABLE_DIR="portable-$PLATFORM"
mkdir -p $PORTABLE_DIR

cp dist/$OUTPUT_NAME $PORTABLE_DIR/
[ -d "Resources" ] && cp -r Resources $PORTABLE_DIR/
[ -f "README.md" ] && cp README.md $PORTABLE_DIR/
[ -f "LICENSE" ] && cp LICENSE $PORTABLE_DIR/

# Erstelle Start-Script
if [[ "$PLATFORM" == "windows" ]]; then
    cat > $PORTABLE_DIR/start.bat << EOF
@echo off
echo Starting os-materialmaker v$VERSION...
echo.
$OUTPUT_NAME
pause
EOF
    
    # Erstelle ZIP
    echo "🗜️ Erstelle ZIP-Archiv..."
    cd $PORTABLE_DIR
    7z a ../os-materialmaker-$VERSION-$PLATFORM-portable.zip . || {
        echo "⚠️ 7z nicht verfügbar, verwende PowerShell..."
        powershell Compress-Archive -Path . -DestinationPath ../os-materialmaker-$VERSION-$PLATFORM-portable.zip
    }
    cd ..
else
    cat > $PORTABLE_DIR/start.sh << EOF
#!/bin/bash
echo "Starting os-materialmaker v$VERSION..."
echo ""
./$OUTPUT_NAME
EOF
    chmod +x $PORTABLE_DIR/start.sh
    chmod +x $PORTABLE_DIR/$OUTPUT_NAME
    
    # Erstelle TAR
    echo "🗜️ Erstelle TAR-Archiv..."
    tar -czf os-materialmaker-$VERSION-$PLATFORM-portable.tar.gz -C $PORTABLE_DIR .
fi

echo ""
echo "✅ Build erfolgreich abgeschlossen!"
echo ""
echo "📁 Generierte Dateien:"
echo "   • dist/$OUTPUT_NAME"
if [[ "$PLATFORM" == "windows" ]]; then
    echo "   • os-materialmaker-$VERSION-$PLATFORM-portable.zip"
else
    echo "   • os-materialmaker-$VERSION-$PLATFORM-portable.tar.gz"
fi
echo ""
echo "🚀 Zum Testen: cd $PORTABLE_DIR && ./start.*"