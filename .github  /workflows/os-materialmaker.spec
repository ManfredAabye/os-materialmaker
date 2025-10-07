# PyInstaller spec file template für os-materialmaker
# Verwende: pyinstaller os-materialmaker.spec

# -*- mode: python ; coding: utf-8 -*-

import os

# Automatische Erkennung der neuesten Version
latest_file = max([f for f in os.listdir('.') if f.startswith('os-materialmaker_') and f.endswith('.py')])
version = latest_file.replace('os-materialmaker_', '').replace('.py', '')

block_cipher = None

a = Analysis(
    [latest_file],
    pathex=[],
    binaries=[],
    datas=[
        ('Resources', 'Resources'),  # Kopiere Resources-Ordner
    ],
    hiddenimports=[
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.simpledialog',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageOps',
        'PIL.ImageFilter',
        'PIL.ImageEnhance',
        'requests',
        'json',
        'os',
        'sys',
        'threading',
        'time',
        'struct'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'pytest',
        'setuptools'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'os-materialmaker-{version}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed mode für GUI
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Resources/icons/app-icon.ico'  # Falls Icon vorhanden
)

# Für macOS (optional)
app = BUNDLE(
    exe,
    name=f'os-materialmaker-{version}.app',
    icon='Resources/icons/app-icon.icns',
    bundle_identifier='com.manfredaabye.os-materialmaker'
)