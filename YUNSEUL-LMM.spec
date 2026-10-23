# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

BASE = Path(r"D:\workspace\YUNSEUL-LMM")

a = Analysis(
    [str(BASE / "widget_app.py")],
    pathex=[str(BASE)],
    binaries=[],
    datas=[
        (str(BASE / "app_icon.ico"), "."),
        (str(BASE / "app_icon.png"), ".")
    ],
    hiddenimports=[
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtMultimedia",
        "PyQt6.QtNetwork",
        "pynvml",
        "mutagen",
        "mutagen.id3",
        "voice_clone_pane",
        "voice_script",
        "mp3_meta",
        "song_studio",
        "lyrics_writer",
        "ebook_writer",
        "ebook_persona",
        "voice_catalog",
        "voice_record_studio",
        "voice_scripts",
        "voice_enroll",
        "voice_idealize",
        "song_finalize",
        "vocal_coach",
        "vocal_post",
        "svc",
        "mp3_lyrics_embed",
        "intro_image",
        "text_engine",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "librosa",
        "scipy",
        "matplotlib"
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="YUNSEUL-LMM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(BASE / "app_icon.ico")],
)
