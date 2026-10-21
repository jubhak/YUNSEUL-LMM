# -*- coding: utf-8 -*-
"""MP3 ID3 태그: 제목, 가사, 커버 이미지를 파일 안에 넣는다."""
from pathlib import Path


def embed_mp3_meta(mp3_path, title=None, lyrics=None, cover_path=None, artist="YUNSEUL-LMM"):
    """기존 태그는 유지하고 주어진 항목만 갱신한다. mutagen 이 없으면 False."""
    path = Path(mp3_path)
    if not path.exists() or path.suffix.lower() != ".mp3":
        return False
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError, TIT2, TPE1, TALB, TCON, USLT, APIC
    except ImportError:
        return False

    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()

    if title and str(title).strip():
        tags["TIT2"] = TIT2(encoding=3, text=str(title).strip())
    tags["TPE1"] = TPE1(encoding=3, text=artist)
    tags["TALB"] = TALB(encoding=3, text="YUNSEUL Song Studio")
    tags["TCON"] = TCON(encoding=3, text="K-Pop")

    if lyrics is not None:
        text = str(lyrics).strip()
        for key in list(tags.keys()):
            if key.startswith("USLT"):
                del tags[key]
        if text:
            tags["USLT::kor"] = USLT(encoding=3, lang="kor", desc="", text=text)

    if cover_path:
        img = Path(cover_path)
        if img.exists():
            data = img.read_bytes()
            mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
            for key in list(tags.keys()):
                if key.startswith("APIC"):
                    del tags[key]
            tags["APIC"] = APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data)

    tags.save(str(path), v2_version=3)
    return True


def read_cover_bytes(mp3_path):
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(mp3_path))
        for key in tags.keys():
            if key.startswith("APIC"):
                return bytes(tags[key].data)
    except Exception:
        return None
    return None
