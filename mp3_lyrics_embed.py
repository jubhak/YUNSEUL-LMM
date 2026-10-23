#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp3_lyrics_embed.py - 만든 노래 mp3 에 '가사'와 '커버(인트로) 이미지'를 실제 태그로 심는다.

들어가는 태그
  USLT : 가사 전체 (동기화 없음, 플레이어 가사창에 그대로 표시)
  SYLT : 줄별 동기화 가사 (지원 플레이어는 노래에 맞춰 줄이 넘어감)
  APIC : 커버/인트로 이미지 (앨범아트, type=3 front cover)
  TIT2/TPE1/TALB/TCON/TDRC : 제목·가수·앨범·장르·연도
  + 같은 폴더에 .lrc 파일도 함께 저장 (범용 플레이어용)

줄 타이밍 추정
  가사 줄 수와 각 줄의 음절 수를 이용해 곡 길이에 비례 배분하고,
  섹션 태그([Verse] 등) 사이에는 짧은 간주를 넣는다. 실제 가창 타이밍과
  완전히 일치하지는 않으므로, 정확한 싱크가 필요하면 --timing 으로
  [{"line": "...", "ms": 12000}, ...] 형태의 JSON 을 넘기면 된다.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TAG_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
LRC_TS = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
HANGUL = re.compile(r"[\uac00-\ud7a3]")

DEFAULT_ARTIST = "YUNSEUL-LMM"
DEFAULT_ALBUM = "YUNSEUL-LMM Studio"


# ---------------------------------------------------------------- 텍스트 처리
def count_syllables(line: str) -> int:
    """한글은 글자당 1음절, 영문/숫자는 대략적인 음절수로 환산."""
    s = re.sub(r"\(.*?\)", "", line)
    ko = len(HANGUL.findall(s))
    etc = re.sub(r"[\uac00-\ud7a3]", " ", s)
    words = [w for w in re.split(r"\s+", etc) if w]
    est = 0
    for w in words:
        n = len(re.findall(r"[aeiouyAEIOUY]+", w)) or max(1, len(w) // 3)
        est += n
    return max(1, ko + est)


def split_lyrics(text: str):
    """(섹션태그, 순수 가사줄 목록) 으로 분해."""
    clean_lines, tags = [], []
    for raw in (text or "").splitlines():
        ln = LRC_TS.sub("", raw).strip()
        if not ln:
            continue
        if TAG_LINE.match(ln):
            tags.append(ln)
            clean_lines.append("")          # 섹션 경계 표시용 빈 줄
        else:
            clean_lines.append(ln)
    return tags, clean_lines


def stripped_lyrics(text: str) -> str:
    _, lines = split_lyrics(text)
    out, blank = [], 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank == 1 and out:
                out.append("")
            continue
        blank = 0
        out.append(ln)
    return "\n".join(out).strip()


def estimate_timing(text: str, duration_sec: float, lead_in: float = 2.5,
                    gap_between_sections: float = 1.6) -> list:
    """줄별 시작 시각(ms) 추정."""
    _, lines = split_lyrics(text)
    if not lines or duration_sec <= 5:
        return []
    usable = max(5.0, duration_sec - lead_in - 2.0)
    weights, sections = [], 0
    for ln in lines:
        if not ln:
            sections += 1
            weights.append(0.0)
            continue
        weights.append(float(count_syllables(ln)) + 1.6)
    total_w = sum(weights) + sections * (gap_between_sections * 2.0)
    if total_w <= 0:
        return []
    t = lead_in
    scale = usable / total_w
    out = []
    for ln, w in zip(lines, weights):
        if not ln:
            t += gap_between_sections * 2.0 * scale
            continue
        out.append({"line": ln, "ms": int(round(t * 1000))})
        t += w * scale
    return out


def make_lrc(timed: list, title: str = "", artist: str = DEFAULT_ARTIST,
             album: str = "", by: str = "YUNSEUL-LMM") -> str:
    head = []
    if title:
        head.append(f"[ti:{title}]")
    if artist:
        head.append(f"[ar:{artist}]")
    if album:
        head.append(f"[al:{album}]")
    head.append(f"[by:{by}]")
    body = []
    for it in timed:
        ms = int(it["ms"])
        m, s, cs = ms // 60000, (ms // 1000) % 60, (ms % 1000) // 10
        body.append(f"[{m:02d}:{s:02d}.{cs:02d}]{it['line']}")
    return "\n".join(head + [""] + body) + "\n"


def probe_duration(path) -> float:
    exe = "ffprobe"
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


# ---------------------------------------------------------------- 태그 쓰기
def embed_song(path, lyrics: str, title: str = "", artist: str = DEFAULT_ARTIST,
               album: str = DEFAULT_ALBUM, genre: str = "", cover: str | None = None,
               duration: float | None = None, timing: list | None = None,
               comments: str = "") -> dict:
    """mp3 에 USLT/SYLT/APIC/기본 태그를 기록. mutagen 이 없으면 .lrc 만 남기고 보고한다."""
    path = Path(path)
    result = {"file": str(path), "written": {}, "warnings": []}
    if not path.exists():
        raise FileNotFoundError(path)

    _, raw_lines = split_lyrics(lyrics)
    plain = stripped_lyrics(lyrics)
    dur = float(duration) if duration else probe_duration(path)
    timed = timing or estimate_timing(lyrics, dur)
    if not timed:
        result["warnings"].append("가사 줄이 없어 동기화 가사를 건너뛰었습니다.")

    lrc_path = path.with_suffix(".lrc")
    try:
        lrc_path.write_text(make_lrc(timed, title, artist, album), encoding="utf-8")
        result["written"]["lrc"] = str(lrc_path)
    except Exception as e:  # noqa: BLE001
        result["warnings"].append(f"LRC 저장 실패: {e}")

    try:
        from mutagen.id3 import (ID3, APIC, SYLT, USLT, TALB, TCON, TDRC, TIT2,
                                 TPE1, COMM, ID3NoHeaderError)
    except Exception:
        result["warnings"].append("mutagen 미설치 → .lrc 파일만 저장했습니다 "
                                  "(pip install mutagen 후 다시 실행하면 태그가 들어갑니다).")
        result["synced_lines"] = len(timed)
        return result

    try:
        try:
            tags = ID3(str(path))
        except ID3NoHeaderError:
            tags = ID3()
        if title:
            tags.delall("TIT2")
            tags.add(TIT2(encoding=3, text=title))
        if artist:
            tags.delall("TPE1")
            tags.add(TPE1(encoding=3, text=artist))
        if album:
            tags.delall("TALB")
            tags.add(TALB(encoding=3, text=album))
        if genre:
            tags.delall("TCON")
            tags.add(TCON(encoding=3, text=genre))

        tags.delall("USLT")
        if plain:
            tags.add(USLT(encoding=3, lang="kor", desc="lyrics", text=plain))
            result["written"]["USLT"] = True

        tags.delall("SYLT")
        if timed:
            pairs = [[it["line"], int(it["ms"])] for it in timed]
            # format=2 (milliseconds), type=1 (lyrics)
            tags.add(SYLT(encoding=3, lang="kor", format=2, type=1, desc="", text=pairs))
            result["written"]["SYLT"] = len(pairs)

        if comments:
            tags.delall("COMM")
            tags.add(COMM(encoding=3, lang="kor", desc="", text=comments))

        if cover and Path(cover).exists():
            data = Path(cover).read_bytes()
            mime = "image/png" if str(cover).lower().endswith(".png") else "image/jpeg"
            tags.delall("APIC")
            tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=data))
            result["written"]["APIC"] = f"{len(data) // 1024}KB"

        tags.save(str(path))
        result["written"]["tags_saved"] = True
    except Exception as e:  # noqa: BLE001
        result["warnings"].append(f"ID3 기록 실패: {type(e).__name__}: {e}")
    result["synced_lines"] = len(timed)
    result["duration"] = round(dur, 2)
    return result


def read_back(path) -> dict:
    """실제로 심어졌는지 다시 읽어 확인한다(검증용)."""
    info = {"file": str(path), "TIT2": None, "USLT": 0, "SYLT": 0, "APIC": 0, "TCON": None}
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(path))
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    if tags.get("TIT2"):
        info["TIT2"] = str(tags["TIT2"])
    uslt = tags.getall("USLT")
    if uslt:
        info["USLT"] = len(str(uslt[0].text or ""))
    sylt = tags.getall("SYLT")
    if sylt:
        info["SYLT"] = len(list(sylt[0].text or []))
    info["APIC"] = len(tags.getall("APIC"))
    if tags.get("TCON"):
        info["TCON"] = str(tags["TCON"])
    if tags.get("APIC"):
        apic = tags.getall("APIC")[0]
        info["APIC_mime"] = apic.mime
        info["APIC_bytes"] = len(apic.data or b"")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="mp3 에 가사(USLT/SYLT) + 커버(APIC) 삽입")
    ap.add_argument("--mp3", required=True)
    ap.add_argument("--lyrics-file", default=None)
    ap.add_argument("--lyrics", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default=DEFAULT_ARTIST)
    ap.add_argument("--album", default=DEFAULT_ALBUM)
    ap.add_argument("--genre", default="")
    ap.add_argument("--cover", default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--timing", default=None, help="줄별 타이밍 JSON")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.lyrics_file:
        lyrics = Path(a.lyrics_file).read_text(encoding="utf-8")
    else:
        lyrics = a.lyrics or ""
    timing = json.loads(Path(a.timing).read_text(encoding="utf-8")) if a.timing else None
    r = embed_song(a.mp3, lyrics, a.title, a.artist, a.album, a.genre, a.cover, a.duration, timing)
    if a.verify:
        r["read_back"] = read_back(a.mp3)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if not r["warnings"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
