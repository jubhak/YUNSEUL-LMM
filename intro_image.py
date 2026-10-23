#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intro_image.py - 노래의 '인트로(커버) 이미지' 생성 + 영상 인트로 합성

- 제목/장르/분위기/가사 첫 줄로 그라데이션 커버 아트를 그린다 (Pillow).
- 곡의 실제 파형을 얇은 라인으로 그려 넣어 곡마다 다른 그림이 나온다.
- 만든 이미지는 mp3 의 APIC(앨범 아트)로 들어가고, 같은 폴더에 .cover.png 로도 저장된다.
- 원하면 ffmpeg 로 커버 + 곡을 합친 '인트로 영상'(.mp4)도 만든다.

단독 실행:
  python intro_image.py --title "봄의 끝" --genre ballad_piano --moods 그리운 따뜻한 \
      --out song.cover.png --audio song.mp3
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except Exception:  # noqa: BLE001
    Image = None

SIZE = 1024

PALETTES = {
    "ballad_piano":      ((9, 14, 28), (32, 48, 84), (236, 72, 153)),
    "ballad_acoustic":   ((18, 16, 14), (60, 48, 36), (245, 176, 106)),
    "ballad_orchestral": ((12, 10, 22), (44, 32, 66), (196, 148, 255)),
    "ballad_rnb":        ((14, 10, 20), (46, 26, 52), (236, 108, 168)),
    "ost_drama":         ((8, 14, 24), (24, 52, 74), (120, 214, 236)),
    "kpop_rock":         ((20, 8, 12), (70, 20, 30), (255, 108, 108)),
    "indie_folk":        ((16, 22, 16), (44, 66, 44), (198, 226, 140)),
    "citypop":           ((10, 12, 30), (34, 44, 96), (250, 140, 210)),
    "kpop_rnb":          ((10, 14, 22), (30, 44, 62), (140, 214, 255)),
    "kpop_dance":        ((16, 8, 26), (52, 22, 78), (168, 120, 255)),
    "custom":            ((12, 14, 22), (40, 46, 66), (150, 190, 255)),
}

MOOD_ACCENT = {
    "애절한": (232, 88, 120), "그리운": (148, 160, 210), "쓸쓸한": (120, 140, 170),
    "담담한": (170, 178, 190), "따뜻한": (245, 176, 106), "설레는": (255, 150, 190),
    "희망찬": (120, 214, 236), "웅장한": (196, 148, 255),
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothicBold.ttf", r"C:\Windows\Fonts\arialbd.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_font(bold: bool = False):
    if Image is None:
        return None
    cands = FONT_CANDIDATES if bold else [c for c in FONT_CANDIDATES if "bd" not in c.lower()]
    # 한글 지원 폰트 우선
    for c in cands:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, 10)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _font(size: int, bold: bool = True):
    f = find_font(bold)
    if f is None:
        return None
    try:
        return ImageFont.truetype(f.path, size)
    except Exception:
        try:
            return f.font_variant(size=size)
        except Exception:
            return f


def _wrap(draw, text, font, max_w):
    words = list(text)  # 한글은 글자 단위 줄바꿈이 자연스럽다
    lines, cur = [], ""
    for ch in words:
        test = (cur + ch).strip()
        if font is None:
            cur = test
            continue
        w = draw.textlength(test, font=font)
        if w > max_w and cur:
            lines.append(cur.strip())
            cur = ch
        else:
            cur = test
    if cur.strip():
        lines.append(cur.strip())
    return lines[:3]


def waveform_points(audio_path, n: int = 220):
    """곡의 실제 진폭 엔벨로프 (없으면 부드러운 사인 곡선)."""
    try:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError
        raw = subprocess.run([exe, "-v", "error", "-i", str(audio_path), "-f", "f32le",
                              "-acodec", "pcm_f32le", "-ac", "1", "-ar", "4000", "-"],
                             capture_output=True, check=True).stdout
        import numpy as np
        y = np.frombuffer(raw, dtype=np.float32)
        if len(y) < n:
            raise RuntimeError
        step = len(y) // n
        env = np.array([float(np.sqrt(np.mean(y[i * step:(i + 1) * step] ** 2) + 1e-9))
                        for i in range(n)])
        env /= (env.max() + 1e-9)
        return [float(v) for v in env]
    except Exception:
        rnd = random.Random(3)
        return [0.25 + 0.6 * abs(math.sin(i / 9.0)) * (0.7 + 0.3 * rnd.random()) for i in range(n)]


def make_cover(title: str, out_path, genre: str = "ballad_piano", moods=(),
               subtitle: str = "", audio_path=None, artist: str = "YUNSEUL-LMM") -> str:
    """커버 아트 PNG 생성 후 경로 반환."""
    if Image is None:
        raise RuntimeError("Pillow 가 필요합니다: pip install pillow")
    base, mid, accent = PALETTES.get(genre, PALETTES["custom"])
    for m in moods or []:
        if m in MOOD_ACCENT:
            accent = MOOD_ACCENT[m]
            break

    img = Image.new("RGB", (SIZE, SIZE), base)
    d = ImageDraw.Draw(img)
    # 세로 그라데이션
    for y in range(SIZE):
        t = y / (SIZE - 1)
        c = tuple(int(base[i] + (mid[i] - base[i]) * (t ** 0.9)) for i in range(3))
        d.line([(0, y), (SIZE, y)], fill=c)
    # 부드러운 빛 덩어리
    glow = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i, (cx, cy, r, k) in enumerate([(780, 220, 420, 0.9), (200, 820, 460, 0.55),
                                       (520, 480, 300, 0.35)]):
        gd.ellipse([cx - r, cy - r, cx + r, cy + r],
                   fill=tuple(int(c * k) for c in accent))
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    img = Image.blend(img, glow, 0.55)
    d = ImageDraw.Draw(img, "RGBA")

    # 얇은 동심원 (음반 질감)
    for i, r in enumerate(range(300, 760, 46)):
        d.ellipse([SIZE - 120 - r, SIZE - 60 - r, SIZE - 120 + r, SIZE - 60 + r],
                  outline=(255, 255, 255, 12), width=1)

    # 파형 라인
    pts = waveform_points(audio_path)
    n = len(pts)
    cy = 700
    amp = 110
    prev = None
    for i, v in enumerate(pts):
        x = 90 + (SIZE - 180) * i / max(1, n - 1)
        y = cy - v * amp
        if prev:
            d.line([prev, (x, y)], fill=accent + (150,), width=3)
        prev = (x, y)
        y2 = cy + v * amp * 0.55
        if i > 0:
            x0 = 90 + (SIZE - 180) * (i - 1) / max(1, n - 1)
            d.line([(x0, 700 + pts[i - 1] * amp * 0.55), (x, y2)],
                   fill=(255, 255, 255, 40), width=1)

    # 텍스트
    f_title = _font(96, True)
    f_sub = _font(34, False)
    f_small = _font(26, False)
    lines = _wrap(d, title or "제목 없음", f_title, SIZE - 200)
    ty = 300 - (len(lines) - 1) * 56
    for ln in lines:
        if f_title:
            d.text((100, ty), ln, font=f_title, fill=(248, 250, 252))
        ty += 112
    if subtitle:
        sub_lines = _wrap(d, subtitle, f_sub, SIZE - 220)[:2]
        sy = ty + 20
        for ln in sub_lines:
            if f_sub:
                d.text((102, sy), ln, font=f_sub, fill=(226, 232, 240))
            sy += 44
    # 상단 라벨
    if f_small:
        d.text((100, 100), artist.upper(), font=f_small, fill=(255, 255, 255, 170))
        d.line([(100, 148), (300, 148)], fill=accent + (220,), width=3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_path), "PNG")
    return str(out_path)


def make_intro_video(audio_path, cover_path, out_path, fade: float = 3.0,
                     title: str = "") -> str:
    """커버 이미지 + 곡 오디오로 인트로 영상(mp4) 생성. ffmpeg 필요."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg 를 찾지 못했습니다.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fade = max(0.5, float(fade))
    vf = (f"scale=1280:1280:force_original_aspect_ratio=decrease,"
          f"pad=1280:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"format=yuv420p,fade=t=in:st=0:d={fade:.2f}")
    cmd = [exe, "-y", "-v", "error", "-loop", "1", "-i", str(cover_path),
           "-i", str(audio_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError("ffmpeg 영상 생성 실패: " +
                           r.stderr.decode("utf-8", "replace")[-300:])
    return str(out_path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="노래 인트로/커버 이미지 생성")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--genre", default="ballad_piano")
    ap.add_argument("--moods", nargs="*", default=[])
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--artist", default="YUNSEUL-LMM")
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    p = make_cover(a.title, a.out, a.genre, a.moods, a.subtitle, a.audio, a.artist)
    res = {"cover": p}
    if a.video and a.audio:
        res["video"] = make_intro_video(a.audio, p, a.video, title=a.title)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
