#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
song_finalize.py - 만들어진 노래 한 곡을 '완성품'으로 마무리하는 단일 진입점

순서 (각 단계는 실패해도 다음 단계로 넘어가고, 사유를 리포트에 남긴다)
  1) svc      : 내 목소리 프로필/ RVC 로 음색 변환        (--profile)
  2) polish   : 거친 대역·레가토·피치 흔들림·리버브·마스터 (--polish soft|natural|rich|off)
  3) cover    : 제목/장르/분위기로 인트로 이미지 생성       (--cover auto|<path>)
  4) embed    : mp3 에 USLT(가사) + SYLT(동기화) + APIC(커버) + 기본 태그
  5) video    : 커버 + 곡 → 인트로 mp4                    (--video)

결과는 <출력>.finalize.json 에 저장되고 stdout 으로도 JSON 을 출력한다.

  python song_finalize.py --input song.mp3 --lyrics-file song.lyrics.txt \
      --title "봄의 끝" --genre-name "피아노 발라드" --genre ballad_piano \
      --moods 그리운 따뜻한 --profile trained_vocal_models/내목소리/voice_profile.json \
      --polish natural --cover auto --video
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def log(msg):
    print(f"[LOG] {msg}", flush=True)


def progress(pct, title):
    print(f"[PROGRESS:1:{int(pct)}:{title}]", flush=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="노래 마무리(음색·다듬기·가사·커버·영상)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="기본: 입력 파일을 그대로 덮어씀")
    ap.add_argument("--lyrics-file", default=None)
    ap.add_argument("--lyrics", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="YUNSEUL-LMM")
    ap.add_argument("--album", default="YUNSEUL-LMM Studio")
    ap.add_argument("--genre", default="ballad_piano")
    ap.add_argument("--genre-name", default="")
    ap.add_argument("--moods", nargs="*", default=[])
    ap.add_argument("--profile", default=None, help="voice_profile.json 경로")
    ap.add_argument("--svc-backend", default="auto", choices=["auto", "rvc", "dsp", "off"])
    ap.add_argument("--polish", default="natural", choices=["off", "soft", "natural", "rich"])
    ap.add_argument("--singer-style", default="ballad",
                    choices=["ballad", "power", "idol", "natural"],
                    help="가수처럼 부르기 프리셋 (vocal_coach)")
    ap.add_argument("--singer-intensity", type=float, default=1.0,
                    help="가수 보정 강도 0(끔)~1.3")
    ap.add_argument("--double-track", default=None, help="겹칠 두 번째 테이크 파일")
    ap.add_argument("--cover", default="auto", help="auto | none | 이미지 경로")
    ap.add_argument("--no-lyrics-tag", action="store_true", help="mp3 가사/동기화 태그를 넣지 않음")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"[ERROR] 입력 파일이 없습니다: {src}", flush=True)
        return 1
    dst = Path(args.output) if args.output else src
    lyrics = args.lyrics or ""
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")

    report = {"input": str(src), "output": str(dst), "steps": {}, "warnings": []}
    tmpdir = Path(tempfile.mkdtemp(prefix="finalize_"))
    stage_in = src

    # ---------------------------------------------------------------- 1) SV C
    if args.profile and args.svc_backend != "off" and Path(args.profile).exists():
        progress(10, "내 목소리 음색으로 변환 중...")
        try:
            import svc  # 같은 폴더
            out1 = tmpdir / "svc.wav"
            r = svc.convert_vocal(str(stage_in), args.profile, str(out1), args.svc_backend)
            stage_in = out1
            report["steps"]["svc"] = r
            log(f"• 음색 변환({r.get('backend')}) 완료" +
                (f" — {len(r.get('notes', []))}개 보정" if r.get("notes") else ""))
            if r.get("disclaimer"):
                report["warnings"].append(r["disclaimer"])
        except Exception as e:  # noqa: BLE001
            report["steps"]["svc"] = {"error": f"{type(e).__name__}: {e}"}
            report["warnings"].append(f"음색 변환 실패 → 원본 음색 유지: {e}")
            log(f"⚠️ 음색 변환 실패: {e}")
    else:
        report["steps"]["svc"] = {"skipped": True}

    # ---------------------------------------------------------------- 2) 다듬기
    # 2a) 가수 코치: 비브라토·프레이즈 다이내믹·싱어 포먼트 (근거: VibE-SVC 2025,
    #     ICASSP 2022 비브라토 모델링, Roebel ICASSP 2012 여기 성분 변조)
    if args.singer_intensity and args.singer_intensity > 0:
        progress(30, "가수처럼 다듬는 중 (비브라토·다이내믹·싱어포먼트)...")
        try:
            import vocal_coach
            out_star = tmpdir / "star.mp3"
            r = vocal_coach.polish_star(str(stage_in), str(out_star),
                                        preset=args.singer_style,
                                        intensity=args.singer_intensity)
            stage_in = out_star
            report["steps"]["singer_coach"] = {
                "preset": args.singer_style, "intensity": args.singer_intensity,
                "before": r.get("before"), "after": r.get("after"),
                "improved": r.get("improved"), "stages": r.get("stages"),
            }
            log(f"• 가수 코치({args.singer_style} x{args.singer_intensity}) 완료"
                + (f" · 음정흔들림 {r['before'].get('jitter_cents')} → {r['after'].get('jitter_cents')} cent"
                   if r.get("before") and r.get("after") else ""))
        except Exception as e:  # noqa: BLE001
            report["steps"]["singer_coach"] = {"error": f"{type(e).__name__}: {e}"}
            report["warnings"].append(f"가수 코치 실패 → 기본 다듬기로 진행: {e}")
            log(f"⚠️ 가수 코치 실패: {e}")
    else:
        report["steps"]["singer_coach"] = {"skipped": True}

    if args.polish != "off":
        progress(35, "부드럽게 다듬는 중 (레가토·피치·리버브·마스터)...")
        try:
            import vocal_post
            out2 = tmpdir / "polished.mp3"
            # 가수 코치를 이미 적용했다면 거친 대역/레가토/리버브/마스터만 다시 한다
            r = vocal_post.polish_song(str(stage_in), str(out2), preset=args.polish,
                                       double_track_path=args.double_track,
                                       pitch=not (args.singer_intensity and args.singer_intensity > 0))
            stage_in = out2
            report["steps"]["polish"] = r
            log(f"• 다듬기({args.polish}) 완료 · 단계 {', '.join(r.get('stages', []))}")
            if r.get("pitch", {}).get("applied"):
                log(f"  - 피치 흔들림 {r.get('pitch_jitter_cents_in')} → "
                    f"{r.get('pitch_jitter_cents_out')} cent")
        except Exception as e:  # noqa: BLE001
            report["steps"]["polish"] = {"error": f"{type(e).__name__}: {e}"}
            report["warnings"].append(f"다듬기 실패 → 원본 유지: {e}")
            log(f"⚠️ 다듬기 실패: {e}")
    else:
        report["steps"]["polish"] = {"skipped": True}

    # 최종 오디오 확정 (mp3 로 통일)
    final_audio = tmpdir / "final.mp3"
    try:
        ff = shutil.which("ffmpeg")
        if ff:
            subprocess.run([ff, "-y", "-v", "error", "-i", str(stage_in), "-c:a", "libmp3lame",
                            "-b:a", "320k", str(final_audio)], check=True)
        else:
            shutil.copy2(stage_in, final_audio)
    except Exception as e:  # noqa: BLE001
        report["warnings"].append(f"최종 인코딩 실패: {e}")
        shutil.copy2(stage_in, final_audio)

    # ---------------------------------------------------------------- 3) 커버
    cover_path = None
    if args.cover and args.cover != "none":
        progress(60, "인트로 이미지 만드는 중...")
        try:
            import intro_image
            if args.cover == "auto":
                cover_path = str(dst.with_suffix(".cover.png"))
                sub = " · ".join(x for x in [args.genre_name] + list(args.moods or []) if x)
                intro_image.make_cover(args.title or dst.stem, cover_path, args.genre,
                                       args.moods, sub, str(final_audio), args.artist)
            else:
                cover_path = args.cover
            report["steps"]["cover"] = {"path": cover_path}
            log(f"• 인트로 이미지: {cover_path}")
        except Exception as e:  # noqa: BLE001
            report["steps"]["cover"] = {"error": f"{type(e).__name__}: {e}"}
            report["warnings"].append(f"인트로 이미지 실패: {e}")
            log(f"⚠️ 인트로 이미지 실패: {e}")
    else:
        report["steps"]["cover"] = {"skipped": True}

    # ---------------------------------------------------------------- 4) 태그
    progress(80, "mp3 에 가사와 커버를 넣는 중...")
    try:
        shutil.copy2(final_audio, dst)
        if args.no_lyrics_tag:
            report["steps"]["embed"] = {"skipped": True, "reason": "--no-lyrics-tag"}
            if cover_path:
                report["steps"]["cover"]["embedded"] = False
        else:
            if dst.suffix.lower() != ".mp3":
                report["warnings"].append("가사 태그는 mp3 에만 기록됩니다.")
            import mp3_lyrics_embed
            r = mp3_lyrics_embed.embed_song(
                dst, lyrics, args.title, args.artist, args.album, args.genre_name or args.genre,
                cover_path, None, None)
            report["steps"]["embed"] = r
            if r.get("warnings"):
                report["warnings"].extend(r["warnings"])
            log(f"• 태그: 가사 {r.get('written', {}).get('USLT', False)} / "
                f"동기화 {r.get('written', {}).get('SYLT', 0)}줄 / "
                f"커버 {r.get('written', {}).get('APIC', '-')}")
    except Exception as e:  # noqa: BLE001
        report["steps"]["embed"] = {"error": f"{type(e).__name__}: {e}"}
        report["warnings"].append(f"가사/커버 삽입 실패: {e}")
        log(f"⚠️ 태그 삽입 실패: {e}")

    # 가사 텍스트도 옆에 남긴다 (앱에서 다시 불러오기용)
    if lyrics.strip():
        try:
            dst.with_suffix(".lyrics.txt").write_text(lyrics, encoding="utf-8")
        except Exception:
            pass

    # ---------------------------------------------------------------- 5) 영상
    if args.video and cover_path:
        progress(90, "인트로 영상 만드는 중...")
        try:
            import intro_image
            vp = str(dst.with_suffix(".intro.mp4"))
            intro_image.make_intro_video(str(dst), cover_path, vp, 3.0, args.title)
            report["steps"]["video"] = {"path": vp}
            log(f"• 인트로 영상: {vp}")
        except Exception as e:  # noqa: BLE001
            report["steps"]["video"] = {"error": f"{type(e).__name__}: {e}"}
            report["warnings"].append(f"인트로 영상 실패: {e}")
    else:
        report["steps"]["video"] = {"skipped": True}

    if not args.keep_temp:
        shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        dst.with_suffix(".finalize.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    progress(100, "완료")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[DONE:{dst}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
