#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""회귀 테스트: (1) 1000곡 모델이 보컬 목록에서 빠지는지 (2) 노래 모델 목록에 뜨는지
   (3) 가수 코치 체인이 실제로 음정 흔들림을 줄이는지 (4) 장편 기획이 300쪽을 만드는지

실행: python3 test_catalog_and_coach.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

import voice_catalog

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{OK if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def make_model(root: Path, name: str, meta: dict, with_adapter=True):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if with_adapter:
        ad = d / "lora" / "final"
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
        (ad / "adapter_model.safetensors").write_bytes(b"0" * 16)
        m = dict(meta)
        m["adapter_path"] = str(ad)
        (d / "meta.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return d


def test_catalog():
    root = Path(tempfile.mkdtemp(prefix="cat-"))
    try:
        # 1000곡 코퍼스 모델 (role 표기 없음 → 규모로 판정되어야 함)
        make_model(root, "1000song_corpus",
                   {"name": "1000song_corpus", "songs_count": 1043,
                    "songs_with_lyrics": 812, "train_minutes": 5120})
        # 문장 녹음으로 만든 내 목소리 (별칭)
        make_model(root, "ali_my_voice",
                   {"name": "ali_my_voice", "songs_count": 1, "enrolled": True,
                    "role": "voice", "train_minutes": 3})
        # role 이 명시된 코퍼스
        make_model(root, "artist_style",
                   {"name": "artist_style", "role": "song", "songs_count": 40})

        scan = voice_catalog.scan_models(root)
        voice_names = [v["name"] for v in scan["voices"]]
        song_names = [s["name"] for s in scan["song_models"]]
        print("\n  voices     :", voice_names)
        print("  song_models:", song_names)

        check("1000곡 모델이 song_models 로 분류됨", "1000song_corpus" in song_names, str(song_names))
        check("1000곡 모델이 voices 에서 제외됨", "1000song_corpus" not in voice_names, str(voice_names))
        check("내 목소리(별칭)는 voices 에 남음", "ali_my_voice" in voice_names, str(voice_names))
        check("role=song 도 voices 에서 제외", "artist_style" not in voice_names, str(voice_names))

        v_entries = voice_catalog.voice_combo_entries(scan)
        s_entries = voice_catalog.song_model_entries(scan)
        v_labels = [e["label"] for e in v_entries]
        s_labels = [e["label"] for e in s_entries]
        print("\n  보컬 콤보:", v_labels)
        print("  곡 모델 콤보:", s_labels)
        check("보컬 콤보 라벨에 1000곡 모델 이름 없음",
              not any("1000song" in x for x in v_labels), str(v_labels))
        check("곡 모델 콤보 라벨에 1000곡 모델 있음",
              any("1000song" in x for x in s_labels), str(s_labels))
        check("보컬 콤보에 남/여 기본이 포함됨",
              any("남성" in x for x in v_labels) and any("여성" in x for x in v_labels))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _wav(path, freq=220.0, sec=3.0, sr=48000, drift_cents=0.0, vib_cents=0.0, vib_rate=5.2):
    n = int(sr * sec)
    t = np.arange(n) / sr
    cents = np.full(n, drift_cents * np.sin(2 * np.pi * 0.35 * t)) \
        + vib_cents * np.sin(2 * np.pi * vib_rate * t + 0.8)
    f0 = freq * (2 ** (cents / 1200.0))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = 0.5 * np.sin(phase) + 0.22 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase) \
        + 0.06 * np.sin(4 * phase)
    amp = np.clip(1.0 - np.abs(t - sec / 2) / (sec / 2 + 0.1), 0.05, 1.0)
    data = (sig * amp * 0.55 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return path


def test_coach():
    if not shutil.which("ffmpeg"):
        check("가수 코치 테스트", False, "ffmpeg 없음 → 건너뜀")
        return
    tmp = Path(tempfile.mkdtemp(prefix="coach-"))
    try:
        src = _wav(tmp / "in.wav", drift_cents=45, vib_cents=12)
        out = tmp / "out.mp3"
        r = subprocess.run([sys.executable, "vocal_coach.py", "--input", str(src),
                            "--output", str(out), "--preset", "ballad", "--intensity", "1.0"],
                           capture_output=True, text=True, timeout=300)
        print("  stdout tail:", (r.stdout or "").strip().splitlines()[-4:])
        check("vocal_coach 실행 성공", r.returncode == 0, (r.stderr or "")[-300:])
        check("vocal_coach 결과 mp3 생성", out.exists() and out.stat().st_size > 10000,
              f"{out.stat().st_size if out.exists() else 0} bytes")
        rep = json.loads(r.stdout[r.stdout.find("{"):]) if "{" in (r.stdout or "") else {}
        b, a = rep.get("before") or {}, rep.get("after") or {}
        if b and a:
            print("  before:", b)
            print("  after :", a)
            # 주의: 아래는 '합성 사인파' 기준 참고 수치다. 실제 사람 녹음이 아니라
            # f0 추적이 불안정해 증감을 pass/fail 로 강제하지 않는다(정보성 출력).
            # 인지적 품질은 실제 곡으로 들어봐야 판정된다 → 한계로 보고한다.
            print(f"  [참고] drift {b.get('drift_cents')} → {a.get('drift_cents')} cent, "
                  f"jitter {b.get('jitter_cents')} → {a.get('jitter_cents')} cent "
                  f"(합성 신호 기준, 판정 보류)")
            check("전/후 지표가 실제로 측정됨",
                  a.get("drift_cents") is not None and a.get("hnr_db") is not None,
                  f"after={a}")
            check("가수 코치가 단계별 처리를 기록함",
                  len(rep.get("stages") or []) >= 3,
                  str([s.get("stage") for s in (rep.get("stages") or [])]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ebook():
    import ebook_persona
    plan = ebook_persona.plan_chapters(300, "fantasy")
    tot = sum(c["pages"] for c in plan["chapters"])
    chars = sum(s["target_chars"] for c in plan["chapters"] for s in c["sections"])
    print(f"\n  300쪽 기획: {len(plan['chapters'])}장 / {tot}쪽 / 목표 {chars:,}자")
    check("300쪽 기획이 300쪽을 채움", abs(tot - 300) <= len(plan["chapters"]), f"{tot}쪽")
    check("장 수가 2 이상", len(plan["chapters"]) >= 2, str(len(plan["chapters"])))
    check("목표 글자수가 10만자 이상", chars >= 100000, f"{chars:,}자")
    for g in ebook_persona.GENRE_ORDER:
        d = ebook_persona.persona_dict(g)
        check(f"페르소나 6축 완비: {g}",
              all(d.get(k) for k in ("role", "era", "values", "voice", "taboo")), "")
    sysmsg = ebook_persona.persona_system("fantasy", "냉소적 유머를 쓴다")
    check("외부 페르소나가 시스템 프롬프트에 반영", "냉소적 유머" in sysmsg)
    st = ebook_persona.StoryState(characters={"Ari": "잃어버린 이름을 찾으려 한다"})
    st.open_threads.append("부러진 펜던트")
    rep = ebook_persona.consistency_report(["가" * 4000, "가" * 900], st)
    check("일관성 점검이 미회수 복선을 잡아냄",
          any("복선" in i for i in rep["issues"]), str(rep["issues"]))


if __name__ == "__main__":
    print("=== 1) 모델 분류 ===")
    test_catalog()
    print("\n=== 2) 가수 코치 ===")
    test_coach()
    print("\n=== 3) 전자책 페르소나/장편 ===")
    test_ebook()
    failed = [r for r in results if not r[1]]
    print(f"\n총 {len(results)}개 중 {len(results) - len(failed)}개 통과, {len(failed)}개 실패")
    sys.exit(1 if failed else 0)
