#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_enroll.py - '내 목소리' 등록/학습 (10문장 ~ 30문장 따라 읽기)

흐름
 1) 대본(voice_scripts.py)을 한 문장씩 읽어 녹음  → <root>/<name>/speech/*.wav
 2) 각 클립을 분석해 'voice_profile.json' 생성
    - f0 중앙값/범위, 포먼트 F1~F3, HNR, 스펙트럼 중심, 성별 추정
    - 이 프로필은 노래 생성 시 (a) 캡션 힌트 (b) 음역 정렬 (c) 후처리 EQ 목표로 쓰인다
 3) 클립을 하나의 '연속 발화 트랙'(약 2~8분)으로 합치고 문장별 가사 파일을 만든다
 4) ACE-Step LoRA 학습 파이프라인(vocal_trainer_runner.py)을 그대로 호출해
    내 음색/발음을 DiT 에 학습시킨다 → 노래 생성 시 자동 적용
 5) (선택) RVC 가 설치되어 있으면 RVC 학습 명령도 실행한다 (없으면 건너뛰고 사유 기록)

이 파일은 단독 실행도 가능하다(위젯은 subprocess 로 호출):
  python voice_enroll.py enroll --name 내목소리 --clips a.wav b.wav --out-root trained_vocal_models
  python voice_enroll.py script --limit 30 --out voice_script.txt
  python voice_enroll.py analyze --clips a.wav b.wav
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
F0_MIN, F0_MAX = 60.0, 600.0
SR = 48000
GAP_SEC = 0.45

try:  # 분석에는 numpy 가 필요하다
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


# ---------------------------------------------------------------- 공통
def log(msg: str) -> None:
    print(f"[LOG] {msg}", flush=True)


def progress(stage: int, pct: int, title: str) -> None:
    print(f"[PROGRESS:{stage}:{int(pct)}:{title}]", flush=True)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg 를 찾지 못했습니다. PATH 에 추가해 주세요.")
    return exe


def wav_duration(path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=nk=1:nw=1", str(path)],
                                 capture_output=True, text=True, check=True).stdout.strip()
            return float(out)
        except Exception:
            return 0.0


def decode_mono(path, sr: int = 24000):
    if np is None:
        raise RuntimeError("numpy 가 필요합니다: pip install numpy")
    raw = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(y, path, sr: int = SR) -> None:
    """stdlib wave 로 int16 WAV 저장 (soundfile 의존 없음)."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    pcm = np.clip(y * 32767.0, -32768, 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def trim_silence(y, sr: int = 24000, thresh: float = 0.012, pad: float = 0.12):
    if not len(y):
        return y
    a = np.abs(y)
    edges = np.where(a > thresh)[0]
    if not len(edges):
        return y
    start = max(0, int(edges[0] - pad * sr))
    end = min(len(y), int(edges[-1] + pad * sr))
    return y[start:end]


# ---------------------------------------------------------------- 분석
def _f0_track(y, sr: int):
    """자기상관 기반 f0 (프레임 50ms, 홉 10ms)."""
    frame = int(sr * 0.05)
    hop = int(sr * 0.01)
    if len(y) < frame * 3:
        return np.zeros(0)
    fr = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop].astype(np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr *= np.hanning(frame).astype(np.float32)
    spec = np.fft.rfft(fr, n=2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame]
    ac /= (ac[:, :1] + 1e-9)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    seg = ac[:, lo:hi]
    idx = np.argmax(seg, axis=1)
    conf = seg[np.arange(len(idx)), idx]
    lag = (idx + lo).astype(np.float64)
    f0 = np.where(conf > 0.28, sr / np.maximum(lag, 1e-6), 0.0)
    f0[(f0 < F0_MIN) | (f0 > F0_MAX)] = 0.0
    # 안정 구간만 (앞뒤 중앙값 근처)
    voiced = f0[f0 > 0]
    if len(voiced) < 5:
        return np.zeros(0)
    med = np.median(voiced)
    keep = (f0 > med * 0.55) & (f0 < med * 1.9)
    return f0[keep]


def _lpc_formants(y, sr: int = 24000, order: int = 12):
    """LPC 로 F1/F2/F3 근사 (모음 구간 평균)."""
    try:
        y = y - y.mean()
        # 안정 구간 1초
        frame = int(sr * 0.04)
        hop = int(sr * 0.02)
        if len(y) < frame * 4:
            return []
        fr = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop].astype(np.float64)
        energy = (fr ** 2).sum(axis=1)
        pick = fr[np.argsort(energy)[-8:]]
        out = []
        for f in pick:
            f = f * np.hamming(frame)
            r = np.correlate(f, f, mode="full")[frame - 1:]
            if r[0] <= 0:
                continue
            r = r / r[0]
            a = np.zeros(order + 1)
            a[0] = 1.0
            err = r[0]
            for i in range(1, order + 1):
                acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1]) if i > 1 else r[i]
                k = -acc / (err + 1e-12)
                a_new = a.copy()
                for j in range(1, i):
                    a_new[j] = a[j] + k * a[i - j]
                a_new[i] = k
                a = a_new
                err *= (1 - k * k)
                if err <= 1e-12:
                    break
            roots = np.roots(a)
            roots = roots[np.imag(roots) > 0.01]
            freqs = np.sort(np.angle(roots) * sr / (2 * np.pi))
            freqs = freqs[(freqs > 180) & (freqs < 4200)]
            if len(freqs) >= 3:
                out.append([float(x) for x in freqs[:3]])
        if not out:
            return []
        arr = np.array(out)
        return [round(float(np.median(arr[:, i])), 1) for i in range(arr.shape[1])]
    except Exception:
        return []


def analyze_clip(path):
    """한 클립의 음성 특징."""
    if np is None:
        raise RuntimeError("numpy 가 필요합니다: pip install numpy")
    sr = 24000
    y = trim_silence(decode_mono(path, sr), sr)
    dur = len(y) / sr
    info = {"file": str(path), "duration": round(dur, 2)}
    if dur < 0.4:
        info["skipped"] = "0.4초 미만"
        return info
    f0 = _f0_track(y, sr)
    if len(f0):
        info["f0_median"] = round(float(np.median(f0)), 1)
        info["f0_p10"] = round(float(np.percentile(f0, 10)), 1)
        info["f0_p90"] = round(float(np.percentile(f0, 90)), 1)
        z = 1200 * np.log2(f0 / np.median(f0))
        info["f0_std_cents"] = round(float(np.std(z)), 1)
    # 스펙트럼 중심 / 고역 비율
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    info["centroid_hz"] = round(float((spec * freqs).sum() / spec.sum()), 1)
    info["hf_ratio"] = round(float(spec[freqs > 4000].sum() / spec.sum()), 4)
    # HNR 근사
    r = np.correlate(y, y, mode="full")[len(y) - 1:]
    r = r / (r[0] + 1e-12)
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    peak = float(np.max(r[lo:hi])) if len(r) > hi else 0.0
    info["hnr_db"] = round(float(10 * math.log10(max(peak, 1e-6) / max(1 - peak, 1e-6))), 1)
    info["formants"] = _lpc_formants(y, sr)
    info["rms_db"] = round(float(20 * math.log10(max(float(np.sqrt(np.mean(y ** 2))), 1e-9))), 1)
    return info


def build_profile(name: str, clips, out_dir: Path):
    per = [analyze_clip(c) for c in clips]
    ok = [p for p in per if p.get("f0_median")]
    if not ok:
        raise RuntimeError("목소리를 분석할 수 있는 클립이 없습니다(무음이거나 너무 짧음).")
    med = float(np.median([p["f0_median"] for p in ok]))
    p10 = float(np.median([p.get("f0_p10", p["f0_median"]) for p in ok]))
    p90 = float(np.median([p.get("f0_p90", p["f0_median"]) for p in ok]))
    cents = float(np.median([p.get("f0_std_cents", 0.0) for p in ok]))
    forms = [p["formants"] for p in ok if p.get("formants")]
    formant = [round(float(np.median([f[i] for f in forms])), 1) for i in range(3)] if forms else []
    total = sum(p.get("duration", 0) for p in per)
    centroid = float(np.median([p.get("centroid_hz", 0) for p in ok]))
    hnr = float(np.median([p.get("hnr_db", 0) for p in ok]))
    gender = "male" if med < 155 else "female"

    bright = "bright" if centroid > 2100 else "warm" if centroid > 1500 else "dark"
    breath = "breathy" if hnr < 8 else "clear" if hnr > 16 else "natural"
    hint = (f"{gender} {bright} {breath} mid-range lead vocal, natural Korean diction, "
            f"close-mic intimate tone")
    profile = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": len([p for p in per if not p.get("skipped")]),
        "total_speech_sec": round(total, 1),
        "f0_median_hz": round(med, 1),
        "f0_p10_hz": round(p10, 1),
        "f0_p90_hz": round(p90, 1),
        "f0_spread_cents": round(cents, 1),
        "hnr_db": round(hnr, 1),
        "centroid_hz": round(centroid, 1),
        "formants_hz": formant,
        "gender_guess": gender,
        "brightness": bright,
        "breathiness": breath,
        "caption_hint": hint,
        "lora_scale_hint": 1.0 if total > 60 else 0.85,
        "sing_range_suggest_hz": [round(p10 * 0.9, 1), round(p90 * 1.35, 1)],
        "clips": per,
        "quality_notes": _quality_notes(len(ok), total, cents, formant, hnr),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "voice_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def _quality_notes(n, total, cents, formant, hnr):
    notes = []
    if n < 10:
        notes.append(f"클립이 {n}개입니다. 최소 10문장을 권장합니다.")
    elif n < 20:
        notes.append(f"{n}문장으로 학습합니다. 30문장이면 억양이 더 자연스러워집니다.")
    if total < 45:
        notes.append(f"총 발화 {total:.0f}초 — 60초 이상이면 노래 표현력이 좋아집니다.")
    if cents > 70:
        notes.append(f"음높이 흔들림이 큽니다({cents:.0f} cent). 같은 문장을 2번 녹음해 보세요.")
    if not formant:
        notes.append("포먼트 분석 실패 — 조용한 곳에서 다시 녹음하면 개선됩니다.")
    if hnr < 6:
        notes.append("잡음 대비 신호가 낮습니다(HNR). 마이크를 조금 더 가까이.")
    if not notes:
        notes.append("녹음 품질 양호 — 바로 학습해도 됩니다.")
    return notes


# ---------------------------------------------------------------- 데이터셋 합치기
def build_montage(clips, script_map, out_dir: Path, name: str):
    """클립을 0.45초 간격으로 이어 붙이고, 문장별 가사 파일을 만든다."""
    parts, lines, used = [], [], 0
    for c in clips:
        try:
            y = trim_silence(decode_mono(c, SR), SR)
        except Exception as e:  # noqa: BLE001
            log(f"  ⚠️ 클립 건너뜀 {Path(c).name}: {e}")
            continue
        if len(y) < SR * 0.4:
            continue
        parts.append(y)
        parts.append(np.zeros(int(SR * GAP_SEC), dtype=np.float32))
        text = (script_map or {}).get(str(c)) or (script_map or {}).get(Path(c).name) or ""
        if text.strip():
            lines.append(text.strip())
        used += 1
    if not parts:
        raise RuntimeError("합칠 수 있는 클립이 없습니다.")
    y = np.concatenate(parts)
    y = y / (float(np.max(np.abs(y))) + 1e-9) * 0.89
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{name}_speech.wav"
    write_wav(y, wav_path, SR)

    blocks = [lines[i:i + 3] for i in range(0, len(lines), 3)] if lines else []
    body = []
    for i, b in enumerate(blocks):
        tag = "[Chorus]" if i % 3 == 2 else f"[Verse {i + 1}]"
        body.append(tag + "\n" + "\n".join(b))
    text = "\n\n".join(body) if body else "[Instrumental]"
    (out_dir / f"{name}_speech.lyrics.txt").write_text(text, encoding="utf-8")
    return wav_path, used, len(y) / SR


# ---------------------------------------------------------------- RVC (선택)
def rvc_root():
    for c in (BASE_DIR / "audio_engines" / "RVC", BASE_DIR / "audio_engines" / "RVC-WebUI",
              BASE_DIR / "RVC"):
        if c.exists():
            return c
    return None


def rvc_train_command(name: str, exp_dir: Path, dataset_dir: Path, sr: str = "48k",
                      epochs: int = 200) -> list | None:
    """RVC WebUI 학습 CLI. 설치돼 있지 않으면 None."""
    root = rvc_root()
    if not root:
        return None
    train = root / "infer" / "modules" / "train" / "train.py"
    if not train.exists():
        return None
    py = root / "venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = root / "runtime" / "python.exe"
    exe = str(py) if Path(py).exists() else sys.executable
    return [exe, str(train), "-e", name, "-sr", sr, "-f0", "1", "-bs", "8", "-g", "0",
            "-te", str(epochs), "-se", str(max(20, epochs // 4)), "-pg", "0.7", "-pd", "0.3",
            "-l", "0", "-c", "0", "-sw", "1", "-m", "40", "-v", "v2"]


# ---------------------------------------------------------------- 등록 전체 실행
def enroll(name: str, clips, out_root: Path, script_map=None, train_lora=True,
           preset="standard", acestep_python=None) -> dict:
    name = re.sub(r"[^\w\s\-가-힣]", "", name).strip() or "MyVoice"
    model_dir = Path(out_root) / name
    speech_dir = model_dir / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)

    report = {"name": name, "model_dir": str(model_dir), "steps": [], "warnings": []}
    progress(1, 5, f"[1/5] 클립 {len(clips)}개 확인 중...")
    kept = []
    for c in clips:
        p = Path(c)
        if not p.exists():
            continue
        dst = speech_dir / p.name
        if p.resolve() != dst.resolve():
            try:
                shutil.copy2(p, dst)
            except Exception:
                dst = p
        kept.append(str(dst))
    if not kept:
        raise RuntimeError("사용할 녹음 파일이 없습니다.")
    total_sec = sum(wav_duration(c) for c in kept)
    log(f"• 녹음 {len(kept)}개 / 총 {total_sec:.1f}초")
    report["steps"].append({"step": "수집", "clips": len(kept), "seconds": round(total_sec, 1)})

    progress(2, 18, "[2/5] 목소리 분석 중 (음높이·포먼트·잡음)...")
    profile = build_profile(name, kept, model_dir)
    log(f"• 음높이 중앙값 {profile['f0_median_hz']}Hz ({profile['gender_guess']}) "
        f"· 포먼트 {profile['formants_hz']} · HNR {profile['hnr_db']}dB")
    for note in profile["quality_notes"]:
        log(f"  - {note}")
    report["steps"].append({"step": "분석", "profile": {k: profile[k] for k in
                            ("f0_median_hz", "f0_p10_hz", "f0_p90_hz", "hnr_db", "formants_hz",
                             "gender_guess", "caption_hint")}})

    progress(3, 28, "[3/5] 학습한 목소리를 노래용으로 이상화하는 중...")
    try:
        import voice_idealize
        ideal = voice_idealize.idealize_voice(kept, model_dir, keep_ratio=0.85)
        if ideal.get("kept_files"):
            kept = ideal["kept_files"]        # HNR 상위 클립만 학습에 사용
        report["steps"].append({"step": "이상화", "kept": ideal.get("clips_kept"),
                               "seconds": ideal.get("total_sec"),
                               "mean_hnr_db": ideal.get("mean_hnr_db"),
                               "notes": ideal.get("notes")})
        for n in ideal.get("notes", []):
            log(f"  - {n}")
        log(f"• 이상화: {ideal.get('clips_kept')}개 유지 / 총 {ideal.get('total_sec')}초 / "
            f"HNR {ideal.get('mean_hnr_db')}dB")
    except Exception as e:  # noqa: BLE001
        log(f"  ⚠️ 이상화 건너뜀: {e}")
        report["warnings"].append(f"이상화 실패 → 원본 클립 사용: {e}")

    progress(3, 34, "[3/5] 학습용 연속 발화 트랙 생성 중...")
    montage, used, msec = build_montage(kept, script_map, model_dir / "dataset", name)
    log(f"• 학습 트랙: {montage.name} ({msec:.1f}초, 클립 {used}개)")
    report["steps"].append({"step": "데이터셋", "montage": str(montage), "seconds": round(msec, 1)})

    triggered_rvc = False
    cmd = rvc_train_command(name, model_dir, model_dir / "dataset")
    if cmd:
        progress(4, 45, "[4/5] RVC 보컬 모델 학습 시도...")
        log("• RVC 감지 → RVC 학습을 실행합니다 (실패해도 ACE-Step LoRA 는 계속 진행).")
        try:
            r = subprocess.run(cmd, cwd=str(rvc_root()), capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            pth = list((rvc_root() / "assets" / "weights").rglob(f"{name}*.pth"))
            if r.returncode == 0 and pth:
                triggered_rvc = True
                log(f"  ✓ RVC 모델: {pth[0]}")
                report["steps"].append({"step": "RVC", "model": str(pth[0])})
            else:
                msg = (r.stderr or r.stdout or "")[-300:]
                log(f"  ⚠️ RVC 학습 건너뜀 (exit {r.returncode}): {msg.strip()[:200]}")
                report["warnings"].append("RVC 학습 실패 → ACE-Step LoRA 로 진행")
        except Exception as e:  # noqa: BLE001
            log(f"  ⚠️ RVC 실행 실패: {e}")
            report["warnings"].append(f"RVC 실행 실패: {e}")
    else:
        log("• RVC 미설치 → ACE-Step LoRA + 프로필 정렬 방식으로 학습합니다.")

    if not train_lora:
        log("• --no-train: 프로필까지만 생성했습니다.")
        report["trained_lora"] = False
        report["profile"] = profile
        print(f"[DONE:{name}]", flush=True)
        return report

    progress(4, 55, "[4/5] ACE-Step LoRA 학습 (내 음색을 노래 엔진에 심는 중)...")
    runner = BASE_DIR / "vocal_trainer_runner.py"
    py = acestep_python or default_runner_python()
    cmd = [str(py), "-u", str(runner), "--model-name", name, "--output-dir", str(out_root),
           "--audio-files", str(montage), "--genre-filter", "all", "--preset", preset]
    log(f"• 실행: {' '.join(cmd[-6:])}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env, cwd=str(BASE_DIR))
    adapter, err = None, None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("[PROGRESS:"):
            print(line, flush=True)
            continue
        if line.startswith("[LOG] "):
            print(line, flush=True)
            continue
        if line.startswith("[DONE:"):
            adapter = "from-runner"
            continue
        if line.startswith("[ERROR]") or "Traceback" in line:
            err = line[:300]
            print(line, flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ACE-Step LoRA 학습 실패: {err or 'exit ' + str(proc.returncode)}")

    # meta.json 에 프로필 정보를 합쳐 둔다 (노래 생성 시 캡션/강도 자동 적용)
    meta_path = model_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta["role"] = "voice"          # voice_catalog 가 '내 목소리'로 분류하는 근거
    meta["model_role"] = "voice"
    meta["source"] = "sentence-enrollment"
    meta["voice_profile"] = profile
    meta["enrolled"] = True
    meta["enrolled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["lora_scale_hint"] = profile["lora_scale_hint"]
    meta["caption_hint"] = profile["caption_hint"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    progress(5, 100, "[5/5] 등록 완료")
    log(f"✅ '{name}' 목소리 등록 완료 — 노래 만들기에서 선택할 수 있습니다.")
    report["trained_lora"] = True
    report["profile"] = profile
    report["rvc"] = triggered_rvc
    print(f"[DONE:{name}]", flush=True)
    return report


def default_runner_python() -> Path:
    """numpy/librosa 가 있는 파이썬 우선 (ACE-Step venv → 현재 파이썬)."""
    venv = BASE_DIR / "audio_engines" / "venv_acestep" / "Scripts" / "python.exe"
    if venv.exists():
        return venv
    return Path(sys.executable)


# ---------------------------------------------------------------- CLI
def _load_script_map(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and it.get("file"):
                out[str(it["file"])] = it.get("text", "")
                out[Path(str(it["file"])).name] = it.get("text", "")
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="내 목소리 등록/학습")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("enroll")
    p1.add_argument("--name", required=True)
    p1.add_argument("--clips", nargs="+", required=True)
    p1.add_argument("--out-root", default=str(BASE_DIR / "trained_vocal_models"))
    p1.add_argument("--script-map", default=None, help="[{file,text}] JSON")
    p1.add_argument("--preset", default="standard", choices=["fast", "standard", "deep"])
    p1.add_argument("--acestep-python", default=None)
    p1.add_argument("--no-train", action="store_true")

    p2 = sub.add_parser("script")
    p2.add_argument("--limit", type=int, default=30)
    p2.add_argument("--out", default=None)
    p2.add_argument("--json", action="store_true")

    p3 = sub.add_parser("analyze")
    p3.add_argument("--clips", nargs="+", required=True)

    a = ap.parse_args()
    if a.cmd == "script":
        import voice_scripts
        if a.json:
            txt = json.dumps(voice_scripts.script_sentences(a.limit), ensure_ascii=False, indent=2)
        else:
            txt = voice_scripts.read_text(a.limit)
        if a.out:
            Path(a.out).write_text(txt, encoding="utf-8")
            print(f"저장: {a.out}")
        else:
            print(txt)
        return 0
    if a.cmd == "analyze":
        for c in a.clips:
            print(json.dumps(analyze_clip(c), ensure_ascii=False))
        return 0

    rep = enroll(a.name, a.clips, Path(a.out_root), _load_script_map(a.script_map),
                 train_lora=not a.no_train, preset=a.preset,
                 acestep_python=Path(a.acestep_python) if a.acestep_python else None)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
