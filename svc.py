#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svc.py - 생성된 노래의 '보컬 음색'을 내 목소리 쪽으로 옮기는 단계 (Singing Voice Conversion)

두 가지 백엔드를 지원한다.

1) rvc  : RVC(Retrieval-based Voice Conversion) 가 설치되어 있고 내 모델(.pth)이 있으면
          실제 신경망 변환을 수행한다. 이 경로만이 '진짜 음색 복제'에 해당한다.
          필요: audio_engines/RVC (RVC-WebUI), assets/weights/<name>.pth
2) dsp  : 설치 없이도 항상 동작하는 근사 변환.
          - 기본: 원곡 가창 음정 보존. 낭독 f0로 강제 이동하지 않음
          - 내 발화의 평균 스펙트럼 포락선과 곡 보컬의 차이를 EQ 로 보정 (음색 결)
          - HNR(숨소리) 차이를 고역 공기감/디에서로 완만하게 보정
          ※ dsp 는 '음색 근사'이며 화자 복제가 아니다. 이 점을 결과 리포트에 명시한다.

사용:
  python svc.py --vocal song.mp3 --profile trained_vocal_models/내목소리/voice_profile.json \
                --output out.wav --backend auto
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SR = 48000

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None


# ---------------------------------------------------------------- 공통
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg 를 찾지 못했습니다.")
    return exe


def decode(path, sr: int = SR, ch: int = 2):
    raw = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(ch), "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    pad = (-len(y)) % ch
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    return np.ascontiguousarray(y.reshape(-1, ch))


def write_wav(y, path, sr: int = SR, codec="pcm_s24le"):
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    if y.ndim == 1:
        y = y[:, None]
    if not np.isfinite(y).all():
        raise ValueError("오디오에 NaN/Inf가 포함되어 저장을 중단했습니다.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(y.shape[1]),
         "-i", "-", "-c:a", codec, "-f", "wav", str(path)],
        input=y.tobytes(), capture_output=True)
    if raw.returncode != 0:
        raise RuntimeError("WAV 저장 실패: " + raw.stderr.decode("utf-8", "replace")[-300:])


def load_profile(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- RVC
def rvc_root():
    for c in (BASE_DIR / "audio_engines" / "RVC", BASE_DIR / "audio_engines" / "RVC-WebUI",
              BASE_DIR / "RVC"):
        if c.exists():
            return c
    return None


def find_rvc_model(name: str):
    root = rvc_root()
    if not root:
        return None, None
    weights = list((root / "assets" / "weights").rglob("*.pth")) if \
        (root / "assets" / "weights").exists() else []
    index = list((root / "logs").rglob(f"{name}*/*.index")) if (root / "logs").exists() else []
    exact = [p for p in weights if p.stem == name]
    return (exact[0] if len(exact) == 1 else None), (index[0] if len(index) == 1 else None)


def rvc_available(name: str) -> bool:
    root = rvc_root()
    if not root:
        return False
    cli = root / "tools" / "infer_cli.py"
    if not cli.exists():
        cli = root / "infer_cli.py"
    model, _ = find_rvc_model(name)
    return bool(cli.exists() and model)


def convert_rvc(vocal_path, profile: dict, out_wav, pitch_shift: int = 0,
                index_rate: float = 0.75, protect: float = 0.33) -> dict:
    root = rvc_root()
    cli = root / "tools" / "infer_cli.py"
    if not cli.exists():
        cli = root / "infer_cli.py"
    model, index = find_rvc_model(profile.get("name", ""))
    py = root / "venv" / "Scripts" / "python.exe"
    exe = str(py) if Path(py).exists() else sys.executable
    cmd = [exe, str(cli), "--input", str(vocal_path), "--output", str(out_wav),
           "--model", str(model), "--f0method", "rmvpe", "--f0up_key", str(int(pitch_shift)),
           "--index_rate", f"{index_rate}", "--filter_radius", "3", "--resample_sr", "0",
           "--rms_mix_rate", "0.25", "--protect", f"{protect}", "--wav_format", "wav"]
    if index:
        cmd += ["--index", str(index)]
    r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0 or not Path(out_wav).exists():
        raise RuntimeError("RVC 변환 실패: " + (r.stderr or r.stdout or "")[-400:])
    return {"backend": "rvc", "model": str(model), "index": str(index) if index else None,
            "f0_up_key": int(pitch_shift)}


# ---------------------------------------------------------------- DSP 근사
def _spectral_envelope(y, sr: int, n_fft: int = 4096, bands: int = 48):
    """장기 평균 스펙트럼을 로그 밴드 48개로 요약."""
    mono = y.mean(axis=1) if y.ndim > 1 else y
    if len(mono) < n_fft * 2:
        mono = np.pad(mono, (0, n_fft * 2 - len(mono)))
    hop = n_fft // 2
    fr = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop]
    if len(fr) > 600:
        fr = fr[:: max(1, len(fr) // 600)]
    spec = np.abs(np.fft.rfft(fr * np.hanning(n_fft), axis=1)) ** 2
    avg = spec.mean(axis=0) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(80, min(16000, sr / 2 - 100), bands + 1)
    out = np.zeros(bands)
    for i in range(bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = avg[m].mean() if m.any() else avg[np.argmin(np.abs(freqs - edges[i]))]
    return 10 * np.log10(out + 1e-12), (edges[:-1] + edges[1:]) / 2


def _f0_median(y, sr: int) -> float:
    mono = y.mean(axis=1) if y.ndim > 1 else y
    frame, hop = int(sr * 0.05), int(sr * 0.02)
    if len(mono) < frame * 4:
        return 0.0
    fr = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop].astype(np.float32)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr *= np.hanning(frame).astype(np.float32)
    spec = np.fft.rfft(fr, n=2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame]
    ac /= (ac[:, :1] + 1e-9)
    lo, hi = int(sr / 900), int(sr / 60)
    idx = np.argmax(ac[:, lo:hi], axis=1)
    conf = ac[np.arange(len(idx)), idx + lo]
    f0 = np.where(conf > 0.3, sr / (idx + lo), 0.0)
    f0 = f0[(f0 > 60) & (f0 < 900)]
    return float(np.median(f0)) if len(f0) > 5 else 0.0


def _apply_eq_curve(y, sr: int, freqs, gains_db):
    from audio_dsp_safe import spectral_gain
    n_fft, hop = 4096, 1024
    ff = np.fft.rfftfreq(n_fft, 1.0 / sr)
    curve = np.interp(ff, freqs, gains_db, left=gains_db[0], right=gains_db[-1])
    return spectral_gain(y, 10 ** (curve / 20.0), n_fft, hop)



def _pitch_shift(y, sr: int, semitones: float):
    """전체 곡을 반음 단위로 이동 (멜로디 유지, 음색 근사). ffmpeg rubberband/atempo 조합."""
    if abs(semitones) < 0.05:
        return y
    ratio = 2 ** (semitones / 12.0)
    tmp_in = Path(tempfile.mkdtemp(prefix="svc_")) / "in.wav"
    tmp_out = tmp_in.with_name("out.wav")
    write_wav(y, tmp_in, sr)
    # rubberband 가 있으면 품질이 좋다
    filters = [
        f"rubberband=pitch={ratio:.6f}",
        None,
    ]
    ok = False
    for f in filters:
        if f is None:
            continue
        cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(tmp_in), "-af", f,
               "-ar", str(sr), "-c:a", "pcm_s24le", str(tmp_out)]
        if subprocess.run(cmd, capture_output=True).returncode == 0 and tmp_out.exists():
            ok = True
            break
    if not ok:
        # 폴백: 리샘플로 피치 이동 후 atempo 로 길이 복구
        cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(tmp_in),
               "-af", f"asetrate={int(sr * ratio)},aresample={sr},atempo={1 / ratio:.6f}",
               "-ar", str(sr), "-c:a", "pcm_s24le", str(tmp_out)]
        subprocess.run(cmd, capture_output=True, check=False)
        ok = tmp_out.exists()
    if not ok:
        return y
    out = decode(tmp_out, sr, y.shape[1])
    shutil.rmtree(tmp_in.parent, ignore_errors=True)
    n = min(len(out), len(y))
    return out[:n]


def convert_dsp(vocal_path, profile: dict, out_wav) -> dict:
    y = decode(vocal_path, SR, 2)
    notes = [] if profile.get("allow_singing_transpose", False) else ["낭독 음역으로 강제 하향하지 않음: 원곡 멜로디/음정 유지"]
    # 1) 원곡 음정 보존. 명시적으로 허용한 프로필만 전체 ±2반음 이동
    target = float(profile.get("f0_median_hz") or 0)
    cur = _f0_median(y, SR)
    semis = 0.0
    if profile.get("allow_singing_transpose", False) and target > 0 and cur > 0:
        semis = 12 * math.log2(target / cur)
        semis = max(-2.0, min(2.0, semis))
        if abs(semis) < 0.25:
            semis = 0.0
    if semis:
        y = _pitch_shift(y, SR, semis)
        notes.append(f"음역을 내 목소리 쪽으로 {semis:+.2f}반음 이동")
    # 2) 스펙트럼 포락선 매칭 (내 발화 평균 vs 곡 보컬 평균)
    clips = [c["file"] for c in (profile.get("clips") or []) if Path(c.get("file", "")).exists()]
    env_gain = None
    if clips:
        speech = np.concatenate([decode(c, SR, 1).mean(axis=1)
                                 for c in clips[:12] if Path(c).exists()])[:, None]
        env_s, freqs = _spectral_envelope(speech, SR)
        env_v, _ = _spectral_envelope(y, SR)
        diff = env_s - env_v
        # 발화 음색으로 가창을 과보정하지 않도록 ±3dB 클램프 + 약한 적용
        diff = np.clip(diff, -3.0, 3.0)
        k = np.ones(5) / 5.0
        diff = np.convolve(diff, k, mode="same")
        env_gain = 0.35  # 100% 적용하면 부자연스러움
        y = _apply_eq_curve(y, SR, freqs, diff * env_gain)
        notes.append(f"내 목소리 평균 음색에 맞춰 밴드 EQ 보정(최대 ±{np.max(np.abs(diff) * env_gain):.1f}dB)")
    # 3) 숨소리(HNR) 방향 보정
    hnr = float(profile.get("hnr_db") or 0)
    if hnr < 8:
        y = _apply_eq_curve(y, SR, np.array([6000.0, 9000.0, 12000.0]),
                            np.array([1.6, 2.2, 2.0]))
        notes.append("숨소리 많은 목소리 → 고역 공기감 +2dB")
    elif hnr > 16:
        y = _apply_eq_curve(y, SR, np.array([6500.0, 9500.0]), np.array([-1.6, -2.0]))
        notes.append("깨끗한 목소리 → 고역 자극 완화 -2dB")
    write_wav(y, out_wav, SR)
    return {"backend": "dsp", "pitch_shift_semitones": round(semis, 2),
            "notes": notes,
            "disclaimer": "dsp 는 음색 '근사'이며 화자 복제가 아닙니다. "
                          "완전한 복제는 RVC 학습 또는 ACE-Step LoRA 경로를 사용하세요."}


def convert_vocal(vocal_path, profile_path, out_wav, backend: str = "auto") -> dict:
    profile = load_profile(profile_path)
    if backend in ("auto", "rvc") and rvc_available(profile.get("name", "")):
        try:
            return convert_rvc(vocal_path, profile, out_wav)
        except Exception as e:  # noqa: BLE001
            if backend == "rvc":
                raise
            raise RuntimeError("학습된 RVC 변환 실패: 다른 음색으로 대체하지 않고 원본을 유지합니다.") from e
    if backend == "rvc":
        raise RuntimeError("RVC 모델을 찾지 못했습니다 (audio_engines/RVC, assets/weights).")
    return convert_dsp(vocal_path, profile, out_wav)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="내 목소리로 노래 음색 변환 (SVC)")
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "rvc", "dsp"])
    a = ap.parse_args()
    r = convert_vocal(a.vocal, a.profile, a.output, a.backend)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
