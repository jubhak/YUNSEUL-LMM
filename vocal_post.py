#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vocal_post.py - 생성된 노래를 '더 부드럽고 안정적으로' 다듬는 후처리 체인

의존성: numpy + ffmpeg 만 사용한다(librosa/torch 불필요).
그래서 메인 앱 파이썬(venv_acestep 아님)에서도 그대로 실행된다.

체인
 1) deharsh     : 1.8~5.5kHz 의 거친 대역만 완만하게 눌러 '쇳소리/찢어짐' 제거
 2) legato      : 프레임 RMS 엔벨로프를 완만하게 만들어 음절 어택의 튐을 줄임
 3) drift       : YIN 계열 f0 추적으로 한 음 안의 미세 피치 흔들림을 보정(기본 35%)
 4) double      : 두 번째 테이크를 살짝 어긋나게 낮은 볼륨으로 겹쳐 두께감 추가
 5) space       : 합성 plate IR 컨볼루션 리버브(기본 wet 0.12)
 6) master      : ffmpeg acompressor + loudnorm(I=-14, TP=-1.0) → 320k mp3

단독 실행 예:
  python vocal_post.py --input song.mp3 --output song_polished.mp3 --preset natural
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

import numpy as np

SR = 48000
PRESETS = {
    #            strength, wet,  cut,  legato, double
    "soft":     dict(strength=0.80, wet=0.16, cut=1.00, legato=0.70, double=-12.0),
    "natural":  dict(strength=0.60, wet=0.12, cut=0.80, legato=0.55, double=-14.0),
    "rich":     dict(strength=0.45, wet=0.20, cut=0.55, legato=0.40, double=-10.0),
}
MAX_CENTS = 55.0        # 한 번에 보정하는 최대 폭 (과보정 방지)
VOICED_CONF = 0.30      # f0 신뢰도 하한
F0_MIN, F0_MAX = 70.0, 900.0


# ---------------------------------------------------------------- 기본 유틸
def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg 를 찾지 못했습니다. PATH 에 ffmpeg 를 추가해 주세요.")
    return exe


def decode(path, sr: int = SR, channels: int = 2) -> np.ndarray:
    """어떤 형식이든 float32 (n, ch) 로 디코딩."""
    raw = subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    pad = (-len(y)) % channels
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    return np.ascontiguousarray(y.reshape(-1, channels))


def encode(y: np.ndarray, sr: int = SR) -> bytes:
    if y.ndim == 1:
        y = y[:, None]
    y = np.ascontiguousarray(y.astype(np.float32))
    return subprocess.run(
        [ffmpeg_bin(), "-v", "error", "-f", "f32le", "-ar", str(sr),
         "-ac", str(y.shape[1]), "-i", "-", "-c:a", "pcm_s16le", "-f", "wav", "-"],
        input=y.tobytes(), capture_output=True, check=True).stdout


def write_wav(y: np.ndarray, path, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encode(y, sr))


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.zeros((0, frame), dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(n, frame), strides=(hop * s, s), writeable=False).astype(np.float32)


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _stft(x: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    return np.fft.rfft(_frames(x, frame, hop) * _hann(frame), axis=1)


def _istft(spec: np.ndarray, frame: int = 2048, hop: int = 512, n: int = 0) -> np.ndarray:
    w = _hann(frame)
    frames = np.fft.irfft(spec, n=frame, axis=1).astype(np.float64)
    # 길이 보정: 마지막 프레임이 신호 끝을 덮지 못하면 원본 길이까지 패딩한다.
    # (패딩 없이 n 보다 짧은 배열을 돌려주면 호출부의 대입에서 shape 오류가 난다)
    total = max(max(0, len(spec) - 1) * hop + frame, int(n or 0))
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    ww = w.astype(np.float64) ** 2
    for i, f in enumerate(frames):
        a = i * hop
        out[a:a + frame] += f * w
        norm[a:a + frame] += ww
    out /= np.maximum(norm, 1e-8)
    return out[:n] if n else out


def rms_db(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2) + 1e-12))
    return 20.0 * math.log10(max(r, 1e-9))


# ---------------------------------------------------------------- 1) 거친 대역 완화
def deharsh(y: np.ndarray, sr: int, amount: float, frame: int = 2048, hop: int = 512) -> np.ndarray:
    """2~4.5kHz 의 자극적인 대역만 종 모양으로 부드럽게 감쇠."""
    if amount <= 0:
        return y
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    # log2 영역 종 모양 (중심 3.1kHz)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(freqs, 1.0) / 3100.0)
    bell = np.exp(-(lf ** 2) / (2 * 0.42 ** 2))
    bell[freqs < 1500] = 0.0
    gain = 1.0 - (amount * 0.55) * bell
    out = y.astype(np.float32).copy()
    for ch in range(y.shape[1]):
        spec = _stft(out[:, ch], frame, hop)
        spec *= gain[None, :]
        out[:, ch] = _istft(spec, frame, hop, n=len(out)).astype(np.float32)
    return out


# ---------------------------------------------------------------- 2) 레가토(엔벨로프 완만화)
def legato(y: np.ndarray, sr: int, amount: float, win_ms: float = 46.0) -> np.ndarray:
    """프레임 RMS 엔벨로프를 따라가며 튀는 어택/급격한 볼륨 변화를 완만하게 만든다."""
    if amount <= 0:
        return y
    hop = max(1, int(sr * 0.010))
    mono = y.mean(axis=1).astype(np.float32)
    n = 1 + len(mono) // hop
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = mono[i * hop:(i + 1) * hop]
        env[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
    w = max(1, int(win_ms / 10.0))
    kernel = np.ones(w, dtype=np.float32) / w
    smooth = np.convolve(env, kernel, mode="same")
    gain = np.clip((smooth + 1e-6) / (env + 1e-6), 0.30, 3.0)
    gain = 1.0 + (gain - 1.0) * amount          # 보정 강도
    # 프레임 게인을 샘플 단위로 보간
    centers = np.arange(n) * hop + hop // 2
    gi = np.interp(np.arange(len(mono)), centers, gain)
    gi = np.convolve(gi, np.ones(64, dtype=np.float32) / 64, mode="same")
    out = y.astype(np.float32).copy()
    for ch in range(out.shape[1]):
        out[:, ch] *= gi.astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out *= 0.999 / peak
    return out


# ---------------------------------------------------------------- 3) 미세 피치 보정
def estimate_f0(mono: np.ndarray, sr: int, frame: int = 1024, hop: int = 256):
    """자기상관 기반 f0 추적 (YIN 축소판). 반환: f0(Hz), 신뢰도, 프레임 중심(초)."""
    x = np.asarray(mono, dtype=np.float32)
    fr = _frames(x, frame, hop)
    if fr.shape[0] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr *= _hann(frame)
    spec = np.fft.rfft(fr, n=2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame]
    ac = ac / (ac[:, :1] + 1e-9)
    lo = max(2, int(sr / F0_MAX))
    hi = min(frame - 2, int(sr / F0_MIN))
    seg = ac[:, lo:hi + 1]
    if seg.shape[1] < 3:
        return np.zeros(fr.shape[0]), np.zeros(fr.shape[0]), np.arange(fr.shape[0]) * hop / sr
    idx = np.argmax(seg, axis=1)
    conf = seg[np.arange(len(idx)), idx].astype(np.float64)
    lag = (idx + lo).astype(np.float64)
    # 포물선 보간
    for i in range(len(idx)):
        j = idx[i] + lo
        if 2 <= j < frame - 1:
            a, b, c = ac[i, j - 1], ac[i, j], ac[i, j + 1]
            denom = (a - 2 * b + c)
            if abs(denom) > 1e-9:
                lag[i] = j + 0.5 * (a - c) / denom
    f0 = np.where(conf >= 0.05, sr / np.maximum(lag, 1e-6), 0.0)
    f0[(f0 < F0_MIN) | (f0 > F0_MAX)] = 0.0
    centers = (np.arange(fr.shape[0]) * hop + frame / 2) / sr
    return f0, conf, centers


def _median_filter(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x.copy()
    pad = w // 2
    p = np.pad(x, pad, mode="edge")
    return np.array([np.median(p[i:i + w]) for i in range(len(x))], dtype=np.float64)


def _shift_grain(seg: np.ndarray, ratio: float) -> np.ndarray:
    """길이는 유지하고 피치만 ratio 배로 (그레인 단위 리샘플)."""
    n = len(seg)
    if n < 4 or abs(ratio - 1.0) < 1e-4:
        return seg.copy()
    idx = np.clip(np.arange(n, dtype=np.float64) * ratio, 0, n - 1.0001)
    return np.interp(idx, np.arange(n, dtype=np.float64), seg)


def correct_drift(y: np.ndarray, sr: int, strength: float, grain: int = 1536) -> tuple[np.ndarray, dict]:
    """한 음(note) 안에서 흔들리는 f0 를 목표 중심으로 당긴다. 과보정 금지."""
    if strength <= 0:
        return y, {"applied": False}
    mono = y.mean(axis=1)
    f0, conf, centers = estimate_f0(mono, sr)
    if len(f0) == 0:
        return y, {"applied": False}

    step = 256.0 / sr
    n_grains = max(1, int(len(y) / (grain / 2)))
    gcent = np.arange(n_grains) * (grain / 2) / sr + (grain / 2) / sr
    fi = np.clip((gcent / max(step, 1e-9)).astype(int), 0, len(f0) - 1)
    f0g = f0[fi].astype(np.float64)
    confg = conf[fi].astype(np.float64)

    voiced = (f0g > 0) & (confg >= VOICED_CONF)
    if voiced.sum() < 8:
        return y, {"applied": False, "reason": "voiced frame 부족"}

    # 음 중심 추정: 충분히 긴 중앙값 창 → 음(note) 단위 기준선
    ref = np.zeros_like(f0g)
    idxs = np.where(voiced)[0]
    vals = f0g[idxs]
    smoothed = _median_filter(vals, 21)
    ref[idxs] = smoothed

    dev = np.zeros_like(f0g)
    ok = voiced & (ref > 0)
    dev[ok] = 1200.0 * np.log2(f0g[ok] / ref[ok])
    target = np.clip(-dev * strength, -MAX_CENTS, MAX_CENTS)
    target[~ok] = 0.0
    target[np.abs(dev) > 160] = 0.0          # 큰 도약(멜로디 이동)은 건드리지 않음

    # 보정 곡선 평활화 (급격한 비율 변화 방지)
    target = _median_filter(target, 9)
    target = np.convolve(target, np.ones(5) / 5.0, mode="same")
    if np.all(np.abs(target) < 0.5):
        return y, {"applied": False, "reason": "보정량 미미"}

    ratio = np.power(2.0, target / 1200.0)
    hop = grain // 2
    win = _hann(grain).astype(np.float64)
    out = np.zeros((len(y) + grain, y.shape[1]), dtype=np.float64)
    norm = np.zeros(len(y) + grain, dtype=np.float64)
    for gi in range(n_grains):
        a = gi * hop
        b = min(a + grain, len(y))
        if b - a < 8:
            break
        r = float(ratio[gi]) if gi < len(ratio) else 1.0
        w = win[: b - a]
        for ch in range(y.shape[1]):
            seg = y[a:b, ch].astype(np.float64) * w
            out[a:b, ch] += _shift_grain(seg, r) * w
        norm[a:b] += w * w
    norm = np.maximum(norm, 1e-8)
    out = out / norm[:, None]
    out = out[: len(y)].astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out *= 0.999 / peak
    stats = {
        "applied": True,
        "voiced_ratio": round(float(voiced.mean()), 3),
        "max_correction_cents": round(float(np.max(np.abs(target))), 2),
        "mean_abs_correction_cents": round(float(np.mean(np.abs(target))), 2),
    }
    return out, stats


# ---------------------------------------------------------------- 4) 두 테이크 겹치기
def double_track(y: np.ndarray, other_path, sr: int, gain_db: float) -> tuple[np.ndarray, dict]:
    try:
        o = decode(other_path, sr, y.shape[1])
    except Exception as e:  # noqa: BLE001
        return y, {"applied": False, "reason": str(e)}
    n = min(len(y), len(o))
    if n < sr * 3:
        return y, {"applied": False, "reason": "테이크가 너무 짧음"}
    a, b = y[:n].mean(axis=1), o[:n].mean(axis=1)
    # 거칠게 정렬 (앞 20초 상호상관)
    k = min(len(a), sr * 20)
    fa = np.fft.rfft(a[:k], n=2 * k)
    fb = np.fft.rfft(b[:k], n=2 * k)
    cc = np.fft.irfft(fa * np.conj(fb), n=2 * k)
    lag = int(np.argmax(np.abs(cc[: sr // 2]))) if len(cc) > sr // 2 else 0
    if lag > 0:
        b = np.concatenate([np.zeros(lag, dtype=np.float32), b])[:n]
    g = 10 ** (gain_db / 20.0)
    mixed = y[:n].astype(np.float32).copy()
    mixed += g * o[:n]
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.999:
        mixed *= 0.999 / peak
    return mixed, {"applied": True, "align_samples": lag, "gain_db": gain_db}


# ---------------------------------------------------------------- 5) 공간계
def reverb(y: np.ndarray, sr: int, wet: float = 0.12, decay: float = 1.25) -> np.ndarray:
    if wet <= 0:
        return y
    rng = np.random.default_rng(7)
    n = int(sr * decay)
    t = np.arange(n, dtype=np.float32) / sr
    irs = []
    for ch in range(max(2, y.shape[1])):
        noise = rng.normal(0, 1, n).astype(np.float32)
        ir = noise * np.exp(-t * (5.0 / decay)).astype(np.float32)
        # 초기 반사 몇 개
        ir[: int(sr * 0.012)] *= 0.25
        ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-9)
        irs.append(ir)
    size = len(y) + n
    nfft = 1 << int(math.ceil(math.log2(size)))
    out = y.astype(np.float32).copy()
    for ch in range(y.shape[1]):
        H = np.fft.rfft(irs[ch % len(irs)], nfft)
        X = np.fft.rfft(y[:, ch], nfft)
        wetsig = np.fft.irfft(X * H, nfft)[: len(y)].astype(np.float32)
        out[:, ch] = (1 - wet) * y[:, ch] + wet * wetsig * 2.2
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out *= 0.999 / peak
    return out


# ---------------------------------------------------------------- 6) 마스터링
def master(src_wav, dst_path, target_lufs: float = -14.0) -> None:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    af = (f"acompressor=threshold=-18dB:ratio=2.5:attack=8:release=180:makeup=1.5,"
          f"loudnorm=I={target_lufs}:TP=-1.0:LRA=11")
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", str(src_wav), "-af", af, "-ar", "48000"]
    if dst_path.suffix.lower() == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
    cmd.append(str(dst_path))
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not dst_path.exists():
        raise RuntimeError("ffmpeg 마스터링 실패: " + r.stderr.decode("utf-8", "replace")[-400:])


# ---------------------------------------------------------------- 통합
def polish_song(input_path, output_path, preset: str = "natural", strength: float | None = None,
                double_track_path=None, pitch: bool = True, keep_intermediate=None) -> dict:
    """노래 한 곡을 다듬어 output_path 로 저장하고, 실제 적용 내역과 수치를 돌려준다."""
    cfg = dict(PRESETS.get(preset, PRESETS["natural"]))
    if strength is not None:
        cfg["strength"] = float(max(0.0, min(1.0, strength)))
    stages, notes = [], []
    y = decode(input_path, SR, 2)
    if len(y) < SR:
        raise RuntimeError("음원이 너무 짧습니다.")
    before_db = rms_db(y)
    mono_before = y.mean(axis=1)
    f0b, cb, _ = estimate_f0(mono_before, SR)
    vb = (f0b > 0) & (cb >= VOICED_CONF)
    if vb.sum() > 20:
        z = 1200 * np.log2(f0b[vb] / np.median(f0b[vb]))
        jitter_before = float(np.std(z))
    else:
        jitter_before = float("nan")

    if double_track_path:
        y, st = double_track(y, double_track_path, SR, cfg["double"])
        stages.append("double")
        notes.append(st)

    y = deharsh(y, SR, cfg["cut"])
    stages.append("deharsh")
    y = legato(y, SR, cfg["legato"])
    stages.append("legato")

    pitch_stats = {"applied": False}
    if pitch:
        y, pitch_stats = correct_drift(y, SR, cfg["strength"])
        if pitch_stats.get("applied"):
            stages.append("pitch_drift")

    y = reverb(y, SR, cfg["wet"])
    stages.append("space")

    f0a, ca, _ = estimate_f0(y.mean(axis=1), SR)
    va = (f0a > 0) & (ca >= VOICED_CONF)
    jitter_after = float("nan")
    if va.sum() > 20:
        z = 1200 * np.log2(f0a[va] / np.median(f0a[va]))
        jitter_after = float(np.std(z))

    tmp = Path(tempfile.mkdtemp(prefix="vocalpost_")) / "pre_master.wav"
    write_wav(y, tmp, SR)
    master(tmp, output_path)
    try:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()
    except Exception:
        pass

    result = {
        "input": str(input_path), "output": str(output_path), "preset": preset,
        "stages": stages, "rms_db_in": round(before_db, 2),
        "rms_db_out": round(rms_db(decode(output_path, SR, 2)), 2),
        "pitch_jitter_cents_in": None if math.isnan(jitter_before) else round(jitter_before, 2),
        "pitch_jitter_cents_out": None if math.isnan(jitter_after) else round(jitter_after, 2),
        "pitch": pitch_stats, "extra": notes,
    }
    if keep_intermediate:
        Path(keep_intermediate).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="노래 보컬 후처리 (부드럽게/안정적으로)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--preset", default="natural", choices=sorted(PRESETS))
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--double-track", default=None)
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    r = polish_song(a.input, a.output, a.preset, a.strength, a.double_track,
                    pitch=not a.no_pitch, keep_intermediate=a.report)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
