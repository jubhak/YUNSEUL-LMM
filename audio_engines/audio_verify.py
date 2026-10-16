#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_verify.py - 생성된 음원이 '음악'인지 '백색잡음/무음'인지 숫자로 판별한다.

단독 실행:  python audio_verify.py batch_out/sample_song.mp3
러너에서 import 해서 사용:  from audio_verify import analyze, verdict

판정 기준 (휴리스틱이므로 '의심' 판단용, 보컬 존재 증명용은 아님)
- spectral flatness(중앙값): 백색잡음 ≈ 0.5 이상, 실제 음악은 보통 0.05 이하
- 8kHz 이상 에너지 비율: 백색잡음이면 크게 나오고(48kHz 기준 약 0.6), 음악은 보통 0.05 이하
- RMS 가 너무 낮으면 무음
- NaN 이 있으면 모델 연산이 깨진 것
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def analyze(path) -> dict:
    y, sr = sf.read(str(path), always_2d=True)
    y = y.astype(np.float32)
    n = y.shape[0]
    mono = y.mean(axis=1)

    info = {
        "path": str(path),
        "sample_rate": int(sr),
        "channels": int(y.shape[1]),
        "frames": int(n),
        "duration_sec": round(n / sr, 3),
        "has_nan": bool(np.isnan(y).any()),
        "peak": float(np.max(np.abs(y))) if n else 0.0,
        "rms": float(np.sqrt(np.mean(mono ** 2))) if n else 0.0,
        "clip_ratio": float(np.mean(np.abs(y) >= 0.999)) if n else 0.0,
    }

    frame, hop = 2048, 4096  # 일부 프레임만 샘플링 (속도)
    if n < frame * 2:
        info.update(median_flatness=None, hf_ratio=None, top24_bin_ratio=None)
        return info

    win = np.hanning(frame).astype(np.float32)
    flat = []
    acc = np.zeros(frame // 2 + 1, dtype=np.float64)
    for i in range(0, n - frame, hop):
        spec = np.abs(np.fft.rfft(mono[i:i + frame] * win)) ** 2 + 1e-12
        flat.append(float(np.exp(np.mean(np.log(spec))) / np.mean(spec)))
        acc += spec

    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    total = float(acc.sum()) + 1e-12
    info["median_flatness"] = round(float(np.median(flat)), 5)
    info["hf_ratio"] = round(float(acc[freqs >= 8000].sum() / total), 5)
    # 전체 에너지 중 상위 24개 FFT 빈 비중: 사인파 합성은 매우 높게 나오는 경향 (참고용)
    info["top24_bin_ratio"] = round(float(np.sort(acc)[-24:].sum() / total), 5)
    return info


def verdict(info: dict) -> tuple[str, list]:
    """('OK' | 'NOISE' | 'SILENT' | 'BROKEN', 사유 목록)"""
    reasons = []
    if info.get("has_nan"):
        return "BROKEN", ["출력에 NaN 포함 - 모델 연산이 깨졌습니다"]
    if info["rms"] < 0.003:
        return "SILENT", [f"RMS {info['rms']:.5f} - 사실상 무음"]
    fl, hf = info.get("median_flatness"), info.get("hf_ratio")
    if fl is not None and fl > 0.2:
        reasons.append(f"spectral flatness {fl} > 0.2 (백색잡음 특성)")
    if hf is not None and hf > 0.35:
        reasons.append(f"8kHz 이상 에너지 비율 {hf} > 0.35 (고역 잡음 특성)")
    if info["clip_ratio"] > 0.01:
        reasons.append(f"클리핑 샘플 {info['clip_ratio']:.2%}")
    if reasons and (
        (fl is not None and fl > 0.2) or (hf is not None and hf > 0.35)
    ):
        return "NOISE", reasons
    return "OK", reasons


def main():
    if len(sys.argv) < 2:
        print("usage: python audio_verify.py <audio-file>")
        return 1
    info = analyze(sys.argv[1])
    v, reasons = verdict(info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n판정: {v}")
    for r in reasons:
        print(f" - {r}")
    if info.get("top24_bin_ratio") and info["top24_bin_ratio"] > 0.7:
        print(" - [참고] 소수의 주파수에 에너지가 극단적으로 몰려 있음 (사인파 합성 가능성)")
    return 0 if v == "OK" else 3


if __name__ == "__main__":
    sys.exit(main())
