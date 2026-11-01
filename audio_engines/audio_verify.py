#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_verify.py - 생성된 음원이 '음악'인지 '백색잡음/무음'인지 숫자로 판별하고,
                   곡이 얼마나 '단조로운지(같은 마디 반복)'를 함께 측정한다.

단독 실행:  python audio_verify.py batch_out/sample_song.mp3
러너에서 import:  from audio_verify import analyze, verdict, monotony

[잡음 판정 기준] (휴리스틱: '의심' 판단용, 보컬 존재 증명용은 아님)
- spectral flatness(중앙값): 백색잡음 ≈ 0.5 이상, 실제 음악은 보통 0.05 이하
- 8kHz 이상 에너지 비율: 백색잡음이면 크게 나오고(48kHz 기준 약 0.6), 음악은 보통 0.05 이하
- RMS 가 너무 낮으면 무음, NaN 이 있으면 모델 연산이 깨진 것

[단조로움 판정] monotony()
곡 전체를 4초 블록으로 나눠 (1) 블록 간 음색 유사도, (2) 다이나믹 레인지,
(3) 온셋(타격) 변동성, (4) 대역 에너지 엔트로피, (5) 조 중심 변화 수를 본다.
같은 마디를 복사한 곡은 (1)이 1.0 에 붙고 (2)가 좁아져 점수가 높게 나온다.
의도된 미니멀 편곡도 높게 나올 수 있으므로 '테이크 선택용 상대 지표'로 쓴다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

BANDS = 24


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


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _empty(duration: float = 0.0) -> dict:
    return {"score": None, "penalty": 0, "duration_sec": round(float(duration), 3),
            "block_similarity": None, "dynamic_range_db": None,
            "onset_variability": None, "band_entropy": None, "tonal_centers": None,
            "note": "분석 구간이 너무 짧아 단조로움을 측정하지 못했습니다"}


def monotony(path_or_signal, sr: int = None, block_sec: float = 4.0) -> dict:
    """0(다채로움)~100(같은 마디 반복) 점수와 0~20 페널티를 돌려준다.

    path(str/Path) 또는 1차원/2차원 신호 배열을 받는다. numpy 만 사용한다.
    """
    if isinstance(path_or_signal, (str, Path)):
        y, sr = sf.read(str(path_or_signal), always_2d=True)
        mono = y.mean(axis=1).astype(np.float32)
    else:
        mono = np.asarray(path_or_signal, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = mono.reshape(-1)
        if sr is None:
            sr = 48000
    if sr is None:
        sr = 48000
    n = mono.size
    if n < int(sr * 2):
        return _empty(n / max(1, sr))

    frame, hop = 2048, 1024
    win = np.hanning(frame).astype(np.float32)
    starts = np.arange(0, n - frame, hop)
    if starts.size < 8:
        return _empty(n / max(1, sr))

    spec = np.empty((starts.size, frame // 2 + 1), dtype=np.float32)
    flux = np.zeros(starts.size, dtype=np.float32)
    prev = None
    for k, i in enumerate(starts):
        s = np.abs(np.fft.rfft(mono[i:i + frame] * win)).astype(np.float32)
        spec[k] = s
        if prev is not None:
            flux[k] = float(np.maximum(s - prev, 0.0).sum())
        prev = s
    flux /= (float(np.mean(flux)) + 1e-9)

    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = np.geomspace(60.0, max(120.0, min(sr * 0.45, 16000.0)), BANDS + 1)
    bands = np.zeros((starts.size, BANDS), dtype=np.float32)
    for b in range(BANDS):
        sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if sel.any():
            bands[:, b] = spec[:, sel].mean(axis=1)

    per_block = max(1, int(block_sec * sr / hop))
    nb = starts.size // per_block
    if nb < 2:
        return _empty(n / max(1, sr))
    cut = nb * per_block

    block_bands = bands[:cut].reshape(nb, per_block, BANDS).mean(axis=1)
    unit = block_bands / (np.linalg.norm(block_bands, axis=1, keepdims=True) + 1e-9)
    block_similarity = float(np.mean(np.sum(unit[:-1] * unit[1:], axis=1)))

    frames = mono[: (n // hop) * hop].reshape(-1, hop).astype(np.float64)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    nb_r = rms_db.size // per_block
    if nb_r >= 2:
        block_db = rms_db[: nb_r * per_block].reshape(nb_r, per_block).mean(axis=1)
        dynamic_range_db = float(np.percentile(block_db, 95) - np.percentile(block_db, 10))
    else:
        dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))

    block_flux = flux[:cut].reshape(nb, per_block).mean(axis=1)
    onset_variability = float(np.std(block_flux) / (np.mean(block_flux) + 1e-9))

    mean_bands = block_bands.mean(axis=0)
    prob = mean_bands / (mean_bands.sum() + 1e-12)
    band_entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(BANDS))

    tonal_centers = 0
    try:
        sel = (freqs >= 80.0) & (freqs <= 1200.0)
        if sel.any():
            pitch_class = np.round(12 * np.log2(freqs[sel] / 440.0) + 69).astype(int) % 12
            block_spec = spec[:cut, sel].reshape(nb, per_block, int(sel.sum())).mean(axis=(1, 2))
            chroma = np.zeros((nb, 12), dtype=np.float64)
            for pc in range(12):
                mask = pitch_class == pc
                if mask.any():
                    chroma[:, pc] = block_spec[:, mask].sum(axis=1)
            tonal_centers = int(len(set(np.argmax(chroma, axis=1).tolist())))
    except Exception:  # noqa: BLE001
        tonal_centers = 0

    score = 0.0
    score += 45.0 * _clip((block_similarity - 0.93) / 0.06, 0.0, 1.0)
    score += 25.0 * _clip((10.0 - dynamic_range_db) / 8.0, 0.0, 1.0)
    score += 20.0 * _clip((0.45 - onset_variability) / 0.45, 0.0, 1.0)
    score += 10.0 * _clip((0.70 - band_entropy) / 0.70, 0.0, 1.0)
    score -= 8.0 * _clip((tonal_centers - 3) / 4.0, 0.0, 1.0)
    score = round(max(0.0, min(100.0, score)), 1)

    return {
        "score": score,
        "penalty": int(round(score / 5.0)),
        "duration_sec": round(n / sr, 2),
        "block_sec": block_sec,
        "block_similarity": round(block_similarity, 4),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "onset_variability": round(onset_variability, 3),
        "band_entropy": round(band_entropy, 4),
        "tonal_centers": tonal_centers,
        "scope": "mix_only_not_musical_judgement",
    }


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
    mono = monotony(sys.argv[1])
    print("\n단조로움: " + json.dumps(mono, ensure_ascii=False))
    return 0 if v == "OK" else 3


if __name__ == "__main__":
    sys.exit(main())
