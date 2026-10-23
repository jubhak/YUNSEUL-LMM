#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vocal_coach.py - "진짜 잘 부르는 가수처럼" 만드는 보컬 코치 체인

근거 논문/자료 (코드에 단계별로 대응)
 [1] VibE-SVC: Vibrato Extraction with High-frequency F0 Contour for Singing Voice Conversion
     (Choi et al., arXiv:2505.20794, 2025)
     → 비브라토를 '느린 드리프트'와 분리해 다뤄야 음정은 고치고 비브라토는 살릴 수 있다.
       본 모듈은 f0 를 (a) 느린 기준선 + (b) 비브라토 성분으로 분해한다.
 [2] Singing Voice Synthesis with Vibrato Modeling and Latent Energy Representation
     (Song et al., ICASSP 2022, arXiv:2211.00996)
     → 비브라토 '깊이/속도'와 에너지(다이내믹)를 별도로 모델링해야 표현력이 산다.
 [3] Analysis and modification of excitation source characteristics for singing voice synthesis
     (Roebel et al., ICASSP 2012) → 여기(勵起) 성분 변조로 비브라토/숨소리를 만든다.
 [4] The Quantification of Emotional Expressions of Vocal Vibrato (ACM, 2024)
     → 비브라토는 감정 지각과 직결. 5~7Hz, 수십 cent, '늦게 시작(onset delay)'이 자연스럽다.
 [5] Phoneme-Level Prosody Anchoring for Emotion-Preserving SVC (SEAIS 2026)
     → 음소 단위 운율을 고정해야 감정이 보존된다 → 무성 구간/도약은 건드리지 않는다.
 [6] Muse-SVS (IEEE TASLP 2023) → 감정 강도에 따라 비브라토·다이내믹을 조절.

체인 (polish_star)
 1) analyze      : 현재 보컬의 음정 흔들림/비브라토/다이내믹/HNR 측정
 2) pitch_style  : 느린 드리프트만 보정 + 목표 비브라토를 '지연 시작'으로 주입 [1][2][4][5]
 3) phrase_dyn   : 프레이즈 단위 크레셴도/어택 완화/호흡 앞 여백 [2]
 4) singer_formant: 2.8~3.1kHz '싱어 포먼트'를 좁게 올려 소리가 앞으로 나오게 [3]
 5) air          : HNR 에 맞춘 고역 공기감(숨소리) [3]
 6) legato/deharsh → vocal_post 재사용
 7) master       : loudnorm I=-14, TP=-1.0, 320k

프리셋: ballad(애절) / power(파워) / idol(밝고 또렷) / natural(자연스러움)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import vocal_post as vp

SR = 48000
VOICED_CONF = 0.30

PRESETS = {
    #            drift, vib_rate, vib_depth, vib_delay, dyn, formant, air, legato, cut, wet
    "ballad":  dict(drift=0.55, rate=5.2, depth=42, delay=0.22, dyn=0.85, formant=1.4, air=1.6, legato=0.55, cut=0.85, wet=0.15),
    "power":   dict(drift=0.70, rate=5.8, depth=32, delay=0.14, dyn=1.15, formant=2.6, air=0.8, legato=0.35, cut=0.70, wet=0.10),
    "idol":    dict(drift=0.75, rate=6.1, depth=26, delay=0.10, dyn=0.95, formant=2.2, air=1.2, legato=0.30, cut=0.75, wet=0.08),
    "natural": dict(drift=0.45, rate=5.0, depth=22, delay=0.26, dyn=0.55, formant=0.9, air=1.0, legato=0.40, cut=0.60, wet=0.12),
}

TARGET_VIB_PRESENCE = 0.55      # 유성 구간 중 비브라토가 들려야 하는 비율


# ---------------------------------------------------------------- 분석
def f0_decompose(f0, conf, sr, hop=256, slow_sec=0.42):
    """f0 → (느린 기준선, 비브라토 성분[cent], 유성 마스크)."""
    voiced = (f0 > 0) & (conf >= VOICED_CONF)
    n = len(f0)
    if n == 0:
        return np.zeros(0), np.zeros(0), voiced
    win = max(9, int(slow_sec * sr / hop) | 1)
    slow = np.zeros(n)
    idx = np.where(voiced)[0]
    if len(idx) == 0:
        return slow, np.zeros(n), voiced
    # voiced 구간만 이어붙여 중앙값 필터 → 기준선
    vals = f0[idx]
    pad = win // 2
    p = np.pad(vals, pad, mode="edge")
    med = np.array([np.median(p[i:i + win]) for i in range(len(vals))], dtype=np.float64)
    # 2차 평활
    k = np.ones(5) / 5.0
    med = np.convolve(med, k, mode="same")
    slow[idx] = med
    dev = np.zeros(n)
    ok = voiced & (slow > 0)
    dev[ok] = 1200.0 * np.log2(f0[ok] / slow[ok])
    return slow, dev, voiced


def vocal_health(y, sr=SR) -> dict:
    """음정 흔들림·비브라토·다이내믹·HNR 지표."""
    mono = y.mean(axis=1)
    f0, conf, _ = vp.estimate_f0(mono, sr)
    slow, dev, voiced = f0_decompose(f0, conf, sr)
    out = {"voiced_ratio": round(float(voiced.mean()), 3) if len(voiced) else 0.0}
    if voiced.sum() < 20:
        out.update({"jitter_cents": None, "vibrato_rate_hz": None, "vibrato_depth_cents": None})
        return out
    d = dev[voiced]
    # 비브라토가 없는 구간은 0 근처, 있는 구간은 진동 → 표준편차의 2배를 깊이로 본다
    out["jitter_cents"] = round(float(np.std(d)), 2)
    # 느린 드리프트: dev 를 0.6초 이동평균으로 평활해 비브라토(빠른 진동)를 지운 뒤 표준편차.
    # 근거: VibE-SVC (arXiv:2505.20794) — 느린 드리프트와 비브라토를 분리해 측정해야
    #       "비브라토를 넣었더니 음정이 나빠졌다"는 잘못된 판정이 나오지 않는다.
    w = max(3, int(0.6 * sr / 256)) | 1
    slow_dev = np.convolve(d, np.ones(w) / w, mode="same")
    edge = w // 2
    core = slow_dev[edge:len(slow_dev) - edge] if len(slow_dev) > w else slow_dev
    out["drift_cents"] = round(float(np.std(core)) if len(core) else 0.0, 2)
    out["vibrato_depth_cents"] = round(float(np.percentile(np.abs(d), 90) * 2), 2)

    # 비브라토 속도: dev 의 영교차로 추정
    hop_t = 256.0 / sr
    seg = dev[voiced]
    zc = np.where(np.diff(np.signbit(seg)))[0]
    if len(zc) > 4:
        periods = np.diff(zc) * hop_t * 2.0
        periods = periods[(periods > 0.08) & (periods < 0.5)]
        out["vibrato_rate_hz"] = round(float(1.0 / np.median(periods)), 2) if len(periods) else None
    else:
        out["vibrato_rate_hz"] = 0.0
    out["vibrato_presence"] = round(float(np.mean(np.abs(d) > 12)), 3)

    # 다이내믹: 프레이즈 RMS 의 상위/하위 폭
    hop = int(sr * 0.02)
    n = len(mono) // hop
    rms = np.array([np.sqrt(np.mean(mono[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    voiced_rms = rms[rms > (rms.max() * 0.05 + 1e-9)]
    if len(voiced_rms) > 20:
        out["dynamics_db"] = round(float(20 * np.log10(
            np.percentile(voiced_rms, 95) / max(np.percentile(voiced_rms, 15), 1e-9))), 2)
    # HNR (자기상관 피크)
    r = np.correlate(mono[: sr * 3], mono[: sr * 3], mode="full")[min(len(mono), sr * 3) - 1:]
    r = r / (r[0] + 1e-12)
    lo, hi = int(sr / 900), int(sr / 60)
    peak = float(np.max(r[lo:hi])) if len(r) > hi else 0.0
    out["hnr_db"] = round(float(10 * math.log10(max(peak, 1e-6) / max(1 - peak, 1e-6))), 1)
    return out


# ---------------------------------------------------------------- 2) 음정 + 비브라토
def pitch_style(y, sr=SR, drift=0.55, rate=5.2, depth_cents=42, delay=0.18,
                max_total_cents=95.0, grain=1536) -> tuple:
    """느린 드리프트만 보정하고, 목표 비브라토를 '지연 시작'으로 더한다.

    [1][2][4][5] 근거: 음정(느린 성분)과 비브라토(빠른 성분)를 분리해 다룬다.
    도약(160 cent 초과)은 멜로디이므로 손대지 않는다.
    """
    mono = y.mean(axis=1)
    f0, conf, _ = vp.estimate_f0(mono, sr)
    slow, dev, voiced = f0_decompose(f0, conf, sr)
    if len(f0) == 0 or voiced.sum() < 8:
        return y, {"applied": False, "reason": "유성 구간 부족"}

    hop = 256.0 / sr
    n_grains = max(1, int(len(y) / (grain / 2)))
    gcent = np.arange(n_grains) * (grain / 2) / sr
    fi = np.clip((gcent / hop).astype(int), 0, len(f0) - 1)
    f0g, slowg, devg, vg = f0[fi], slow[fi], dev[fi], voiced[fi]

    # 1) 드리프트 보정 (dev 전체에서 '느린 성분'은 이미 빠져 있으므로 dev 자체를 줄인다)
    corr = -devg * drift
    # 2) 비브라토 주입: 유성 + 지속음(도약 없음) 구간에만, 앞부분은 페이드 인
    sustained = vg & (np.abs(devg) < 160) & (slowg > 0)
    # 연속 유성 구간 길이 계산 → 0.35초 이상 지속되는 음에만 비브라토 (발라드 창법)
    run, runs = 0, np.zeros(len(vg))
    for i in range(len(vg)):
        run = run + 1 if vg[i] else 0
        runs[i] = run
    long_note = runs >= int(0.35 / (grain / 2 / sr))
    add_mask = sustained & long_note
    # 지연 시작: 각 음의 시작 후 delay 초 동안 0→1
    fade = np.zeros(len(vg))
    if add_mask.any():
        count = 0
        need = max(1, int(delay / (grain / 2 / sr)))
        for i in range(len(vg)):
            if not vg[i]:
                count = 0
                fade[i] = 0.0
                continue
            count += 1
            fade[i] = min(1.0, count / need)
    t = gcent
    vib = (depth_cents / 2.0) * np.sin(2 * np.pi * rate * t)
    target = corr + np.where(add_mask, vib * fade, 0.0)
    target = np.clip(target, -max_total_cents, max_total_cents)
    # 급격한 비율 변화 방지
    target = vp._median_filter(target, 5)
    if np.all(np.abs(target) < 0.4):
        return y, {"applied": False, "reason": "보정량 미미"}

    ratio = np.power(2.0, target / 1200.0)
    hop_g = grain // 2
    win = vp._hann(grain).astype(np.float64)
    out = np.zeros((len(y) + grain, y.shape[1]), dtype=np.float64)
    norm = np.zeros(len(y) + grain, dtype=np.float64)
    for gi in range(n_grains):
        a = gi * hop_g
        b = min(a + grain, len(y))
        if b - a < 8:
            break
        r = float(ratio[gi]) if gi < len(ratio) else 1.0
        w = win[: b - a]
        for ch in range(y.shape[1]):
            seg = y[a:b, ch].astype(np.float64) * w
            out[a:b, ch] += vp._shift_grain(seg, r) * w
        norm[a:b] += w * w
    out = (out / np.maximum(norm, 1e-8)[:, None])[: len(y)].astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out *= 0.999 / peak
    stats = {
        "applied": True,
        "drift_correction": f"{drift:.2f}",
        "vibrato_rate_hz": rate,
        "vibrato_depth_cents": depth_cents,
        "vibrato_delay_sec": delay,
        "max_shift_cents": round(float(np.max(np.abs(target))), 1),
        "vibrato_target_note_ratio": round(float(add_mask.mean()), 3),
    }
    return out, stats


# ---------------------------------------------------------------- 3) 프레이즈 다이내믹
def phrase_dynamics(y, sr=SR, amount=0.85) -> tuple:
    """프레이즈 끝으로 갈수록 살짝 커지고(크레셴도), 시작 어택은 부드럽게.

    [2] 에너지(latent energy)를 별도로 다루는 것에 대응한다.
    """
    if amount <= 0:
        return y, {"applied": False}
    hop = int(sr * 0.02)
    mono = y.mean(axis=1)
    n = max(1, len(mono) // hop)
    rms = np.array([np.sqrt(np.mean(mono[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    if len(rms) < 10:
        return y, {"applied": False}
    thr = max(rms.max() * 0.06, 1e-4)
    gains = np.ones(n, dtype=np.float64)
    phrases = 0
    i = 0
    while i < n:
        if rms[i] <= thr:
            i += 1
            continue
        j = i
        while j < n and rms[j] > thr:
            j += 1
        L = j - i
        if L >= 6:
            phrases += 1
            ramp = np.linspace(-0.03, 0.05, L)
            gains[i:j] = 1.0 + ramp * amount
            atk = max(2, int(0.045 / 0.02))
            if L > atk * 2:
                gains[i:i + atk] *= np.linspace(0.90, 1.0, atk) ** amount
        i = j + 1
    gi = np.interp(np.arange(len(mono)), np.arange(n) * hop + hop / 2, gains)
    gi = np.convolve(gi, np.ones(48, dtype=np.float32) / 48, mode="same")
    out = y.astype(np.float32).copy()
    for ch in range(out.shape[1]):
        out[:, ch] *= gi.astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out *= 0.999 / peak
    return out, {"applied": True, "phrases": phrases}


# ---------------------------------------------------------------- 4) 싱어 포먼트
def singer_formant(y, sr=SR, db=1.5) -> np.ndarray:
    """2.9kHz 근처를 좁게 올리고 4.2kHz 는 살짝 눌러 '앞으로 나오는' 소리로."""
    if db <= 0:
        return y
    n_fft, hop = 4096, 1024
    ff = np.fft.rfftfreq(n_fft, 1.0 / sr)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(ff, 1.0) / 2900.0)
        lf2 = np.log2(np.maximum(ff, 1.0) / 4300.0)
    curve = db * np.exp(-(lf ** 2) / (2 * 0.115 ** 2)) - (db * 0.35) * np.exp(-(lf2 ** 2) / (2 * 0.10 ** 2))
    curve[ff < 1200] = 0.0
    return _apply_eq(y, sr, curve, n_fft, hop)


# ---------------------------------------------------------------- 5) 공기감
def air(y, sr=SR, db=1.2) -> np.ndarray:
    if db <= 0:
        return y
    n_fft, hop = 4096, 1024
    ff = np.fft.rfftfreq(n_fft, 1.0 / sr)
    curve = np.where(ff > 7500, db * np.clip((ff - 7500) / 7000.0, 0, 1), 0.0)
    return _apply_eq(y, sr, curve, n_fft, hop)


def _apply_eq(y, sr, curve_db, n_fft=4096, hop=1024):
    out = y.astype(np.float32).copy()
    gain = 10 ** (curve_db / 20.0)
    win = np.hanning(n_fft).astype(np.float32)
    for ch in range(y.shape[1]):
        x = out[:, ch]
        if len(x) < n_fft * 2:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
        spec = np.fft.rfft(frames * win, axis=1) * gain[None, :]
        rec = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)
        acc = np.zeros(len(x) + n_fft, dtype=np.float32)
        nrm = np.zeros(len(x) + n_fft, dtype=np.float32)
        for i, f in enumerate(rec):
            a = i * hop
            acc[a:a + n_fft] += f * win
            nrm[a:a + n_fft] += win ** 2
        out[:, ch] = (acc / np.maximum(nrm, 1e-6))[: len(x)]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out *= 0.999 / peak
    return out


# ---------------------------------------------------------------- 통합
def polish_star(input_path, output_path, preset="ballad", intensity=1.0,
                report_path=None) -> dict:
    """가수처럼 들리게 만드는 전체 체인. 실제 적용 내역과 전/후 지표를 남긴다."""
    cfg = dict(PRESETS.get(preset, PRESETS["ballad"]))
    k = max(0.0, min(1.5, float(intensity)))
    y = vp.decode(input_path, SR, 2)
    if len(y) < SR:
        raise RuntimeError("음원이 너무 짧습니다.")
    before = vocal_health(y)
    stages = []

    y, st = pitch_style(y, SR, drift=cfg["drift"] * k, rate=cfg["rate"],
                        depth_cents=cfg["depth"] * k, delay=cfg["delay"])
    stages.append({"stage": "pitch_style", **st})

    y, st = phrase_dynamics(y, SR, cfg["dyn"] * k)
    stages.append({"stage": "phrase_dynamics", **st})

    y = singer_formant(y, SR, cfg["formant"] * k)
    stages.append({"stage": "singer_formant", "db": round(cfg["formant"] * k, 2)})

    y = air(y, SR, cfg["air"] * k)
    stages.append({"stage": "air", "db": round(cfg["air"] * k, 2)})

    y = vp.deharsh(y, SR, cfg["cut"])
    y = vp.legato(y, SR, cfg["legato"])
    y = vp.reverb(y, SR, cfg["wet"])
    stages.append({"stage": "deharsh/legato/space"})

    tmp = Path(output_path).with_suffix(".premaster.wav")
    vp.write_wav(y, tmp, SR)
    vp.master(tmp, output_path)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    after = vocal_health(vp.decode(output_path, SR, 2))

    result = {
        "input": str(input_path), "output": str(output_path), "preset": preset,
        "intensity": k, "stages": stages,
        "before": before, "after": after,
        "improved": {
            "drift_down_cents": _delta(before.get("drift_cents"), after.get("drift_cents")),
            "jitter_down_cents": _delta(before.get("jitter_cents"), after.get("jitter_cents")),
            "vibrato_depth_change": _delta(before.get("vibrato_depth_cents"),
                                           after.get("vibrato_depth_cents")),
            "dynamics_change_db": _delta(before.get("dynamics_db"), after.get("dynamics_db")),
        },
    }
    if report_path:
        Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return result


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="가수처럼: 비브라토·다이내믹·싱어포먼트 코치")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--preset", default="ballad", choices=sorted(PRESETS))
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    print(json.dumps(polish_star(a.input, a.output, a.preset, a.intensity, a.report),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
