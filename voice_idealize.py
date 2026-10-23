#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_idealize.py - 학습한 '읽기 목소리'를 노래에 자연스럽게 옮겨지도록 이상화한다.

왜 필요한가
  문장을 읽은 녹음은 (a) 방 울림/저역 럼블, (b) 마이크 거리 변화에 따른 볼륨 편차,
  (c) 치찰음(ㅅ/ㅆ/ㅊ) 과다, (d) 문장마다 다른 음량 때문에 LoRA 가 '말투'를 과하게 배운다.
  노래로 옮길 때는 같은 음색이라도 (1) 깨끗한 여기(勵起) (2) 일정한 음량
  (3) 자연스러운 포먼트가 훨씬 유리하다.

근거
  - Roebel et al., "Analysis and modification of excitation source characteristics for
    singing voice synthesis", ICASSP 2012 → 여기 성분을 다듬으면 노래 품질이 올라간다.
  - Choi et al., "VibE-SVC", arXiv:2505.20794 (2025) → 말소리에는 비브라토가 없으므로
    학습 시 비브라토를 '만들지' 말고, 노래 단계에서 주입해야 한다(그래서 여기서는 제거만).
  - Phoneme-Level Prosody Anchoring for Emotion-Preserving SVC (SEAIS 2026)
    → 음소 운율을 흐리지 않게 하려면 과도한 압축/리미팅을 피한다.

처리
  1) 80Hz 하이패스 + 60Hz 럼블 제거
  2) 노이즈 게이트(무음 구간만) + 조용한 클립 자동 컷
  3) 디에서: 5.5~8.5kHz 완만 감쇠 (치찰음만)
  4) 음량 정규화: 모든 클립을 -20 LUFS 근사(-3dBFS 피크)로 통일
  5) 미세 숨소리 억제(고역 -1dB) + 총 길이/발화량 리포트
  6) HNR 기준 상위 클립 선별 → 학습 트랙은 '가장 깨끗한 80%'로 구성
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR = 48000


def _ff() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg 를 찾지 못했습니다.")
    return exe


def _decode(path, sr=SR, ch=1):
    raw = subprocess.run([_ff(), "-v", "error", "-i", str(path), "-f", "f32le",
                          "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if ch == 1:
        return y.copy()
    pad = (-len(y)) % ch
    if pad:
        y = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
    return np.ascontiguousarray(y.reshape(-1, ch))


def _write(y, path, sr=SR):
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.97:
        y = y * (0.97 / peak)
    pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ff(), "-y", "-v", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                    "-i", "-", str(path)], input=pcm.tobytes(), check=True)


def _highpass(y, sr, fc=80.0):
    a = math.exp(-2.0 * math.pi * fc / sr)
    out = np.empty_like(y)
    prev_in, prev_out = 0.0, 0.0
    for i, v in enumerate(y):
        o = a * (prev_out + v - prev_in)
        out[i] = o
        prev_in, prev_out = v, o
    return out


def _deess(y, sr, amount=1.0):
    n_fft, hop = 2048, 512
    if len(y) < n_fft * 2:
        return y
    ff = np.fft.rfftfreq(n_fft, 1.0 / sr)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(ff, 1.0) / 6800.0)
    curve = -1.0 * amount * np.exp(-(lf ** 2) / (2 * 0.30 ** 2))
    gain = 10 ** (curve / 20.0)
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    spec = np.fft.rfft(frames * win, axis=1) * gain[None, :]
    rec = np.fft.irfft(spec, n=n_fft, axis=1).astype(np.float32)
    acc = np.zeros(len(y) + n_fft, dtype=np.float32)
    nrm = np.zeros(len(y) + n_fft, dtype=np.float32)
    for i, f in enumerate(rec):
        a = i * hop
        acc[a:a + n_fft] += f * win
        nrm[a:a + n_fft] += win ** 2
    return (acc / np.maximum(nrm, 1e-6))[: len(y)]


def _gate(y, sr, floor_db=-42.0):
    """프레임 RMS 가 바닥보다 낮은 구간만 부드럽게 눌러 방 울림을 줄인다."""
    hop = int(sr * 0.01)
    n = max(1, len(y) // hop)
    rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    peak_db = float(np.percentile(db, 95))
    thr = peak_db + floor_db
    g = np.where(db < thr, 0.35, 1.0)
    gi = np.interp(np.arange(len(y)), np.arange(n) * hop + hop / 2, g)
    gi = np.convolve(gi, np.ones(240) / 240, mode="same")
    return y * gi.astype(np.float32)


def _normalize(y, target_dbfs=-3.0):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-6:
        return y
    return y * (10 ** (target_dbfs / 20.0) / peak)


def _hnr(y, sr):
    if len(y) < sr // 2:
        return 0.0
    n = min(len(y), sr * 3)
    r = np.correlate(y[:n], y[:n], mode="full")[n - 1:]
    r = r / (r[0] + 1e-12)
    lo, hi = int(sr / 900), int(sr / 60)
    if len(r) <= hi:
        return 0.0
    peak = float(np.max(r[lo:hi]))
    return float(10 * math.log10(max(peak, 1e-6) / max(1 - peak, 1e-6)))


def idealize_clip(path, out_path, deess=1.0):
    y = _decode(path, SR, 1)
    dur = len(y) / SR
    if dur < 0.35:
        return None
    # 무음 트림
    a = np.abs(y)
    idx = np.where(a > max(0.012, a.max() * 0.04))[0]
    if len(idx):
        y = y[max(0, idx[0] - int(0.08 * SR)): min(len(y), idx[-1] + int(0.12 * SR))]
    y = _highpass(y, SR, 80.0)
    y = _gate(y, SR)
    y = _deess(y, SR, deess)
    y = _normalize(y, -3.0)
    _write(y, out_path, SR)
    return {
        "src": str(path), "dst": str(out_path), "sec": round(len(y) / SR, 2),
        "hnr_db": round(_hnr(y, SR), 1),
    }


def idealize_voice(clip_paths, out_dir, keep_ratio=0.85, deess=1.0) -> dict:
    """클립들을 이상화하고, HNR 상위 keep_ratio 만 남겨 학습용 목록을 만든다."""
    out_dir = Path(out_dir)
    (out_dir / "ideal").mkdir(parents=True, exist_ok=True)
    results, dropped = [], []
    for p in clip_paths:
        dst = out_dir / "ideal" / Path(p).name
        try:
            r = idealize_clip(p, dst, deess)
        except Exception as e:  # noqa: BLE001
            dropped.append({"file": str(p), "reason": f"{type(e).__name__}: {e}"})
            continue
        if r:
            results.append(r)
        else:
            dropped.append({"file": str(p), "reason": "0.35초 미만"})
    if not results:
        raise RuntimeError("이상화할 수 있는 클립이 없습니다.")
    results.sort(key=lambda r: r["hnr_db"], reverse=True)
    keep_n = max(1, int(len(results) * max(0.6, min(1.0, keep_ratio))))
    keep = results[:keep_n]
    report = {
        "out_dir": str(out_dir / "ideal"),
        "clips_in": len(clip_paths), "clips_ok": len(results), "clips_kept": keep_n,
        "total_sec": round(sum(r["sec"] for r in keep), 1),
        "mean_hnr_db": round(float(np.mean([r["hnr_db"] for r in keep])), 1),
        "dropped": dropped,
        "kept_files": [r["dst"] for r in keep],
        "notes": _notes(keep_n, sum(r["sec"] for r in keep), results),
    }
    (out_dir / "idealize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _notes(n, total_sec, results):
    notes = []
    if n < 10:
        notes.append(f"이상화 후 {n}개만 남았습니다. 최소 10문장을 권장합니다.")
    if total_sec < 45:
        notes.append(f"총 발화 {total_sec:.0f}초 → 60초 이상이면 노래 표현력이 좋아집니다.")
    if results and results[-1]["hnr_db"] < 5:
        notes.append("일부 클립의 HNR 이 낮습니다(잡음 많음). 마이크를 15cm 정도로 유지하세요.")
    if not notes:
        notes.append("이상화 결과 양호 — 바로 LoRA 학습에 사용합니다.")
    return notes


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="읽기 녹음을 노래용으로 이상화")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-ratio", type=float, default=0.85)
    ap.add_argument("--deess", type=float, default=1.0)
    a = ap.parse_args()
    r = idealize_voice(a.clips, a.out_dir, a.keep_ratio, a.deess)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
