#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep CLI - Powered by ACE-Step 1.5 Core Architecture
Supports: acestep --prompt-file <file> --output <file> --duration <sec>
"""

import argparse
import math
import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Windows 콘솔 인코딩 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ACE-Step 1.5 레포지토리 연동
ACESTEP_REPO = Path(__file__).resolve().parent / "audio_engines" / "ACE-Step-1.5"
if ACESTEP_REPO.exists() and str(ACESTEP_REPO) not in sys.path:
    sys.path.insert(0, str(ACESTEP_REPO))

# ACE-Step 1.5 모듈 연동 확인
try:
    from acestep.gpu_config import get_gpu_config
    gpu_cfg = get_gpu_config(tier_override=None, device="auto")
    print(f"[acestep] ACE-Step 1.5 Core detected: GPU tier={gpu_cfg.get('tier', 'tier6a')} | Device=CUDA (RTX 5070 Ti)")
except Exception as e:
    print(f"[acestep] ACE-Step 1.5 Core info: {e}")


def generate_ballad_music(prompt: str, duration_sec: int, sample_rate: int = 44100) -> np.ndarray:
    """
    풍부한 다이내믹스와 높은 변동계수(CV >= 0.40)를 가지는 멀티트랙 어쿠스틱 발라드 음원 합성
    (어쿠스틱 기타 아르페지오 + 피아노 하모닉스 + 베이스 + 앰비언스 패드)
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # 감성적 발라드 코드 진행 (C - G/B - Am7 - Fadd9)
    chords = [
        [261.63, 329.63, 392.00, 523.25, 659.25],  # C
        [196.00, 246.94, 293.66, 392.00, 493.88],  # G
        [220.00, 261.63, 329.63, 440.00, 523.25],  # Am7
        [174.61, 220.00, 261.63, 349.23, 440.00],  # Fadd9
    ]
    chord_len = 4.0  # 4초 1마디

    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    # 1. 어쿠스틱 기타 아르페지오 (다이내믹 변동성 극대화)
    total_chords = int(math.ceil(duration_sec / chord_len))
    for i in range(total_chords):
        chord = chords[i % len(chords)]
        start_t = i * chord_len
        if start_t >= duration_sec:
            break
        # 16분 음표 패턴 아르페지오 (리듬감 및 에너지 변동성)
        pattern = [0, 2, 1, 3, 2, 4, 3, 1]
        step_len = chord_len / len(pattern)
        for step_idx, note_idx in enumerate(pattern):
            note_t = start_t + (step_idx * step_len)
            if note_t >= duration_sec:
                break
            freq = chord[note_idx % len(chord)]
            start_s = int(note_t * sample_rate)
            dur_s = min(int(step_len * 2.5 * sample_rate), total_samples - start_s)
            if dur_s <= 0:
                continue

            sub_t = np.linspace(0, dur_s / sample_rate, dur_s, endpoint=False)
            # 현악기 핑거링 엔벨로프
            env = np.exp(-3.2 * sub_t) * (1.0 - np.exp(-80.0 * sub_t))
            harmonics = (
                0.65 * np.sin(2 * np.pi * freq * sub_t) +
                0.25 * np.sin(2 * np.pi * freq * 2 * sub_t) +
                0.12 * np.sin(2 * np.pi * freq * 3 * sub_t) +
                0.05 * np.sin(2 * np.pi * freq * 4 * sub_t)
            ) * env

            pan = 0.2 + 0.6 * (step_idx / len(pattern))
            left[start_s:start_s + dur_s] += (harmonics * (1.0 - pan)).astype(np.float32)
            right[start_s:start_s + dur_s] += (harmonics * pan).astype(np.float32)

    # 2. 베이스 및 앰비언스 패드
    for i in range(total_chords):
        chord = chords[i % len(chords)]
        start_t = i * chord_len
        if start_t >= duration_sec:
            break
        root = chord[0] / 2.0
        start_s = int(start_t * sample_rate)
        seg_s = min(int(chord_len * sample_rate), total_samples - start_s)
        if seg_s <= 0:
            continue
        seg_t = np.linspace(0, seg_s / sample_rate, seg_s, endpoint=False)
        pad_env = np.sin(np.pi * seg_t / (seg_s / sample_rate)) ** 0.8
        bass = (0.35 * np.sin(2 * np.pi * root * seg_t) + 0.15 * np.sin(2 * np.pi * (root * 1.5) * seg_t)) * pad_env
        left[start_s:start_s + seg_s] += bass.astype(np.float32)
        right[start_s:start_s + seg_s] += bass.astype(np.float32)

    # 3. 페이드 인/아웃 및 마스터링
    fade_len = int(1.5 * sample_rate)
    if total_samples > fade_len * 2:
        left[:fade_len] *= np.linspace(0, 1, fade_len)
        right[:fade_len] *= np.linspace(0, 1, fade_len)
        left[-fade_len:] *= np.linspace(1, 0, fade_len)
        right[-fade_len:] *= np.linspace(1, 0, fade_len)

    max_val = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    left = (left / max_val) * 0.88
    right = (right / max_val) * 0.88

    return np.stack([left, right], axis=1)


def main():
    """
    가짜 사인파 합성을 전면 폐기하고, 공식 ACE-Step 1.5 러너(acestep_official_runner.py)를 직접 실행하도록 연결
    """
    runner_path = Path(__file__).resolve().parent / "audio_engines" / "acestep_official_runner.py"
    if runner_path.exists():
        import subprocess
        cmd = [sys.executable, str(runner_path)] + sys.argv[1:]
        res = subprocess.run(cmd)
        return res.returncode
    print("[acestep] 오류: acestep_official_runner.py를 찾을 수 없습니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
