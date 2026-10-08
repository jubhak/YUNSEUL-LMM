#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_official_runner.py - Official ACE-Step 1.5 DiT + LM Vocal Music Generator
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Windows 콘솔 인코딩
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import transformers.modeling_utils

_orig_get_init_context = transformers.modeling_utils.PreTrainedModel.get_init_context
@classmethod
def _safe_get_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
    ctxs = _orig_get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
    return [c for c in ctxs if not (isinstance(c, torch.device) and c.type == "meta")]
transformers.modeling_utils.PreTrainedModel.get_init_context = _safe_get_init_context

ACESTEP_ROOT = Path(__file__).resolve().parent / "ACE-Step-1.5"
sys.path.insert(0, str(ACESTEP_ROOT))

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import set_global_gpu_config, get_gpu_config

def main():
    parser = argparse.ArgumentParser(description="Official ACE-Step 1.5 Vocal Music Generator")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--lyrics-file", default=None, help="Path to lyrics file")
    parser.add_argument("--steps", type=int, default=24, help="Diffusion inference steps (e.g. 24 for clean high-fi audio)")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio for vocal timbre conditioning")
    parser.add_argument("--task-type", default="text2music", choices=["text2music", "cover", "style_transfer", "repaint"], help="Task type")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_file)
    caption = prompt_path.read_text(encoding="utf-8").strip("\ufeff").strip() if prompt_path.exists() else "warm acoustic ballad"

    # 가사 로드
    lyrics = ""
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").strip("\ufeff").strip()
    else:
        # batch_out/lyrics-01.txt 자동 탐색
        candidates = [
            Path("batch_out/vocal_lyrics.txt"),
            prompt_path.parent / "vocal_lyrics.txt",
            Path("batch_out/lyrics-01.txt"),
            prompt_path.parent / "lyrics-01.txt",
        ]
        for c in candidates:
            if c.exists():
                lyrics = c.read_text(encoding="utf-8").strip("\ufeff").strip()
                print(f"[acestep-official] Loaded lyrics from {c} ({len(lyrics)} chars)")
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_save_dir = out_path.parent / "temp_acestep"
    temp_save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = str(ACESTEP_ROOT / "checkpoints")

    print("[acestep-official] Initializing DiT handler (acestep-v15-turbo)...")
    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(ACESTEP_ROOT),
        config_path="acestep-v15-turbo",
        device="cuda",
        offload_to_cpu=True,
    )
    if not ok:
        print(f"[acestep-official] DiT init failed: {status}")
        return 1

    print("[acestep-official] Initializing 5Hz LM handler (acestep-5Hz-lm-1.7B)...")
    llm_handler = LLMHandler()
    status, ok = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cuda",
        offload_to_cpu=True,
    )
    if not ok:
        print(f"[acestep-official] LM init failed: {status}")
        return 1

    # text2music 태스크에서는 reference_audio가 DiT 잠재 공간과 충돌하여 잡음(소음)을 유발하므로 None 처리.
    # 보컬 음색은 모델의 고유 보컬 특성 프롬프트 및 파라미터로 정밀 제어.
    safe_ref_audio = None
    if args.task_type in ("cover", "style_transfer", "repaint"):
        if args.ref_audio and Path(args.ref_audio).exists():
            safe_ref_audio = str(Path(args.ref_audio).resolve())

    # acestep-v15-turbo 모델의 최적 스텝은 8~10스텝 (24스텝 이상 시 디스틸 왜곡 방지)
    effective_steps = min(12, max(8, args.steps if args.steps <= 12 else 8))

    print(f"[acestep-official] Generating vocal song (duration={args.duration}s, steps={effective_steps})...")
    params = GenerationParams(
        task_type="text2music",
        thinking=True,
        use_cot_caption=True,
        use_cot_language=True,
        caption=caption,
        lyrics=lyrics,
        bpm=65,
        keyscale="C major",
        timesignature="4/4",
        vocal_language="ko",
        duration=args.duration,
        inference_steps=effective_steps,
        guidance_scale=1.0,
        enable_normalization=True,
        normalization_db=-1.0,
        reference_audio=safe_ref_audio,
        seed=-1,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format="mp3" if out_path.suffix.lower() == ".mp3" else "wav",
    )

    t0 = time.time()
    result = generate_music(dit_handler, llm_handler, params=params, config=config, save_dir=str(temp_save_dir))
    elapsed = time.time() - t0

    if result.success and result.audios:
        gen_file = result.audios[0]["path"]
        print(f"[acestep-official] Generated in {elapsed:.1f}s -> {gen_file}")
        shutil.copy2(gen_file, str(out_path))
        print(f"[acestep-official] Copied to final output: {out_path} ({out_path.stat().st_size} bytes)")
        return 0
    else:
        print(f"[acestep-official] Generation failed: {result.status_message} / error: {result.error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
