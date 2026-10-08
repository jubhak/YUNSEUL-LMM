#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py - Download official weights for ACE-Step 1.5 and GPT-SoVITS
"""

import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# Windows 콘솔 인코딩
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def download_acestep():
    print("=== [1/2] Downloading official ACE-Step 1.5 weights (9.5 GB) ===")
    target_dir = Path(__file__).resolve().parent / "audio_engines" / "ACE-Step-1.5" / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        repo_id="ACE-Step/Ace-Step1.5",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"=== ACE-Step 1.5 weights downloaded in {time.time()-t0:.1f}s ===")

def download_gpt_sovits():
    print("=== [2/2] Downloading official GPT-SoVITS pretrained models ===")
    target_dir = Path(__file__).resolve().parent / "audio_engines" / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models"
    target_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"=== GPT-SoVITS pretrained models downloaded in {time.time()-t0:.1f}s ===")

if __name__ == "__main__":
    download_acestep()
    download_gpt_sovits()
    print("All weights downloaded successfully!")
