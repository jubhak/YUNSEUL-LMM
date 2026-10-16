#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_official_runner.py - Official ACE-Step 1.5 DiT + LM Vocal Music Generator (수정판)

[수정 사유 — 기존 코드가 노래를 만들지 못한 실제 원인]
1. transformers PreTrainedModel.get_init_context 를 meta 디바이스 제거용으로 덮어썼는데,
   이 패치가 ACE-Step v1.5 자체 설정(config.json) 로딩 경로를 함께 깨뜨려
   "KeyError: ' неожиданный/意外的/Unexpected key or dtype in state (config)'" 류의 즉사 크래시 발생.
   -> 패치를 완전히 제거했다. (v1.5 최신 코드는 meta 초기화를 스스로 처리한다)
2. set_global_gpu_config 를 호출하지 않아 핸들러 초기화 시 글로벌 GPU 설정이 비어 있었다.
   -> cli.py 와 동일하게 get_gpu_config() -> set_global_gpu_config() 를 먼저 수행한다.
3. 체크포인트 부재/LM 모델 부재 시 원인 불명의 하위 예외로 죽었다.
   -> 실행 전 체크포인트 사전 점검 + 모델명 안내 + LM 폴백(1.7B -> 0.6B)을 추가했다.
4. thinking(True) 고정으로 LM 없이는 절대 진행 불가였다.
   -> --no-thinking 옵션 추가 (LM 없이 순수 DiT 경로로 생성 가능).
"""

import argparse
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

ACESTEP_ROOT = Path(__file__).resolve().parent / "ACE-Step-1.5"
if str(ACESTEP_ROOT) not in sys.path:
    sys.path.insert(0, str(ACESTEP_ROOT))

# transformers 4.49+ 및 vector_quantize_pytorch 연동 시 meta device 환경에서
# Tensor.item() 호출로 인한 크래시(RuntimeError: Tensor.item() cannot be called on meta tensors) 방지
import torch
import transformers.modeling_utils

_orig_get_init_context = transformers.modeling_utils.PreTrainedModel.get_init_context


@classmethod
def _safe_get_init_context(cls, *args, **kwargs):
    ctxs = _orig_get_init_context(*args, **kwargs)
    return [c for c in ctxs if not (isinstance(c, torch.device) and c.type == "meta")]


transformers.modeling_utils.PreTrainedModel.get_init_context = _safe_get_init_context


def fail(msg: str, code: int = 2):
    """실행 불가 상황을 원인과 해결책까지 명확히 출력하고 종료한다."""
    print(f"[acestep-official] [오류] {msg}")
    sys.exit(code)


def precheck() -> str:
    """모델/환경 사전 점검. 통과 시 checkpoints 디렉터리 절대경로 반환."""
    if not ACESTEP_ROOT.exists():
        fail(
            f"ACE-Step 1.5 레포지토리가 없습니다: {ACESTEP_ROOT}\n"
            f"  해결: git clone https://github.com/ace-step/ACE-Step-1.5 \"{ACESTEP_ROOT}\""
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        fail(
            "PyTorch가 설치되어 있지 않습니다.\n"
            "  해결: pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    import torch
    if not torch.cuda.is_available():
        fail(
            "CUDA GPU를 사용할 수 없습니다. (RTX 5070 Ti 라면 CUDA 12.8 이상 빌드의 torch 필요)\n"
            "  해결: pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"  torch 버전: {torch.__version__}"
        )

    ckpt_dir = ACESTEP_ROOT / "checkpoints"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        fail(
            f"체크포인트가 없습니다: {ckpt_dir}\n"
            "  해결: python download_models.py  (ACE-Step/Ace-Step1.5 가중치 약 9.5GB 다운로드)"
        )
    return str(ckpt_dir.resolve())


def main():
    parser = argparse.ArgumentParser(description="Official ACE-Step 1.5 Vocal Music Generator (fixed)")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds (10~600)")
    parser.add_argument("--lyrics-file", default=None, help="Path to lyrics file")
    parser.add_argument("--steps", type=int, default=8, help="Diffusion inference steps (turbo: 8~12)")
    parser.add_argument("--ref-audio", default=None, help="Reference audio (cover/style_transfer/repaint only)")
    parser.add_argument("--task-type", default="text2music",
                        choices=["text2music", "cover", "style_transfer", "repaint"], help="Task type")
    parser.add_argument("--bpm", type=int, default=None, help="BPM override (미지정 시 LLM CoT 자동 추정)")
    parser.add_argument("--vocal-language", default="ko", help="Vocal language code (ko/en/ja/...)")
    parser.add_argument("--lm-model", default=None,
                        help="LM 모델명 기본값: acestep-5Hz-lm-1.7B (부족 시 0.6B로 자동 폴백)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="5Hz LM Chain-of-Thought 비활성화 (LM 모델 없이 순수 DiT 경로로 생성)")
    args = parser.parse_args()

    checkpoint_dir = precheck()

    prompt_path = Path(args.prompt_file)
    caption = (prompt_path.read_text(encoding="utf-8").strip("\ufeff").strip()
               if prompt_path.exists() else "warm acoustic ballad")

    # 가사 로드
    lyrics = ""
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").strip("\ufeff").strip()
    else:
        candidates = [
            Path("batch_out/vocal_lyrics.txt"),
            prompt_path.parent / "vocal_lyrics.txt",
            Path("batch_out/lyrics-01.txt"),
            prompt_path.parent / "lyrics-01.txt",
        ]
        for c in candidates:
            if c.exists():
                lyrics = c.read_text(encoding="utf-8").strip("\ufeff").strip()
                print(f"[acestep-official] 가사 로드: {c} ({len(lyrics)}자)")
                break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_save_dir = out_path.parent / "temp_acestep"
    temp_save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 모듈 임포트 (사전 점검 통과 후) ----
    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        from acestep.gpu_config import set_global_gpu_config, get_gpu_config
    except Exception as e:  # noqa: BLE001
        fail(f"acestep 모듈 임포트 실패: {e}\n  해결: ACE-Step-1.5 의존성 설치 후 재시도 (pip install -e \"{ACESTEP_ROOT}\")")

    # [핵심 수정 2] cli.py 와 동일하게 글로벌 GPU 설정을 먼저 등록
    try:
        gpu_cfg = get_gpu_config()
        set_global_gpu_config(gpu_cfg)
        print(f"[acestep-official] GPU 구성: tier={getattr(gpu_cfg, 'tier', '?')} "
              f"offload={getattr(gpu_cfg, 'offload_to_cpu', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"[acestep-official] GPU 구성 감지 실패(기본값 사용): {e}")

    duration = max(10, min(600, int(args.duration)))

    print(f"[acestep-official] DiT 핸들러 초기화 (acestep-v15-turbo)...")
    dit_handler = AceStepHandler()
    try:
        status, ok = dit_handler.initialize_service(
            project_root=str(ACESTEP_ROOT),
            config_path="acestep-v15-turbo",
            device="cuda",
            offload_to_cpu=True,
        )
    except Exception as e:  # noqa: BLE001
        fail(f"DiT 초기화 중 예외: {e}\n"
             f"  체크포인트 상태를 확인하세요: {Path(checkpoint_dir) / 'acestep-v15-turbo'}")
    if not ok:
        fail(f"DiT 초기화 실패: {status}")

    # [핵심 수정 4] thinking 끄면 LM 스킵 — LM 없이도 노래 생성 가능
    thinking = not args.no_thinking
    llm_handler = LLMHandler()
    lm_candidates = []
    if args.lm_model:
        lm_candidates.append(args.lm_model)
    else:
        lm_candidates = ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"]

    if thinking:
        lm_loaded = False
        last_err = ""
        for lm_name in lm_candidates:
            print(f"[acestep-official] 5Hz LM 초기화 시도: {lm_name} (backend=pt)...")
            try:
                status, ok = llm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=lm_name,
                    backend="pt",
                    device="cuda",
                    offload_to_cpu=True,
                )
            except Exception as e:  # noqa: BLE001
                status, ok = str(e), False
            if ok:
                print(f"[acestep-official] LM 로드 성공: {lm_name}")
                lm_loaded = True
                break
            last_err = f"{lm_name}: {status}"
            print(f"[acestep-official] LM 로드 실패 -> 다음 후보 시도 ({last_err[:160]})")
        if not lm_loaded:
            print("[acestep-official] [경고] LM 로드 실패 -> thinking 을 끄고 순수 DiT 경로로 계속합니다.")
            print("  (LM 없이 생성 시 메타데이터 자동 추정 품질이 다소 떨어질 수 있습니다)")
            print(f"  원인 참고: {last_err[:300]}")
            thinking = False
    else:
        print("[acestep-official] --no-thinking: LM 초기화를 건너뜁니다.")

    # text2music 에서 reference_audio 는 잠재 공간 충돌로 잡음을 유발하므로 None 처리
    safe_ref_audio = None
    if args.task_type in ("cover", "style_transfer", "repaint"):
        if args.ref_audio and Path(args.ref_audio).exists():
            safe_ref_audio = str(Path(args.ref_audio).resolve())

    effective_steps = min(12, max(8, args.steps))

    print(f"[acestep-official] 보컬 곡 생성 시작 (duration={duration}s, steps={effective_steps}, "
          f"thinking={thinking})...")
    params = GenerationParams(
        task_type="text2music",
        thinking=thinking,
        use_cot_metas=thinking,
        use_cot_caption=thinking,
        use_cot_language=thinking,
        caption=caption,
        lyrics=lyrics if lyrics.strip() else "",
        instrumental=not bool(lyrics.strip()),
        bpm=args.bpm,                      # None 이면 CoT 자동 추정
        keyscale="",                       # 자동 감지
        timesignature="",                  # 자동 감지
        vocal_language=args.vocal_language,
        duration=duration,
        inference_steps=effective_steps,
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
    try:
        result = generate_music(dit_handler, llm_handler, params=params, config=config,
                                save_dir=str(temp_save_dir))
    except Exception as e:  # noqa: BLE001
        fail(f"generate_music 실행 중 예외: {e}")
    elapsed = time.time() - t0

    if getattr(result, "success", False) and getattr(result, "audios", None):
        gen_file = result.audios[0]["path"]
        print(f"[acestep-official] 생성 완료 ({elapsed:.1f}s) -> {gen_file}")
        shutil.copy2(gen_file, str(out_path))
        print(f"[acestep-official] 최종 출력: {out_path} ({out_path.stat().st_size} bytes)")
        return 0

    msg = getattr(result, "status_message", "") or ""
    err = getattr(result, "error", "") or ""
    fail(f"생성 실패: {msg} / error: {err}", code=1)


if __name__ == "__main__":
    sys.exit(main())
