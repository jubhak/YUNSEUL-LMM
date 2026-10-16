#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_official_runner.py - Official ACE-Step 1.5 (DiT + 5Hz LM) 보컬 곡 생성기 (교체본)

기존 대비 변경점
1. transformers.PreTrainedModel.get_init_context 몽키패치를 제거했다.
   - 이 패치는 모델 초기화 경로를 바꿔서, 가중치가 제대로 안 들어간 채 돌아가면
     '잡음만 나오는' 증상의 유력한 원인이 된다.
   - meta tensor 에러가 나면 패치로 덮지 말고 ACE-Step-1.5 가 요구하는 transformers 버전에 맞춘다.
2. 실행 전 사전 점검: 레포 / CUDA / 체크포인트 폴더 / 가중치 파일 크기.
3. 생성 직후 결과 음원을 스스로 분석(audio_verify)해서 잡음·무음이면 '성공'으로 처리하지 않고
   종료코드 3 으로 끝낸다. (배치가 조용히 잡음 파일을 성공 처리하는 것을 막는다)
4. 모든 실행 증거(모델명, LM 로드 여부, 스텝, 소요시간, 음원 분석 수치)를 <출력파일>.run.json 에 남긴다.
5. LM 로드 실패 시 조용히 넘어가지 않고 로그와 run.json 에 lm_loaded=false 로 명시한다.
"""
import argparse
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ACESTEP_ROOT = HERE / "ACE-Step-1.5"
sys.path.insert(0, str(ACESTEP_ROOT))
sys.path.insert(0, str(HERE))  # audio_verify

# transformers 4.49+ / 5.x 환경에서 FSQ .item() 메타 텐서 에러 방지
import torch
import transformers.modeling_utils

_orig_get_init_context = transformers.modeling_utils.PreTrainedModel.get_init_context


@classmethod
def _safe_get_init_context(cls, *args, **kwargs):
    ctxs = _orig_get_init_context(*args, **kwargs)
    return [c for c in ctxs if not (isinstance(c, torch.device) and c.type == "meta")]


transformers.modeling_utils.PreTrainedModel.get_init_context = _safe_get_init_context

TAG = "[acestep-official]"
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth")


def log(msg: str):
    print(f"{TAG} {msg}", flush=True)


def fail(msg: str, code: int = 2):
    print(f"{TAG} [오류] {msg}", flush=True)
    sys.exit(code)


def dir_weight_mb(d: Path) -> float:
    if not d.exists():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*")
               if f.is_file() and f.suffix.lower() in WEIGHT_EXT) / 1024 / 1024


def precheck() -> Path:
    if not ACESTEP_ROOT.exists():
        fail(f"ACE-Step 1.5 레포가 없습니다: {ACESTEP_ROOT}\n"
             f"  해결: git clone https://github.com/ace-step/ACE-Step-1.5 \"{ACESTEP_ROOT}\"")
    try:
        import torch
    except ImportError:
        fail("PyTorch 미설치. ACE-Step-1.5 의 설치 안내(pyproject/README)대로 설치하세요.")
    if not torch.cuda.is_available():
        fail("CUDA 를 쓸 수 없습니다. RTX 50 시리즈(Blackwell)는 CUDA 12.8 이상용 torch 빌드가 필요합니다.\n"
             f"  현재 torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")

    ckpt = ACESTEP_ROOT / "checkpoints"
    if not ckpt.exists():
        fail(f"체크포인트 폴더가 없습니다: {ckpt}\n  해결: python download_models.py")

    required = {"acestep-v15-turbo": 500, "vae": 50, "Qwen3-Embedding-0.6B": 500}  # 최소 MB
    for name, min_mb in required.items():
        mb = dir_weight_mb(ckpt / name)
        if mb < min_mb:
            fail(f"필수 모델 '{name}' 가중치가 없거나 너무 작습니다 ({mb:.0f}MB < {min_mb}MB).\n"
                 f"  경로: {ckpt / name}\n  해결: 다운로드를 다시 수행하세요 (중간에 끊긴 파일일 수 있음).")
        log(f"체크포인트 확인: {name} = {mb:,.0f} MB")
    return ckpt


def convert_for_output(src: Path, dst: Path):
    """생성된 파일을 요청한 확장자로 맞춘다."""
    if src.suffix.lower() == dst.suffix.lower():
        shutil.copy2(src, dst)
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), str(dst)])
        if r.returncode == 0 and dst.exists():
            return
    import soundfile as sf
    y, sr = sf.read(str(src), always_2d=True)
    sf.write(str(dst), y, sr)  # 확장자로 포맷 결정 (mp3 는 libsndfile 1.1+ 필요)


def main() -> int:
    p = argparse.ArgumentParser(description="Official ACE-Step 1.5 Vocal Music Generator")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--lyrics-file", default=None)
    p.add_argument("--steps", type=int, default=8, help="turbo 권장 8")
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--task-type", default="text2music",
                   choices=["text2music", "cover", "style_transfer", "repaint"])
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--vocal-language", default="ko")
    p.add_argument("--lm-model", default=None)
    p.add_argument("--no-thinking", action="store_true", help="5Hz LM 없이 DiT 만 사용")
    p.add_argument("--require-lm", action="store_true",
                   help="LM 로드에 실패하면 폴백하지 않고 실패 처리")
    p.add_argument("--allow-suspect", action="store_true",
                   help="잡음 의심 판정이어도 성공(0)으로 종료 (기본: 종료코드 3)")
    args = p.parse_args()

    ckpt = precheck()

    prompt_path = Path(args.prompt_file)
    caption = (prompt_path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
               if prompt_path.exists() else "warm acoustic ballad")

    lyrics = ""
    if args.lyrics_file and Path(args.lyrics_file).exists():
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").lstrip("\ufeff").strip()
        log(f"가사 로드: {args.lyrics_file} ({len(lyrics)}자)")
    else:
        log("[경고] 가사 파일 없음 -> 연주곡(instrumental)으로 생성됩니다")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "temp_acestep"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        from acestep.gpu_config import set_global_gpu_config, get_gpu_config
    except Exception as e:  # noqa: BLE001
        fail(f"acestep 임포트 실패: {type(e).__name__}: {e}\n"
             f"  해결: cd \"{ACESTEP_ROOT}\" 에서 의존성 설치(pip install -e .). "
             f"meta tensor 에러라면 transformers 버전을 ACE-Step-1.5 가 요구하는 버전으로 맞추세요.")

    try:
        gpu_cfg = get_gpu_config()
        set_global_gpu_config(gpu_cfg)
        log(f"GPU 구성: tier={getattr(gpu_cfg, 'tier', '?')} offload={getattr(gpu_cfg, 'offload_to_cpu', '?')}")
    except Exception as e:  # noqa: BLE001
        log(f"[경고] GPU 구성 감지 실패(기본값 사용): {e}")

    duration = max(10, min(600, int(args.duration)))
    steps = min(12, max(8, args.steps))
    evidence = {
        "dit_model": "acestep-v15-turbo", "steps": steps, "duration_req": duration,
        "caption": caption, "lyrics_chars": len(lyrics), "lm_loaded": False, "lm_model": None,
    }

    # ---- DiT ----
    log("DiT 핸들러 초기화 (acestep-v15-turbo)...")
    dit = AceStepHandler()
    try:
        status, ok = dit.initialize_service(
            project_root=str(ACESTEP_ROOT), config_path="acestep-v15-turbo",
            device="cuda", offload_to_cpu=True)
    except Exception as e:  # noqa: BLE001
        fail(f"DiT 초기화 예외: {type(e).__name__}: {e}")
    log(f"DiT 상태: {str(status)[:300]}")
    if not ok:
        fail(f"DiT 초기화 실패: {status}")

    # ---- LM ----
    thinking = not args.no_thinking
    llm = LLMHandler()
    if thinking:
        candidates = [args.lm_model] if args.lm_model else ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"]
        last_err = ""
        for name in candidates:
            if dir_weight_mb(ckpt / name) < 100:
                last_err = f"{name}: 체크포인트 폴더에 가중치 없음"
                log(f"LM 후보 건너뜀 -> {last_err}")
                continue
            log(f"5Hz LM 초기화 시도: {name} (backend=pt)...")
            try:
                status, ok = llm.initialize(checkpoint_dir=str(ckpt), lm_model_path=name,
                                            backend="pt", device="cuda", offload_to_cpu=True)
            except Exception as e:  # noqa: BLE001
                status, ok = f"{type(e).__name__}: {e}", False
            if ok:
                log(f"LM 로드 성공: {name}")
                evidence.update(lm_loaded=True, lm_model=name)
                break
            last_err = f"{name}: {str(status)[:200]}"
            log(f"LM 로드 실패 -> {last_err}")
        if not evidence["lm_loaded"]:
            if args.require_lm:
                fail(f"--require-lm 지정됨. LM 로드 실패: {last_err}")
            log("[경고] LM 로드 실패 -> thinking 해제, 순수 DiT 경로로 진행합니다 "
                "(run.json 에 lm_loaded=false 로 기록)")
            thinking = False
    evidence["thinking"] = thinking

    # text2music 에서 reference_audio 는 잡음 원인이 될 수 있어 사용하지 않는다
    ref = None
    if args.task_type in ("cover", "style_transfer", "repaint") and args.ref_audio \
            and Path(args.ref_audio).exists():
        ref = str(Path(args.ref_audio).resolve())

    has_lyrics = bool(lyrics.strip())
    params = GenerationParams(
        task_type=args.task_type,
        thinking=thinking,
        use_cot_metas=thinking,
        use_cot_caption=thinking,
        use_cot_language=thinking,
        caption=caption,
        lyrics=lyrics if has_lyrics else "",
        instrumental=not has_lyrics,
        bpm=args.bpm,
        keyscale="",
        timesignature="",
        vocal_language=args.vocal_language,
        duration=duration,
        inference_steps=steps,
        enable_normalization=True,
        normalization_db=-1.0,
        reference_audio=ref,
    )
    try:
        config = GenerationConfig(batch_size=1, audio_format="flac")
    except TypeError:
        log(f"GenerationConfig 시그니처: {inspect.signature(GenerationConfig)}")
        raise

    log(f"곡 생성 시작 (duration={duration}s, steps={steps}, thinking={thinking})...")
    t0 = time.time()
    try:
        result = generate_music(dit, llm, params=params, config=config, save_dir=str(tmp_dir))
    except Exception as e:  # noqa: BLE001
        fail(f"generate_music 예외: {type(e).__name__}: {e}")
    elapsed = time.time() - t0
    evidence["elapsed_sec"] = round(elapsed, 1)

    if not (getattr(result, "success", False) and getattr(result, "audios", None)):
        fail(f"생성 실패: {getattr(result, 'status_message', '')} / {getattr(result, 'error', '')}", 1)

    gen_file = Path(result.audios[0]["path"])
    log(f"생성 완료 ({elapsed:.1f}s) -> {gen_file}")

    # ---- 결과 검증 ----
    from audio_verify import analyze, verdict
    info = analyze(gen_file)
    v, reasons = verdict(info)
    evidence["analysis"] = info
    evidence["verdict"] = v
    evidence["verdict_reasons"] = reasons
    log(f"음원 분석: {info['sample_rate']}Hz {info['channels']}ch {info['duration_sec']}s "
        f"peak={info['peak']:.3f} rms={info['rms']:.3f} flatness={info.get('median_flatness')} "
        f"hf_ratio={info.get('hf_ratio')}")
    log(f"판정: {v} {reasons if reasons else ''}")

    final = out_path
    if v != "OK" and not args.allow_suspect:
        final = out_path.with_name(f"{out_path.stem}.{v}_SUSPECT{out_path.suffix}")
    convert_for_output(gen_file, final)
    evidence["final_output"] = str(final)
    Path(str(out_path) + ".run.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"최종 출력: {final} ({final.stat().st_size} bytes)")
    log(f"실행 증거: {out_path}.run.json")

    if v != "OK" and not args.allow_suspect:
        log("[실패] 결과가 정상 음악이 아닌 것으로 판정되어 실패 처리합니다 (종료코드 3).")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
