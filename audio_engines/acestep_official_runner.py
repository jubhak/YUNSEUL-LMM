#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_official_runner.py - Official ACE-Step 1.5 (DiT + 5Hz LM) 보컬 곡 생성기

- 학습된 보컬/스타일 LoRA(또는 LoKr)를 DiT에 로드해 생성에 반영 (--lora-path, --lora-scale)
  로드에 실패하면 기본 보컬로 몰래 진행하지 않고 종료코드 4로 실패한다.
- 공식 turbo 권장값: 8 steps, shift 3.0
- 후보 여러 개(--candidates)를 한 번에 생성 -> 잡음 판정 + DiT 가사 정렬 점수로 최선의 테이크 선택
- 최종 출력은 -14 LUFS 라우드니스 마스터링 후 320kbps(mp3) 저장, 다른 테이크도 _takeN 으로 보존
- 실행 증거(모델, LoRA, LM, 점수, 분석 수치)는 <출력파일>.run.json 에 기록
"""
import argparse
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # audio_verify

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
    log(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)}")
    try:
        import transformers
        log(f"transformers={transformers.__version__}")
    except ImportError:
        pass

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


def export_master(src: Path, dst: Path, master: bool = True):
    """생성 파일을 요청한 확장자로 저장. ffmpeg 가 있으면 라우드니스 마스터링을 함께 적용."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src)]
        if master:
            cmd += ["-af", "loudnorm=I=-14:TP=-1.0:LRA=11", "-ar", "48000"]
        if dst.suffix.lower() == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
        cmd.append(str(dst))
        r = subprocess.run(cmd)
        if r.returncode == 0 and dst.exists():
            return
        log("[경고] ffmpeg 변환 실패 -> soundfile 로 저장 (마스터링 생략)")
    if src.suffix.lower() == dst.suffix.lower():
        shutil.copy2(src, dst)
        return
    import soundfile as sf
    y, sr = sf.read(str(src), always_2d=True)
    sf.write(str(dst), y, sr)


def load_adapter(dit, path: str, scale: float, evidence: dict, name: str = "voice"):
    p = Path(path)
    # 학습 출력 폴더를 넘겨도 final 어댑터를 찾아 쓴다
    if p.is_dir() and not (p / "adapter_config.json").exists() and not list(p.glob("*.safetensors")):
        if (p / "final").exists():
            p = p / "final"
    if not p.exists():
        fail(f"LoRA 경로가 없습니다: {p}", 4)
    log(f"학습된 어댑터 로드: {p} (이름 {name}, 반영 강도 {scale:.2f})")
    msg = dit.add_lora(str(p), adapter_name=name)
    log(f"  -> {msg}")
    if not str(msg).startswith("✅"):
        fail(f"학습된 보컬 어댑터 로드 실패: {msg}\n"
             f"  기본 보컬로 몰래 진행하지 않고 중단합니다. 모델을 다시 학습하거나 '기본 보컬'을 선택하세요.", 4)
    smsg = dit.set_lora_scale(name, float(scale))
    log(f"  -> {smsg}")
    evidence.update({f"lora_{name}_path": str(p), f"lora_{name}_scale": float(scale),
                     f"lora_{name}_loaded": True})
    if name == "voice":
        evidence.update(lora_path=str(p), lora_scale=float(scale), lora_loaded=True)


def lyric_alignment_scores(dit, extra: dict, n: int, language: str, steps: int) -> list:
    """ACE-Step 공식 DiT 가사 정렬 점수 (높을수록 가사가 음성에 정확히 실림)."""
    keys = {
        "pred_latent": extra.get("pred_latents"),
        "encoder_hidden_states": extra.get("encoder_hidden_states"),
        "encoder_attention_mask": extra.get("encoder_attention_mask"),
        "context_latents": extra.get("context_latents"),
        "lyric_token_ids": extra.get("lyric_token_idss", extra.get("lyric_token_ids")),
    }
    if any(v is None for v in keys.values()):
        return [None] * n
    scores = []
    for i in range(n):
        try:
            res = dit.get_lyric_score(
                **{k: v[i:i + 1] for k, v in keys.items()},
                vocal_language=language, inference_steps=steps, seed=42)
            scores.append(float(res.get("dit_score")) if res.get("success") else None)
        except Exception as e:  # noqa: BLE001
            log(f"[경고] 가사 정렬 점수 계산 실패 (take {i + 1}): {e}")
            scores.append(None)
    return scores


def main() -> int:
    p = argparse.ArgumentParser(description="Official ACE-Step 1.5 Vocal Music Generator")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=int, default=0, help="0 이하면 가사 길이에 맞춰 LM 이 결정")
    p.add_argument("--lyrics-file", default=None)
    p.add_argument("--steps", type=int, default=8, help="turbo 권장 8")
    p.add_argument("--shift", type=float, default=3.0, help="turbo 공식 권장 3.0")
    p.add_argument("--ref-audio", default=None)
    p.add_argument("--task-type", default="text2music",
                   choices=["text2music", "cover", "style_transfer", "repaint"])
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--keyscale", default="")
    p.add_argument("--timesignature", default="")
    p.add_argument("--vocal-language", default="ko")
    p.add_argument("--lora-path", default=None, help="학습된 보컬(내 목소리) 어댑터 폴더")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--style-lora-path", default=None,
                   help="곡 학습 모델(1000곡 코퍼스) 어댑터 폴더 - 목소리 LoRA와 별개로 함께 적용")
    p.add_argument("--style-lora-scale", type=float, default=0.8)
    p.add_argument("--candidates", type=int, default=2, help="한 번에 생성해 고를 테이크 수 (1~4)")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--lm-model", default=None)
    p.add_argument("--lm-temperature", type=float, default=0.85)
    p.add_argument("--no-thinking", action="store_true", help="5Hz LM 없이 DiT 만 사용")
    p.add_argument("--require-lm", action="store_true",
                   help="LM 로드에 실패하면 폴백하지 않고 실패 처리")
    p.add_argument("--no-master", action="store_true", help="라우드니스 마스터링 생략")
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
             f"  해결: audio_engines\\venv_acestep 의 Python 으로 실행해야 합니다 (transformers 4.57.x).")

    try:
        gpu_cfg = get_gpu_config()
        set_global_gpu_config(gpu_cfg)
        log(f"GPU 구성: tier={getattr(gpu_cfg, 'tier', '?')} offload={getattr(gpu_cfg, 'offload_to_cpu', '?')}")
    except Exception as e:  # noqa: BLE001
        log(f"[경고] GPU 구성 감지 실패(기본값 사용): {e}")

    duration = -1.0 if args.duration <= 0 else float(max(10, min(600, args.duration)))
    steps = min(20, max(4, args.steps))
    candidates = min(4, max(1, args.candidates))
    evidence = {
        "dit_model": "acestep-v15-turbo", "steps": steps, "shift": args.shift,
        "duration_req": duration, "caption": caption, "lyrics_chars": len(lyrics),
        "bpm": args.bpm, "keyscale": args.keyscale, "candidates": candidates,
        "continuity_screen": "mix_energy_only_not_speaker_verification",
        "lm_loaded": False, "lm_model": None, "lora_loaded": False,
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

    if args.lora_path:
        load_adapter(dit, args.lora_path, args.lora_scale, evidence, name="voice")
    if args.style_lora_path:
        # 곡 코퍼스 모델은 '창법/스타일' 어댑터로 따로 붙인다 (내 목소리 LoRA와 충돌하지 않음)
        load_adapter(dit, args.style_lora_path, args.style_lora_scale, evidence, name="style")

    # ---- LM ----
    thinking = not args.no_thinking
    llm = LLMHandler()
    if thinking:
        candidates_lm = [args.lm_model] if args.lm_model else ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B"]
        last_err = ""
        for name in candidates_lm:
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
    if has_lyrics:
        caption += (", one consistent solo lead singer throughout, stable vocal timbre, "
                    "smooth connected register transitions, preserve melodic contour, "
                    "continuous accompaniment, balanced vocal level, no abrupt octave drops, no singer switching")
    evidence["caption"] = caption  # actual prompt including continuity constraints
    params = GenerationParams(
        task_type=args.task_type,
        thinking=thinking,
        use_cot_metas=thinking,
        # 학습 어댑터의 트리거 태그가 LM 캡션 재작성으로 사라지지 않게 한다
        use_cot_caption=False,  # keep identity and continuity instructions verbatim
        use_cot_language=False,
        caption=caption,
        lyrics=lyrics if has_lyrics else "",
        instrumental=not has_lyrics,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        vocal_language=args.vocal_language,
        duration=duration,
        inference_steps=steps,
        shift=args.shift,
        seed=args.seed,
        lm_temperature=args.lm_temperature,
        enable_normalization=True,
        normalization_db=-1.0,
        fade_out_duration=1.5,
        reference_audio=ref,
    )
    config = GenerationConfig(batch_size=candidates, audio_format="flac",
                              use_random_seed=args.seed < 0,
                              seeds=None if args.seed < 0 else [args.seed + i for i in range(candidates)])

    log(f"곡 생성 시작 (duration={'auto' if duration < 0 else int(duration)}, steps={steps}, "
        f"shift={args.shift}, takes={candidates}, thinking={thinking}, "
        f"lora={'on' if evidence['lora_loaded'] else 'off'})...")
    t0 = time.time()
    try:
        result = generate_music(dit, llm, params=params, config=config, save_dir=str(tmp_dir))
    except Exception as e:  # noqa: BLE001
        fail(f"generate_music 예외: {type(e).__name__}: {e}")
    elapsed = time.time() - t0
    evidence["elapsed_sec"] = round(elapsed, 1)

    audios = getattr(result, "audios", None) or []
    if not (getattr(result, "success", False) and audios):
        fail(f"생성 실패: {getattr(result, 'status_message', '')} / {getattr(result, 'error', '')}", 1)
    log(f"생성 완료 ({elapsed:.1f}s, {len(audios)}개 테이크)")

    # ---- 테이크 검증 및 선택 ----
    from audio_verify import analyze, verdict
    align = lyric_alignment_scores(dit, getattr(result, "extra_outputs", {}) or {}, len(audios),
                                   args.vocal_language, steps) if has_lyrics else [None] * len(audios)
    takes = []
    for i, a in enumerate(audios):
        f = Path(a.get("path") or "")
        if not f.exists():
            continue
        info = analyze(f)
        v, reasons = verdict(info)
        try:
            from vocal_pipeline import continuity_metrics
            info["continuity"] = continuity_metrics(f)
        except Exception as exc:
            info["continuity"] = {"penalty": 0, "unavailable": str(exc)[:200]}
        takes.append({"idx": i, "path": f, "verdict": v, "reasons": reasons, "analysis": info,
                      "align": align[i] if i < len(align) else None, "seed": a.get("params", {}).get("seed")})
        log(f"  take {i + 1}: 판정={v} 가사정렬={takes[-1]['align']} rms={info['rms']:.3f} "
            f"flatness={info.get('median_flatness')}")
    if not takes:
        fail("생성된 음원 파일을 찾지 못했습니다.", 1)

    takes.sort(key=lambda t: (t["verdict"] == "OK", -t["analysis"].get("continuity", {}).get("penalty", 0), t["align"] if t["align"] is not None else -1.0),
               reverse=True)
    best = takes[0]
    v = best["verdict"]
    evidence.update(
        analysis=best["analysis"], verdict=v, verdict_reasons=best["reasons"],
        best_take=best["idx"] + 1, best_seed=best["seed"],
        lm_metadata=(getattr(result, "extra_outputs", {}) or {}).get("lm_metadata"),
        takes=[{"take": t["idx"] + 1, "verdict": t["verdict"], "align": t["align"], "seed": t["seed"], "continuity": t["analysis"].get("continuity")}
               for t in takes],
    )
    log(f"최종 선택: take {best['idx'] + 1} (판정 {v}, 가사정렬 {best['align']})")

    final = out_path
    if v != "OK" and not args.allow_suspect:
        final = out_path.with_name(f"{out_path.stem}.{v}_SUSPECT{out_path.suffix}")
    export_master(best["path"], final, master=not args.no_master)
    for t in takes[1:]:
        export_master(t["path"], out_path.with_name(f"{out_path.stem}_take{t['idx'] + 1}{out_path.suffix}"),
                      master=not args.no_master)
    evidence["final_output"] = str(final)
    Path(str(out_path) + ".run.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"최종 출력: {final} ({final.stat().st_size} bytes)")
    log(f"실행 증거: {out_path}.run.json")

    if v != "OK" and not args.allow_suspect:
        log("[실패] 결과가 정상 음악이 아닌 것으로 판정되어 실패 처리합니다 (종료코드 3).")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
