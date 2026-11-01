#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_lora_trainer.py - ACE-Step 1.5 보컬/스타일 LoRA 실제 학습 (venv_acestep 전용)

1) 전처리: dataset.json(음원 + 가사 + 캡션 + BPM/키) -> VAE/텍스트 인코딩 텐서 (.pt, 재실행 시 이어서)
2) 학습: Side-Step 'fixed' 트레이너 (연속 타임스텝 + CFG dropout) 로 turbo DiT 에 LoRA 학습
3) 출력: <output-dir>/final/adapter_config.json + adapter_model.safetensors -> 생성 러너가 로드

stdout 프로토콜 (vocal_trainer_runner.py 가 위젯으로 중계):
  [LOG] 메시지 / [PROGRESS:단계:퍼센트:제목] / [ADAPTER:경로] / [FINAL_LOSS:값]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ACESTEP_ROOT = HERE / "ACE-Step-1.5"
CKPT = ACESTEP_ROOT / "checkpoints"
sys.path.insert(0, str(ACESTEP_ROOT))
os.chdir(str(ACESTEP_ROOT))  # sidestep.log 등 상대 경로 산출물을 레포 안에 둔다


def log(msg):
    print(f"[LOG] {msg}", flush=True)


def progress(stage, pct, title):
    print(f"[PROGRESS:{stage}:{int(pct)}:{title}]", flush=True)


def dataset_diversity_report(dataset_json: Path, n_tensors: int) -> dict:
    """학습 데이터가 '한 곡 반복'이면 LoRA 가 단조로운 평균만 배운다. 그 위험을 수치로 남긴다."""
    report = {"tensors": n_tensors, "samples": 0, "distinct_captions": 0,
              "distinct_keys": 0, "distinct_bpm": 0, "avg_duration_sec": None, "warnings": []}
    try:
        data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report["warnings"].append(f"dataset.json 을 읽지 못했습니다: {exc}")
        return report
    samples = data.get("samples") or []
    report["samples"] = len(samples)
    captions = {(s.get("caption") or "").strip() for s in samples}
    keys = {(s.get("keyscale") or "").strip() for s in samples}
    bpms = {s.get("bpm") for s in samples}
    durations = [float(s.get("duration") or 0) for s in samples if s.get("duration")]
    report.update(distinct_captions=len(captions), distinct_keys=len(keys), distinct_bpm=len(bpms))
    if durations:
        report["avg_duration_sec"] = round(sum(durations) / len(durations), 1)
    if samples and len(captions) == 1:
        report["warnings"].append(
            "모든 학습 곡의 caption 이 동일합니다. LoRA 가 편곡 차이를 배우지 못해 결과가 단조로워집니다. "
            "training_genres.caption_variant() 로 곡마다 다른 전개 서술을 넣으세요.")
    if samples and len(bpms) <= 1:
        report["warnings"].append("학습 곡의 BPM 이 한 값뿐입니다. 템포/그루브 다양성이 학습되지 않습니다.")
    if durations and sum(durations) / len(durations) < 30:
        report["warnings"].append(
            "평균 학습 길이가 30초 미만입니다. 한 구간만 배우면 곡 전체가 같은 마디를 반복합니다 "
            "(max-duration 을 늘리거나 곡당 여러 구간을 넣으세요).")
    return report


def install_audio_loader():
    """torchaudio 2.9+ 의 load 는 TorchCodec(+FFmpeg DLL) 이 필요하다. soundfile, 실패 시 ffmpeg 로 대체."""
    import shutil
    import subprocess
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *args, **kwargs):
        path = str(path)
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise
            raw = subprocess.run(
                [ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", "48000", "-"],
                capture_output=True, check=True,
            ).stdout
            data, sr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2), 48000
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr

    torchaudio.load = _load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--tensor-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-duration", type=float, default=150.0)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--variant", default="turbo")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 를 사용할 수 없습니다 (venv_acestep 의 torch cu128 빌드 확인).", flush=True)
        return 1
    log(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------- 전처리
    install_audio_loader()
    from acestep.training_v2.preprocess import preprocess_audio_files

    progress(3, 40, "[3/5] 음원을 학습 텐서로 인코딩 중 (VAE + 텍스트 인코더)...")
    log("▶ [3단계: 전처리] 음원 → 잠재 표현, 가사/캡션 → 임베딩 (이미 처리된 곡은 건너뜀)")

    def on_pre(cur, total, msg):
        pass_no = 0 if "Pass 1" in msg else 1
        pct = 40 + int(((cur + 1) / max(1, total) * 0.5 + pass_no * 0.5) * 15)
        progress(3, min(55, pct), f"[3/5] 전처리 {msg} ({cur + 1}/{total})")

    try:
        res = preprocess_audio_files(
            audio_dir=None, output_dir=args.tensor_dir, checkpoint_dir=str(CKPT),
            variant=args.variant, max_duration=args.max_duration,
            dataset_json=args.dataset_json, progress_callback=on_pre,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] 전처리 실패: {type(e).__name__}: {e}", flush=True)
        return 1
    finally:
        torch.cuda.empty_cache()
    n_pt = len([p for p in Path(args.tensor_dir).glob("*.pt") if not p.name.endswith(".tmp.pt")])
    log(f"  ✓ 전처리 완료: 신규 {res.get('processed', 0)}곡, 실패 {res.get('failed', 0)}곡, 학습 텐서 총 {n_pt}개")

    # 단조로움의 뿌리는 대개 '데이터'다. 학습 전에 진단을 남긴다.
    report = dataset_diversity_report(Path(args.dataset_json), n_pt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  • 데이터 다양성: caption {report['distinct_captions']}종 / BPM {report['distinct_bpm']}종 "
        f"/ 평균 길이 {report['avg_duration_sec']}초")
    for w in report["warnings"]:
        log(f"  ⚠️ {w}")
    if n_pt == 0:
        print("[ERROR] 학습 가능한 텐서가 없습니다.", flush=True)
        return 1

    # ---------------------------------------------------------------- 학습
    from acestep.training.path_safety import set_safe_root
    try:
        safe_root = os.path.commonpath([str(ACESTEP_ROOT), os.path.abspath(args.tensor_dir),
                                        os.path.abspath(args.output_dir)])
    except ValueError:  # 서로 다른 드라이브
        safe_root = Path(os.path.abspath(args.tensor_dir)).anchor
    set_safe_root(safe_root)

    from acestep.training_v2.cli.args import build_root_parser
    from acestep.training_v2.cli.config_builder import build_configs
    from acestep.training_v2.cli.validation import validate_paths
    from acestep.training_v2.model_loader import load_decoder_for_training
    from acestep.training_v2.trainer_fixed import FixedLoRATrainer

    cli = [
        "--plain", "--yes", "fixed",
        "--checkpoint-dir", str(CKPT), "--model-variant", args.variant,
        "--dataset-dir", args.tensor_dir, "--output-dir", args.output_dir,
        "--adapter-type", "lora", "--rank", str(args.rank), "--alpha", str(args.rank * 2),
        "--lr", str(args.lr), "--epochs", str(args.epochs), "--batch-size", "1",
        "--gradient-accumulation", "4", "--save-every", str(args.save_every),
        "--offload-encoder", "--gradient-checkpointing",
        "--shift", "3.0", "--num-inference-steps", "8",
    ]
    targs = build_root_parser().parse_args(cli)
    if not validate_paths(targs):
        print("[ERROR] 학습 경로 검증 실패", flush=True)
        return 1
    adapter_cfg, train_cfg = build_configs(targs)

    log(f"▶ [4단계: LoRA 학습] rank={args.rank}, lr={args.lr}, epochs={args.epochs}, 곡 수={n_pt}")
    progress(4, 56, "[4/5] ACE-Step DiT 로드 중...")
    model = load_decoder_for_training(
        checkpoint_dir=train_cfg.checkpoint_dir, variant=train_cfg.model_variant,
        device=train_cfg.device, precision=train_cfg.precision,
    )
    trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)

    t0 = time.time()
    last_loss = None
    failed = None
    for upd in trainer.train():
        kind = getattr(upd, "kind", "info")
        msg = getattr(upd, "msg", "")
        if kind == "fail":
            failed = msg
            log(f"❌ {msg}")
            continue
        if kind in ("step", "epoch"):
            ep, max_ep = getattr(upd, "epoch", 0) or 0, getattr(upd, "max_epochs", 0) or args.epochs
            last_loss = getattr(upd, "loss", last_loss)
            frac = ep / max(1, max_ep)
            elapsed = time.time() - t0
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            progress(4, 56 + frac * 40,
                     f"[4/5] LoRA 학습 Epoch {ep}/{max_ep} | loss {last_loss:.4f} | 남은 시간 약 {int(eta // 60)}분")
            if kind == "epoch":
                log(f"  • {msg}")
        elif kind in ("checkpoint", "complete", "warn", "info"):
            log(f"  {msg}")

    if failed:
        print(f"[ERROR] 학습 실패: {failed}", flush=True)
        return 1

    final_dir = Path(args.output_dir) / "final"
    if not (final_dir / "adapter_config.json").exists():
        print(f"[ERROR] 학습은 끝났지만 어댑터 파일이 없습니다: {final_dir}", flush=True)
        return 1
    size_mb = sum(f.stat().st_size for f in final_dir.glob("*.safetensors")) / 1024 / 1024
    log(f"  ✓ LoRA 어댑터 저장: {final_dir} ({size_mb:.1f} MB, {int((time.time() - t0) // 60)}분 소요)")
    print(f"[ADAPTER:{final_dir}]", flush=True)
    if last_loss is not None:
        print(f"[FINAL_LOSS:{last_loss:.5f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
