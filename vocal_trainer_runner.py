#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vocal_trainer_runner.py - ACE-Step 1.5 보컬/스타일 LoRA 실제 학습 파이프라인
Stage 1: 장르 선별 + 중복 검사 (이미 분석한 곡은 캐시 재사용)
Stage 2: 곡별 학습 라벨 생성 - 가사(옆 가사 파일 우선, 없으면 Whisper 받아쓰기), BPM/키, 캡션
Stage 3: ACE-Step 전처리 (VAE 잠재 표현 + 가사/캡션 임베딩)
Stage 4: turbo DiT 에 LoRA 학습 (venv_acestep 의 Side-Step 트레이너)
Stage 5: 모델 등록 (meta.json 에 adapter_path / trigger_tag 기록 -> 노래 생성 시 자동 적용)

원곡 전체를 가사와 함께 학습해야 ACE-Step 이 '이 가수의 음색/창법'을 가사 정렬과 함께 배운다.
(이전 버전은 HPSS 분리 + 슬라이스만 만들고 실제 학습 없이 ref_vocal.wav 만 남겨, 생성에 반영되지 않았다.)
"""

import sys
import os

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import json
import time
import re
import hashlib
import gc
import argparse
import subprocess
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
VENV_PY = BASE_DIR / "audio_engines" / "venv_acestep" / "Scripts" / "python.exe"
TRAINER = BASE_DIR / "audio_engines" / "acestep_lora_trainer.py"

MAX_TRAIN_SEC = 150.0      # 16GB VRAM 에서 안정적인 곡당 학습 길이 (앞부분 150초 = 1절 + 후렴 포함)
WHISPER_MODEL = "large-v3-turbo"


def log(msg):
    print(f"[LOG] {msg}", flush=True)


def progress(stage_num, percent, title):
    print(f"[PROGRESS:{stage_num}:{int(percent)}:{title}]", flush=True)


def get_gpu_vram():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return int(info.used // (1024 * 1024)), int(info.total // (1024 * 1024))
    except Exception:
        return 0, 0


def get_audio_signature(path: Path) -> str:
    """음원 파일 크기 + 헤더/푸터 해시로 고유 디지털 지문 생성 (빠르고 정확한 중복 판별)"""
    try:
        st = path.stat()
        hasher = hashlib.md5()
        hasher.update(str(st.st_size).encode())
        with open(path, "rb") as f:
            hasher.update(f.read(65536))
            if st.st_size > 131072:
                f.seek(-65536, os.SEEK_END)
                hasher.update(f.read(65536))
        return f"{st.st_size}_{hasher.hexdigest()[:16]}"
    except Exception:
        return f"{path.name}_{path.stat().st_size}"


def fix_long_path(p: str) -> str:
    """Windows MAX_PATH(260자) 제한 해제를 위한 Extended Path 접두어 자동 처리"""
    try:
        p_abs = os.path.abspath(p)
        if sys.platform == "win32" and not p_abs.startswith("\\\\?\\") and len(p_abs) > 240:
            return "\\\\?\\" + p_abs
        return p_abs
    except Exception:
        return p


def classify_audio_genre(path: Path) -> tuple:
    """
    3중 하이브리드 K-pop & 발라드 장르 분류기
    Returns: (is_target: bool, reason: str, category: str)
    """
    name_and_path = str(path).lower()

    EXCLUDE_KEYWORDS = [
        "trot", "트로트", "뽕짝", "hiphop", "hip-hop", "힙합", "rap", "랩",
        "metal", "메탈", "death metal", "heavy metal", "hard rock",
        "edm", "techno", "house", "trance", "dubstep", "club",
        "instrumental", "inst", "mr", "연주곡", "경음악", "반주",
        "ccm", "찬송가", "gospel", "성가", "동요", "classic", "클래식", "opera", "오페라"
    ]
    for kw in EXCLUDE_KEYWORDS:
        if kw in name_and_path:
            return False, f"비대상 장르 키워드 감지 ('{kw}')", kw

    try:
        import mutagen
        audio = mutagen.File(str(path))
        if audio and hasattr(audio, "tags") and audio.tags:
            tag_text = " ".join([str(v) for v in audio.tags.values()]).lower()
            for kw in EXCLUDE_KEYWORDS:
                if kw in tag_text:
                    return False, f"메타데이터 비대상 장르 감지 ('{kw}')", kw

            TARGET_KEYWORDS = [
                "ballad", "발라드", "kpop", "k-pop", "k pop", "dance", "댄스",
                "pop", "팝", "r&b", "rnb", "알앤비", "가요", "아이돌", "idol",
                "소울", "soul", "acoustic", "어쿠스틱", "ost", "korean"
            ]
            for kw in TARGET_KEYWORDS:
                if kw in tag_text or kw in name_and_path:
                    return True, f"타깃 장르 태그 일치 ('{kw}')", "K-pop/발라드"
    except Exception:
        pass

    TARGET_NAME_KEYWORDS = [
        "ballad", "발라드", "kpop", "k-pop", "dance", "댄스", "pop", "팝",
        "r&b", "rnb", "알앤비", "가요", "아이돌", "ost", "acoustic", "어쿠스틱"
    ]
    for kw in TARGET_NAME_KEYWORDS:
        if kw in name_and_path:
            return True, f"파일명 타깃 키워드 일치 ('{kw}')", "K-pop/발라드"

    try:
        import librosa
        import numpy as np
        y_probe, sr_probe = librosa.load(str(path), sr=22050, duration=6.0, mono=True)
        if len(y_probe) > sr_probe * 2:
            tempo, _ = librosa.beat.beat_track(y=y_probe, sr=sr_probe)
            bpm = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)

            if bpm > 155 or (bpm > 0 and bpm < 45):
                return False, f"발라드/K-pop 템포 범위 초과 (BPM {int(bpm)})", "Extremum"

            y_harm, y_perc = librosa.effects.hpss(y_probe)
            harm_energy = float(np.sum(y_harm ** 2))
            perc_energy = float(np.sum(y_perc ** 2))
            total_energy = harm_energy + perc_energy + 1e-6
            harm_ratio = harm_energy / total_energy

            if harm_ratio >= 0.35 and 50 <= bpm <= 145:
                genre_label = "감성 발라드" if bpm <= 88 else "K-pop 팝/댄스"
                return True, f"음향 특성 일치 (BPM {int(bpm)}, 멜로디 {harm_ratio*100:.0f}%)", genre_label
            elif harm_ratio < 0.25:
                return False, f"비보컬 타악/노이즈 성향 (하모닉 {harm_ratio*100:.0f}%)", "Percussive/Noise"
    except Exception:
        pass

    return True, "기본 K-pop/발라드 대역 통과", "K-pop/발라드"


# =============================================================================
# 곡 분석: 가사 / BPM / 키 / 캡션
# =============================================================================
_LRC_TS = re.compile(r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
_LRC_META = re.compile(r"^\[(ar|ti|al|by|offset|length|re|ve):", re.I)
_TAG_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
_HALLUCINATIONS = [
    "시청해 주셔서", "시청해주셔서", "구독과 좋아요", "구독 좋아요", "좋아요와 구독", "자막 제공",
    "자막 by", "mbc 뉴스", "kbs 뉴스", "sbs 뉴스", "다음 영상에서", "감사합니다.",
    "thank you for watching", "subscribe",
]


def find_sidecar_lyrics(af: Path):
    """음원 옆의 가사 파일(.lyrics.txt / .txt / .lrc) 을 찾아 ACE-Step 가사 형식으로 정리."""
    stem = af.with_suffix("")
    for cand in (Path(str(stem) + ".lyrics.txt"), af.with_suffix(".txt"), af.with_suffix(".lrc")):
        if not cand.exists():
            continue
        try:
            raw = cand.read_text(encoding="utf-8-sig", errors="ignore")
        except Exception:
            continue
        lines = []
        for ln in raw.splitlines():
            if _LRC_META.match(ln.strip()):
                continue
            ln = _LRC_TS.sub("", ln).strip()
            lines.append(ln)
        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(re.sub(r"\s", "", text)) < 20:
            continue
        if not any(_TAG_LINE.match(l) for l in text.splitlines()):
            blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
            text = _label_blocks([[l for l in b.splitlines() if l.strip()] for b in blocks])
        return text, cand.name
    return None, None


def _bigrams(text):
    t = re.sub(r"\s", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _label_blocks(blocks):
    """반복되는 블록은 [Chorus], 나머지는 [Verse N] 으로 구조 태그를 붙인다."""
    grams = [_bigrams(" ".join(b)) for b in blocks]
    is_chorus = []
    for i, g in enumerate(grams):
        rep = False
        for j, h in enumerate(grams):
            if i != j and g and h and len(g & h) / max(1, len(g | h)) >= 0.45:
                rep = True
                break
        is_chorus.append(rep)
    out, verse_no = [], 0
    for b, ch in zip(blocks, is_chorus):
        if ch:
            tag = "[Chorus]"
        else:
            verse_no += 1
            tag = f"[Verse {verse_no}]"
        out.append(tag + "\n" + "\n".join(b))
    return "\n\n".join(out)


class Transcriber:
    """faster-whisper 한국어 가창 받아쓰기 (GPU float16, 실패 시 CPU int8)."""

    def __init__(self):
        self.model = None
        self.failed = False

    def _load(self):
        if self.model is not None or self.failed:
            return
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
            log(f"  • 가사 받아쓰기 엔진: Whisper {WHISPER_MODEL} (GPU)")
        except Exception as e:
            log(f"  ⚠️ Whisper GPU 로드 실패 ({type(e).__name__}) → CPU 모드로 전환 (느림)")
            try:
                self.model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=8)
            except Exception as e2:
                log(f"  ❌ Whisper 로드 실패: {e2} — 가사 없이 스타일만 학습합니다.")
                self.failed = True

    def transcribe(self, path: Path, max_sec: float):
        self._load()
        if self.model is None:
            return None
        segments, _info = self.model.transcribe(
            str(path), language="ko", beam_size=5, vad_filter=False,
            condition_on_previous_text=False, no_speech_threshold=0.6,
            compression_ratio_threshold=2.2, temperature=[0.0, 0.2, 0.4],
        )
        segs = []
        for s in segments:
            if s.start > max_sec:
                break
            text = re.sub(r"[\"“”♪~]+", "", s.text).strip()
            if not text or any(h in text.lower() for h in _HALLUCINATIONS):
                continue
            if segs and text == segs[-1][2]:
                continue
            segs.append((s.start, s.end, text))
        return segs

    def unload(self):
        self.model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def structure_transcript(segs):
    """받아쓰기 구간을 3초 이상 쉼 기준으로 블록화하고 구조 태그를 붙인다."""
    if not segs:
        return ""
    blocks, cur, last_end = [], [], None
    for st, en, text in segs:
        if cur and (st - last_end >= 3.0 or len(cur) >= 6):
            blocks.append(cur)
            cur = []
        cur.append(text.rstrip(".,!?"))
        last_end = en
    if cur:
        blocks.append(cur)
    body = _label_blocks(blocks)
    if segs[0][0] > 8.0:
        body = "[Intro]\n\n" + body
    return body


_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def analyze_music(path: Path, max_sec: float):
    """librosa 로 BPM / 키 / 실제 길이 추정."""
    import librosa
    import numpy as np
    try:
        total = float(librosa.get_duration(path=str(path)))
    except Exception:
        total = 0.0
    y, sr = librosa.load(str(path), sr=22050, mono=True, offset=min(20.0, total * 0.15), duration=90.0)
    bpm = None
    keyscale = None
    if len(y) > sr * 5:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        while bpm and bpm < 60:
            bpm *= 2
        while bpm and bpm > 180:
            bpm /= 2
        bpm = int(round(bpm)) if bpm else None
        chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr).mean(axis=1)
        best = (-2.0, None)
        for i in range(12):
            for prof, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
                r = float(np.corrcoef(chroma, np.roll(prof, i))[0, 1])
                if r > best[0]:
                    best = (r, f"{_NOTES[i]} {mode}")
        keyscale = best[1]
    return bpm, keyscale, total


def build_caption(category: str, bpm, is_instrumental: bool):
    from training_genres import GENRES, caption_for
    if category in GENRES or category == "unknown":
        base = caption_for(category)
        if is_instrumental:
            base = base.replace(", clear lead vocal", "").replace(", lead vocal", "").replace(", expressive lead vocal", "")
            base += ", instrumental"
        return base
    if "발라드" in category and "K-pop" not in category:
        base = "emotional Korean ballad, piano and strings, expressive lead vocal, clear diction"
    elif "댄스" in category:
        base = "Korean K-pop dance pop, punchy drums, synth bass, bright lead vocal"
    else:
        if bpm and bpm < 92:
            base = "Korean pop ballad, warm band arrangement, expressive lead vocal"
        else:
            base = "Korean pop song, polished band arrangement, confident lead vocal"
    if is_instrumental:
        base = base.replace(", expressive lead vocal", "").replace(", bright lead vocal", "")
        base = base.replace(", confident lead vocal", "") + ", instrumental"
    return base


def make_trigger(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", model_name.lower())
    if not slug:
        slug = "sig" + hashlib.md5(model_name.encode("utf-8")).hexdigest()[:6]
    return f"{slug} voice"


def link_into_dataset(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(fix_long_path(str(src)), str(dst))
    except Exception:
        shutil.copy2(fix_long_path(str(src)), str(dst))


def plan_epochs(n_songs: int, preset: str):
    """ACE-Step Side-Step 가이드: 1~10곡 200~500, 10~50곡 100~200, 50곡+ 50~100 epoch."""
    base = 300 if n_songs <= 10 else 150 if n_songs <= 50 else 80
    mult = {"fast": 0.5, "standard": 1.0, "deep": 1.6}.get(preset, 1.0)
    cap = {"fast": 1500, "standard": 4000, "deep": 9000}.get(preset, 4000)
    return max(5, min(int(base * mult), cap // max(1, n_songs)))


def plan_speech_epochs(n_clips: int, preset: str):
    """짧은 낭독 클립용. 클립이 적으면 epoch 를 늘리고, 표준은 대략 15~40분 목표."""
    base = 120 if n_clips < 10 else 90 if n_clips < 20 else 70
    mult = {"fast": 0.5, "standard": 1.0, "deep": 1.6}.get(preset, 1.0)
    return max(20, int(base * mult))


def prepare_speech_clip(src: Path, dst: Path, max_sec: float = 20.0):
    """모노 48k WAV 로 맞추고, 너무 짧은 클립은 앞뒤에 무음을 붙인다."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(src, dst)
        return dst
    af = "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11,apad=pad_dur=1.2"
    r = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-af", af,
         "-t", str(max_sec), "-ar", "48000", "-ac", "1", str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not dst.exists():
        shutil.copy2(src, dst)
    return dst


def train_speech_voice(args) -> int:
    """따라 읽은 문장 녹음으로 음색 LoRA 를 학습하고 별칭으로 등록한다."""
    payload = json.loads(Path(args.clips_json).read_text(encoding="utf-8-sig"))
    clips = payload.get("clips") or []
    gender = (payload.get("gender") or args.gender or "female").lower()
    if gender not in ("female", "male"):
        gender = "female"
    gender_en = "female" if gender == "female" else "male"
    model_name = re.sub(r"[^\w\s-]", "", args.model_name).strip() or "MyVoice"
    model_dir = Path(args.output_dir) / model_name
    dataset_dir = model_dir / "dataset"
    tensor_dir = model_dir / "tensors"
    lora_dir = model_dir / "lora"
    for d in (dataset_dir, tensor_dir, lora_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not VENV_PY.exists() and not args.skip_train:
        print(f"[ERROR] ACE-Step 학습 환경이 없습니다: {VENV_PY}", flush=True)
        return 1

    trigger = make_trigger(model_name)
    progress(1, 8, "[1/5] 녹음 문장 정리 중...")
    samples = []
    for i, c in enumerate(clips, 1):
        src = Path(c.get("path") or "")
        text = (c.get("text") or "").strip()
        if not src.exists() or not text:
            log(f"  ⚠️ 건너뜀 ({i}): 파일 또는 문장 없음")
            continue
        dst = dataset_dir / f"clip_{i:02d}.wav"
        try:
            prepare_speech_clip(src, dst)
        except Exception as e:
            log(f"  ⚠️ 변환 실패 {src.name}: {e}")
            continue
        caption = (f"{trigger}, dry studio close-mic a cappella {gender_en} vocal, "
                   "natural Korean spoken voice, speaker identity reference, clear diction, no instruments, no reverb")
        samples.append({
            "audio_path": str(dst), "filename": dst.name, "caption": caption,
            "lyrics": f"[Verse]\n{text}", "bpm": 72, "keyscale": "",
            "timesignature": "4", "language": "ko", "is_instrumental": False,
            "duration": 20.0, "custom_tag": trigger,
        })
        log(f"  ✓ {i:02d} {text[:28]}")

    if len(samples) < 5:
        print("[ERROR] 학습에 쓸 녹음이 너무 적습니다. 문장을 5개 이상 녹음해 주세요.", flush=True)
        return 1

    dataset_json = model_dir / "dataset.json"
    dataset_json.write_text(json.dumps({
        "metadata": {"custom_tag": trigger, "tag_position": "prepend", "genre_ratio": 0},
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = {Path(s["filename"]).stem for s in samples}
    for pt in tensor_dir.glob("*.pt"):
        if pt.stem not in valid:
            pt.unlink(missing_ok=True)

    epochs = plan_speech_epochs(len(samples), args.preset)
    save_every = max(5, epochs // 4)
    log(f"🚀 [{model_name}] 내 목소리 학습 시작 — {len(samples)}문장, {gender_en}, {epochs} epoch")
    progress(2, 20, f"[2/5] {len(samples)}문장 데이터셋 준비 완료")

    if args.skip_train:
        log("• --skip-train: 데이터셋까지만 생성했습니다.")
        return 0

    if (lora_dir / "final").exists():
        shutil.rmtree(lora_dir / "final", ignore_errors=True)
    cmd = [str(VENV_PY), "-u", str(TRAINER),
           "--dataset-json", str(dataset_json), "--tensor-dir", str(tensor_dir),
           "--output-dir", str(lora_dir), "--epochs", str(epochs),
           "--max-duration", "20", "--rank", "32", "--save-every", str(save_every)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    t0 = time.time()
    adapter_path, final_loss, last_error = None, None, None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", cwd=str(BASE_DIR / "audio_engines"))
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = re.match(r"\[ADAPTER:(.+)\]$", line)
        if m:
            adapter_path = m.group(1)
            continue
        m = re.match(r"\[FINAL_LOSS:([\d.]+)\]$", line)
        if m:
            final_loss = float(m.group(1))
            continue
        if line.startswith("[ERROR]"):
            last_error = line
            log("  " + line)
            continue
        if line.startswith("[LOG]") or line.startswith("[PROGRESS:"):
            print(line, flush=True)
        elif re.search(r"error|traceback|exception", line, re.I):
            log("  " + line[:300])
    proc.wait()
    if proc.returncode != 0 or not adapter_path:
        print(last_error or f"[ERROR] LoRA 학습 프로세스 실패 (exit {proc.returncode})", flush=True)
        return 1

    progress(5, 98, "[5/5] 별칭으로 목소리 등록 중...")
    meta = {
        "name": model_name, "kind": "speech_clone", "gender": gender_en,
        "engine": "ACE-Step 1.5 turbo", "adapter_type": "lora",
        "adapter_path": adapter_path, "trigger_tag": trigger,
        "clips_count": len(samples), "songs_count": len(samples),
        "songs_with_lyrics": len(samples), "epochs": epochs, "preset": args.preset,
        "final_loss": final_loss, "train_minutes": round((time.time() - t0) / 60, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"✅ 내 목소리 등록: {model_name} | {len(samples)}문장 | {epochs} epoch")
    progress(5, 100, "[5/5] 학습 완료")
    print(f"[DONE:{model_name}]", flush=True)
    return 0


# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ACE-Step LoRA vocal/style training pipeline")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--audio-files", nargs="*", default=[])
    parser.add_argument("--preset", default="standard", choices=["fast", "standard", "deep"])
    parser.add_argument("--genre-filter", default="all", choices=["kpop_ballad", "all", "trot", "ballad", "kpop", "rock", "rnb", "hiphop", "jazz", "folk", "electronic", "classical", "gospel"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-train", action="store_true", help="데이터셋만 만들고 학습은 생략 (점검용)")
    parser.add_argument("--mode", default="songs", choices=["songs", "speech"])
    parser.add_argument("--clips-json", default=None, help="따라 읽기 학습용 클립 목록")
    parser.add_argument("--gender", default="female", choices=["female", "male"])
    args = parser.parse_args()

    if args.mode == "speech":
        if not args.clips_json or not Path(args.clips_json).exists():
            print("[ERROR] --clips-json 파일이 필요합니다.", flush=True)
            return 1
        return train_speech_voice(args)

    model_name = re.sub(r'[^\w\s-]', '', args.model_name).strip() or "CustomSinger"
    model_dir = Path(args.output_dir) / model_name
    dataset_dir = model_dir / "dataset"
    tensor_dir = model_dir / "tensors"
    lora_dir = model_dir / "lora"
    for d in (dataset_dir, tensor_dir, lora_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not VENV_PY.exists() and not args.skip_train:
        print(f"[ERROR] ACE-Step 학습 환경이 없습니다: {VENV_PY}", flush=True)
        return 1

    raw_files = []
    if args.file_list and Path(args.file_list).exists():
        try:
            raw_files = json.loads(Path(args.file_list).read_text(encoding="utf-8-sig"))
        except Exception as ex:
            log(f"⚠️ file-list 로드 경고: {ex}")
    if not raw_files and args.audio_files:
        raw_files = args.audio_files
    audio_files = [Path(fix_long_path(str(f))) for f in raw_files if Path(fix_long_path(str(f))).exists()]

    cache_file = model_dir / "audio_cache.json"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    except Exception:
        cache = {}
    # 이전(가짜 학습) 버전 캐시에는 가사/BPM 라벨이 없으므로 다시 분석한다.
    cache = {k: v for k, v in cache.items() if isinstance(v, dict) and "lyrics" in v}

    if not audio_files and not cache:
        print("[ERROR] 유효한 음원 파일이 존재하지 않습니다.", flush=True)
        return 1

    trigger = make_trigger(model_name)
    log(f"🚀 [{model_name}] ACE-Step LoRA 실제 학습 파이프라인 시작")
    log(f"• 입력 음원 {len(audio_files)}곡 | 학습 모드 {args.preset.upper()} | 트리거 태그 '{trigger}'")
    used, total = get_gpu_vram()
    if total:
        log(f"• GPU VRAM 사용 {used} / {total} MiB")

    # ------------------------------------------------------------- 1단계
    progress(1, 2, f"[1/5] {len(audio_files)}곡 장르 선별 및 중복 검사 중...")
    todo = []
    seen = set()
    genre_skipped = dup_skipped = 0
    for idx, af in enumerate(audio_files):
        sig = get_audio_signature(af)
        if sig in seen:
            dup_skipped += 1
            continue
        seen.add(sig)
        if sig in cache:
            dup_skipped += 1
            continue
        if args.genre_filter == "kpop_ballad":
            ok, reason, category = classify_audio_genre(af)
            if not ok:
                genre_skipped += 1
                log(f"  ⏭️ PASS ({reason}): {af.name}")
                continue
        else:
            from training_genres import detect_genre
            category = detect_genre(af) if args.genre_filter == "all" else args.genre_filter
        todo.append((af, sig, category))
        if idx % 10 == 0:
            progress(1, 2 + int(idx / max(1, len(audio_files)) * 10), f"[1/5] 선별 중 ({idx + 1}/{len(audio_files)})")
    log(f"  ✓ 신규 분석 {len(todo)}곡 | 이미 학습 데이터에 있음/중복 {dup_skipped}곡 | 장르 제외 {genre_skipped}곡")

    # ------------------------------------------------------------- 2단계
    transcriber = Transcriber()
    if todo:
        log("▶ [2단계: 학습 라벨 생성] 가사(가사 파일 우선, 없으면 받아쓰기) + BPM/키 + 캡션")
    for i, (af, sig, category) in enumerate(todo):
        progress(2, 12 + int(i / max(1, len(todo)) * 26), f"[2/5] 곡 분석 ({i + 1}/{len(todo)}): {af.name[:40]}")
        try:
            bpm, keyscale, total_sec = analyze_music(af, MAX_TRAIN_SEC)
        except Exception as e:
            log(f"  ⚠️ 음원 분석 실패, 건너뜀: {af.name} ({e})")
            continue
        if total_sec and total_sec < 20:
            log(f"  ⏭️ 20초 미만 음원 제외: {af.name}")
            continue

        lyrics, src = find_sidecar_lyrics(af)
        if lyrics:
            source = f"가사 파일({src})"
        else:
            segs = None
            try:
                segs = transcriber.transcribe(af, MAX_TRAIN_SEC)
            except Exception as e:
                log(f"  ⚠️ 받아쓰기 실패: {af.name} ({type(e).__name__}: {e})")
            lyrics = structure_transcript(segs or [])
            source = "Whisper 받아쓰기"
        n_lines = len([l for l in lyrics.splitlines() if l.strip() and not _TAG_LINE.match(l)])
        is_inst = n_lines < 3
        if is_inst:
            lyrics = "[Instrumental]"
            source = "가사 없음(연주 위주로 학습)"

        ext = af.suffix.lower()
        link_name = f"trk_{sig.split('_')[-1]}{ext}"
        try:
            link_into_dataset(af, dataset_dir / link_name)
        except Exception as e:
            log(f"  ⚠️ 데이터셋 복사 실패, 건너뜀: {af.name} ({e})")
            continue
        cache[sig] = {
            "source_name": af.name, "file": link_name, "category": category,
            "lyrics": lyrics, "lyrics_source": source, "is_instrumental": is_inst,
            "bpm": bpm, "keyscale": keyscale, "duration": round(total_sec, 1),
            "caption": build_caption(category, bpm, is_inst),
        }
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  ✓ {af.name[:40]} | {source} {n_lines}줄 | BPM {bpm} | {keyscale}")
    transcriber.unload()

    samples = []
    for sig, e in cache.items():
        p = dataset_dir / e["file"]
        if not p.exists():
            continue
        samples.append({
            "audio_path": str(p), "filename": e["file"], "caption": e["caption"],
            "lyrics": e["lyrics"], "bpm": e.get("bpm"), "keyscale": e.get("keyscale") or "",
            "timesignature": "4", "language": "ko", "is_instrumental": e.get("is_instrumental", False),
            "duration": min(e.get("duration") or MAX_TRAIN_SEC, MAX_TRAIN_SEC), "custom_tag": trigger,
        })
    if not samples:
        print("[ERROR] 학습할 곡이 없습니다 (모두 제외되었거나 분석 실패).", flush=True)
        return 1
    with_lyrics = sum(1 for s in samples if not s["is_instrumental"])
    log(f"• 학습 데이터셋: 총 {len(samples)}곡 (가사 포함 {with_lyrics}곡) — 이전 학습곡 누적 포함")
    if with_lyrics < len(samples) * 0.5:
        log("  ⚠️ 가사가 있는 곡이 절반 미만입니다. 음원 옆에 같은 이름의 .txt/.lrc 가사 파일을 두면 보컬 학습 품질이 크게 좋아집니다.")

    dataset_json = model_dir / "dataset.json"
    dataset_json.write_text(json.dumps({
        "metadata": {"custom_tag": trigger, "tag_position": "prepend", "genre_ratio": 0},
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # 곡 목록/라벨이 바뀌면 전처리 텐서의 캡션이 달라지므로, 사라진 곡의 텐서는 지운다.
    valid_stems = {Path(s["filename"]).stem for s in samples}
    for pt in tensor_dir.glob("*.pt"):
        if pt.stem not in valid_stems:
            pt.unlink(missing_ok=True)

    epochs = plan_epochs(len(samples), args.preset)
    save_every = max(5, epochs // 4)
    log(f"• 학습 계획: {len(samples)}곡 × {epochs} epoch (ACE-Step 공식 가이드 기준, 모드 {args.preset})")

    if args.skip_train:
        log("• --skip-train: 데이터셋까지만 생성했습니다.")
        return 0

    # ------------------------------------------------------------- 3~4단계
    out_dir = lora_dir
    if (out_dir / "final").exists():
        shutil.rmtree(out_dir / "final", ignore_errors=True)
    cmd = [str(VENV_PY), "-u", str(TRAINER),
           "--dataset-json", str(dataset_json), "--tensor-dir", str(tensor_dir),
           "--output-dir", str(out_dir), "--epochs", str(epochs),
           "--max-duration", str(MAX_TRAIN_SEC), "--save-every", str(save_every)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("OMP_NUM_THREADS", None)
    env.pop("MKL_NUM_THREADS", None)
    t0 = time.time()
    adapter_path, final_loss, last_error = None, None, None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", cwd=str(BASE_DIR / "audio_engines"))
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        m = re.match(r"\[ADAPTER:(.+)\]$", line)
        if m:
            adapter_path = m.group(1)
            continue
        m = re.match(r"\[FINAL_LOSS:([\d.]+)\]$", line)
        if m:
            final_loss = float(m.group(1))
            continue
        if line.startswith("[ERROR]"):
            last_error = line
            log("  " + line)
            continue
        if line.startswith("[LOG]") or line.startswith("[PROGRESS:"):
            print(line, flush=True)
        elif re.search(r"error|traceback|exception", line, re.I):
            log("  " + line[:300])
    proc.wait()
    if proc.returncode != 0 or not adapter_path:
        print(last_error or f"[ERROR] LoRA 학습 프로세스 실패 (exit {proc.returncode})", flush=True)
        return 1

    # ------------------------------------------------------------- 5단계
    progress(5, 98, "[5/5] 학습된 보컬 모델 등록 중...")
    meta_path = model_dir / "meta.json"
    meta = {
        "name": model_name,
        "kind": "song_lora",
        "engine": "ACE-Step 1.5 turbo",
        "adapter_type": "lora",
        "adapter_path": adapter_path,
        "trigger_tag": trigger,
        "caption_hint": samples[0]["caption"],
        "songs_count": len(samples),
        "songs_with_lyrics": with_lyrics,
        "epochs": epochs,
        "preset": args.preset,
        "final_loss": final_loss,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy = model_dir / "ref_vocal.wav"
    if legacy.exists():
        legacy.unlink(missing_ok=True)
    legacy_work = model_dir / "work"
    if legacy_work.exists():
        shutil.rmtree(legacy_work, ignore_errors=True)
    log(f"✅ 학습 완료: {model_name} | {len(samples)}곡 | {epochs} epoch | loss {final_loss}")
    progress(5, 100, "[5/5] 학습 완료")
    print(f"[DONE:{model_name}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
