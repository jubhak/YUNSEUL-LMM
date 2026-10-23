"""Protect accompaniment: vocal operations require a separated or explicit vocal stem."""
import importlib.util
import os
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def demucs_python():
    configured = os.environ.get("YUNSEUL_AUDIO_PYTHON", "").strip()
    if configured:
        if not Path(configured).is_file():
            raise RuntimeError("YUNSEUL_AUDIO_PYTHON 경로를 찾을 수 없습니다.")
        candidates = [configured]
    else:
        base = Path(__file__).resolve().parent
        candidates = [] if getattr(sys, "frozen", False) else [sys.executable]
        candidates.extend(str(base / "audio_engines" / "venv_acestep" / suffix)
                          for suffix in ("Scripts/python.exe", "bin/python"))
    for exe in dict.fromkeys(candidates):
        if not Path(exe).is_file():
            continue
        if exe == sys.executable and not getattr(sys, "frozen", False):
            if importlib.util.find_spec("demucs") is not None:
                return exe
            continue
        try:
            check = subprocess.run([exe, "-c", "import demucs"], capture_output=True, timeout=15)
            if check.returncode == 0:
                return exe
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def prepare_stems(source, workdir, vocal=None, accompaniment=None):
    """Return (vocal, backing). Never treat a full mix as a solo vocal."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if bool(vocal) != bool(accompaniment):
        raise ValueError("보컬과 반주 스템을 함께 지정하세요.")
    if vocal:
        if not Path(vocal).is_file() or not Path(accompaniment).is_file():
            raise ValueError("지정된 스템 파일을 찾을 수 없습니다.")
        return Path(vocal), Path(accompaniment)
    exe = demucs_python()
    if exe is None:
        return None, None
    proc = subprocess.run(
        [exe, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
         "--out", str(workdir / "stems"), str(Path(source).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if proc.returncode:
        raise RuntimeError("보컬 분리 실패: " + (proc.stderr or proc.stdout)[-300:])
    stem = workdir / "stems" / "htdemucs" / Path(source).stem / "vocals.wav"
    if not stem.is_file():
        raise RuntimeError("분리된 보컬 파일이 없습니다.")
    # Reconstruct residual from original instead of independently normalizing backing.
    # This guarantees that an unchanged vocal stem sums back to the original mix.
    import svc
    original, voice = svc.decode(source), svc.decode(stem)
    if abs(len(original) - len(voice)) > svc.SR // 20:
        raise RuntimeError("분리 스템 길이가 원본과 다릅니다. 원본을 보호하기 위해 중단합니다.")
    voice = np.pad(voice[:len(original)], ((0, max(0, len(original) - len(voice))), (0, 0)))
    vocal_path, backing = workdir / "vocal.wav", workdir / "residual.wav"
    svc.write_wav(voice, vocal_path, codec="pcm_f32le")
    svc.write_wav(original - voice, backing, codec="pcm_f32le")
    return vocal_path, backing


def remix(processed, reference_vocal, backing, output):
    import svc
    v, ref, music = svc.decode(processed), svc.decode(reference_vocal), svc.decode(backing)
    if abs(len(v) - len(ref)) > svc.SR // 20 or abs(len(music) - len(ref)) > svc.SR // 20:
        raise RuntimeError("보컬 보정 중 길이가 바뀌었습니다. 리믹스를 중단합니다.")
    n = len(music)
    v = np.pad(v[:n], ((0, max(0, n - len(v))), (0, 0)))
    rms_ref = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))
    if rms_out < 1e-8:
        raise RuntimeError("보정된 보컬이 무음입니다.")
    # Undo stem-only loudness normalization before restoring original backing balance.
    gain = rms_ref / rms_out
    y = music + v * gain
    peak = float(np.max(np.abs(y))) if y.size else 0
    attenuation = min(1.0, 0.98 / max(peak, 1e-9))
    svc.write_wav(y * attenuation, output)
    return {"vocal_gain": round(gain, 5), "peak_gain": round(attenuation, 5),
            "samples": n, "backing_preserved": True}


def continuity_metrics(path):
    """Conservative mixed-audio dropout screen; NOT a singer identity detector."""
    import svc
    y = svc.decode(path, sr=16000, ch=1).reshape(-1)
    hop = 1600
    n = len(y) // hop
    if n < 7:
        return {"dropout_count": 0, "penalty": 0, "scope": "mix_energy_only"}
    blocks = y[:n * hop].reshape(n, hop).astype(np.float64)
    db = 20 * np.log10(np.sqrt(np.mean(blocks ** 2, axis=1)) + 1e-9)
    events = []
    for i in range(2, n - 2):
        neighbors = min(float(np.median(db[i-2:i])), float(np.median(db[i+1:i+3])))
        if neighbors > -40 and db[i] < neighbors - 18:
            if not events or (i * 0.1 - events[-1]) > 0.3:
                events.append(round(i * 0.1, 2))
    return {"dropout_count": len(events), "times_sec": events[:50],
            "penalty": min(20, len(events) * 3), "scope": "mix_energy_only",
            "note": "의도된 쉼도 포함할 수 있음. 음정/화자 동일성은 별도 청음 필요"}
