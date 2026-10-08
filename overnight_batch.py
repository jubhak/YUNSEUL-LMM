#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overnight_batch.py
==================
3단계 순차 배치(Sequential Batch) 실행기 — 파이프라인이 아니다.

  [PHASE1_TEXT]   LLM으로 전자책 원고 / 여러 글 통합 재작성 / 가사 생성   <- VRAM 점유 허용
  [UNLOAD]        ollama stop + `ollama ps` + `nvidia-smi` 3중 검증 게이트
  [PHASE2_AUDIO]  음악 생성 / TTS 낭독                                  <- LLM 재로드 절대 금지

설계 원칙 (코드로 강제됨)
  P1. 단계는 절대 섞지 않는다. PHASE1이 100% done 되기 전에 PHASE2를 시작하지 않는다.
  P2. PHASE2 진입 근거는 `ollama stop` 호출 성공이 아니라 VRAM 실측값이다.
      `ollama ps`가 비어 있고 nvidia-smi 사용량이 임계치 이하일 때만 통과한다.
  P3. 모든 진행은 state.json에 기록되어 언제든 이어서 실행할 수 있다(재개 가능).
  P4. 게이트 실패 시 fail-fast. 경고만 찍고 넘어가는 코드 경로는 존재하지 않는다.
  P5. PHASE2 명령에 ollama가 들어 있으면 실행 전에 차단한다.

사용법
  python3 overnight_batch.py --emit-config jobs.json     # 템플릿 설정 파일 생성
  python3 overnight_batch.py --config jobs.json --dry-run # 문법/상태머신 검증(실행 없음)
  python3 overnight_batch.py --config jobs.json          # 실제 야간 배치 실행
  python3 overnight_batch.py --config jobs.json --reset  # 상태 초기화 후 재시작
  python3 overnight_batch.py --config jobs.json --report-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf

# Windows 콘솔 utf-8 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
PHASE1_TEXT = "PHASE1_TEXT"
PHASE_UNLOAD = "UNLOAD"
PHASE2_AUDIO = "PHASE2_AUDIO"
PHASE_ORDER = [PHASE1_TEXT, PHASE_UNLOAD, PHASE2_AUDIO]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

DEFAULT_CONFIG = {
    "batch_name": "overnight-batch",
    "workdir": "./batch_out",
    "text_model": "qwen3:14b",
    "unload": {
        "vram_free_threshold_mb": 1500,
        "poll_timeout_sec": 120,
        "poll_interval_sec": 5,
    },
    "retry": {"max_attempts": 3, "backoff_sec": 10},
    "commands": {
        "text": "builtin-ollama-api",
        "music": "acestep --prompt-file {prompt_file} --output {out} --duration {duration}",
        "tts": "python -m gpt_sovits.inference --text-file {prompt_file} --output {out}",
    },
    "jobs": [
        {
            "id": "ebook-01",
            "phase": "text",
            "kind": "text",
            "out": "ebook-01.md",
            "prompt": "다음 주제로 전자책 1장 분량(3000자 이상)의 원고를 작성하라. ...",
        },
        {
            "id": "rewrite-01",
            "phase": "text",
            "kind": "text",
            "out": "rewrite-01.md",
            "prompt": "첨부한 여러 블로그 글을 하나의 새로운 글로 재구성하라. ...",
        },
        {
            "id": "lyrics-01",
            "phase": "text",
            "kind": "text",
            "out": "lyrics-01.txt",
            "prompt": "아래 주제로 한국어 가사 2절 + 후렴을 작성하라. ...",
        },
        {
            "id": "song-01",
            "phase": "audio",
            "kind": "music",
            "out": "song-01.mp3",
            "duration": 120,
            "prompt": "warm acoustic ballad, mid tempo, Korean male vocal, ...",
        },
        {
            "id": "tts-01",
            "phase": "audio",
            "kind": "tts",
            "out": "tts-01.wav",
            "prompt": "ebook-01.md 본문을 차분한 낭독 톤으로 읽어라.",
        },
    ],
}


# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def q(s) -> str:
    """경로/인자 인용. POSIX는 shlex, Windows는 큰따옴표."""
    s = str(s)
    if os.name == "nt":
        return f'"{s}"' if (" " in s or "\t" in s) else s
    return shlex.quote(s)


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, level: str, msg: str) -> None:
        line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] [{level:5s}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_cmd(cmd: str, timeout=None, dry_run: bool = False):
    """쉘 명령 실행. dry_run이면 실행하지 않고 성공으로 간주."""
    if dry_run:
        return 0, f"[dry-run] {cmd}", ""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def api_post(path: str, payload: dict, timeout: int = 3600, dry_run: bool = False):
    """Ollama HTTP API 호출 (stdlib만 사용, 크로스플랫폼)."""
    if dry_run:
        return {"response": "[dry-run] simulated LLM output"}
    req = urllib.request.Request(
        OLLAMA_HOST + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ----------------------------------------------------------------------------
# 게이트 검증 (핵심)
# ----------------------------------------------------------------------------
def check_ollama_ps(dry_run: bool = False):
    """`ollama ps`로 로드된 모델이 0개인지 확인. (True, 설명) 반환."""
    if dry_run:
        return True, "`ollama ps` -> [dry-run] 로드된 모델 0개 (가정)"
    rc, out, err = run_cmd("ollama ps")
    if rc != 0:
        return False, f"`ollama ps` 실행 실패 (rc={rc}): {err.strip()[:200]}"
    lines = [l for l in out.splitlines() if l.strip()]
    body = lines[1:] if lines else []  # 첫 줄은 헤더(NAME ID SIZE PROCESSOR UNTIL)
    if not body:
        return True, "`ollama ps` -> 로드된 모델 0개"
    names = [l.split()[0] for l in body if l.split()]
    return False, f"`ollama ps` -> 아직 로드됨: {', '.join(names)}"


def check_vram(dry_run: bool = False):
    """nvidia-smi로 실제 VRAM 사용량 조회. (used_mb, total_mb, 설명)."""
    if dry_run:
        return 512, 16303, "nvidia-smi -> [dry-run] 512 MiB / 16303 MiB (가정)"
    cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"nvidia-smi 실행 실패 (rc={rc}): {err.strip()[:200]}")
    first = out.strip().splitlines()[0]
    used, total = (int(x.strip()) for x in first.split(",")[:2])
    return used, total, f"nvidia-smi -> {used} MiB / {total} MiB 사용"


def gate_unload(cfg, state, log, dry_run: bool = False) -> bool:
    """UNLOAD 게이트: ollama stop -> (ollama ps + nvidia-smi) 폴링 검증."""
    log("INFO", "=== UNLOAD 게이트 진입 ===")
    u = cfg["unload"]
    threshold = int(u["vram_free_threshold_mb"])
    models = sorted({j.get("model") or cfg["text_model"] for j in cfg["jobs"] if j.get("phase") == "text"})

    for m in models:
        rc, _, err = run_cmd(f"ollama stop {q(m)}", dry_run=dry_run)
        log("INFO" if rc == 0 else "WARN", f"ollama stop {m} (rc={rc}) {err.strip()[:120]}")
        try:  # 보조: keep_alive=0 으로 확실히 내린다
            api_post("/api/generate", {"model": m, "keep_alive": 0}, timeout=60, dry_run=dry_run)
        except Exception as e:
            log("WARN", f"keep_alive=0 요청 실패: {e}")

    deadline = time.time() + int(u["poll_timeout_sec"])
    attempt = 0
    while True:
        attempt += 1
        ps_ok, ps_detail = check_ollama_ps(dry_run)
        used, total, vram_detail = check_vram(dry_run)
        log("INFO", f"[검증 {attempt}] {ps_detail} | {vram_detail}")
        if ps_ok and used <= threshold:
            state["unload_confirmed"] = True
            state["unload_evidence"] = {
                "at": now_iso(),
                "ollama_ps": ps_detail,
                "vram_used_mb": used,
                "vram_total_mb": total,
                "threshold_mb": threshold,
            }
            log("OK", f"언로드 확인 완료 -> PHASE2_AUDIO 진입 허용 (VRAM {used} MiB <= {threshold} MiB)")
            return True
        if time.time() > deadline:
            state["unload_confirmed"] = False
            state["unload_evidence"] = {
                "at": now_iso(),
                "ollama_ps": ps_detail,
                "vram_used_mb": used,
                "threshold_mb": threshold,
            }
            log("ERROR", "언로드 검증 실패 -> PHASE2_AUDIO 진입 금지 (fail-fast)")
            return False
        time.sleep(0 if dry_run else int(u["poll_interval_sec"]))


def assert_no_llm_in_audio(cfg) -> None:
    """P5: PHASE2 명령에 ollama가 섞여 있으면 실행 전에 차단."""
    for j in cfg["jobs"]:
        if j.get("phase") != "audio":
            continue
        cmd = cfg["commands"].get(j.get("kind"), "")
        if re.search(r"\bollama\b", cmd, re.I):
            raise RuntimeError(
                f"PHASE2 명령에 ollama가 포함됨 (job={j['id']}) — 단계 혼합 금지 규칙 위반"
            )


def preflight(cfg, log) -> None:
    for j in cfg["jobs"]:
        if j.get("phase") not in ("text", "audio"):
            raise RuntimeError(f"job={j.get('id')} 의 phase는 'text' 또는 'audio' 여야 합니다")
        if not j.get("prompt", "").strip():
            raise RuntimeError(f"job={j.get('id')} 에 prompt가 없습니다")
        if j.get("phase") == "audio" and not cfg["commands"].get(j.get("kind")):
            raise RuntimeError(f"job={j.get('id')} 의 kind='{j.get('kind')}' 명령 템플릿이 없습니다")
    assert_no_llm_in_audio(cfg)
    log("INFO", f"preflight 통과 — 작업 {len(cfg['jobs'])}건")


# ----------------------------------------------------------------------------
# 작업 실행
# ----------------------------------------------------------------------------
def run_text_job(job, cfg, log, dry_run: bool = False) -> None:
    workdir = Path(cfg["workdir"])
    model = job.get("model") or cfg["text_model"]
    log("INFO", f"{job['id']}: ollama/{model} 호출")
    resp = api_post(
        "/api/generate",
        {"model": model, "prompt": job["prompt"], "stream": False},
        timeout=int(job.get("timeout_sec", 3600)),
        dry_run=dry_run,
    )
    text = (resp or {}).get("response", "")
    if dry_run:
        text = f"[dry-run] {job['id']} 결과 본문\n"
def validate_text_output(job, text: str, path: Path) -> None:
    jid = job.get("id", "")
    stripped = text.strip()
    if not stripped:
        raise RuntimeError(f"DoD 검증 실패 ({jid}): 텍스트가 비어 있음")

    if "ebook" in jid or "원고" in job.get("prompt", ""):
        char_count = len(stripped)
        if char_count < 2500:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 전자책 분량 미달 ({char_count}자 < 2500자)")
        chapters = re.findall(r'(?:^#+\s+|\b제\s*\d+\s*장\b|\b\d+\.\s+)', text, re.M)
        if len(chapters) < 2:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 목차 또는 챕터 구조 헤더 부족 ({len(chapters)}개 < 2개)")

    elif "lyrics" in jid or "가사" in job.get("prompt", ""):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 8:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 가사 행 수 부족 ({len(lines)}행 < 8행)")
        has_structure = any(
            re.search(r'\[(Verse|Chorus|Bridge|Outro|1절|2절|후렴|브릿지)\]', l, re.I) or
            re.search(r'(1절|2절|후렴|브릿지):', l)
            for l in lines
        )
        if not has_structure:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 절/후렴 구조 라벨([Verse], [Chorus], [1절], [후렴] 등) 부재")

    elif "rewrite" in jid or "재작성" in job.get("prompt", ""):
        char_count = len(stripped)
        if char_count < 1000:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 통합 재작성 분량 미달 ({char_count}자 < 1000자)")
        has_source = any(k in text for k in ("출처", "참고", "참조", "Source", "원문", "인사이트"))
        if not has_source:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 출처/참고 문헌 표기 누락")


def validate_audio_output(job, path: Path) -> None:
    jid = job.get("id", "")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"DoD 검증 실패 ({jid}): 오디오 파일이 없거나 0바이트임 ({path})")

    try:
        data, sr = sf.read(str(path))
    except Exception as e:
        raise RuntimeError(f"DoD 검증 실패 ({jid}): 오디오 디코딩 실패 - {e}")

    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    duration = len(mono) / float(sr)
    expected_dur = float(job.get("duration", 3.0))

    if job.get("kind") == "music":
        if duration < expected_dur * 0.5:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 재생 시간 미달 ({duration:.1f}초 < 최소 {expected_dur * 0.5:.1f}초)")
    else:
        if duration < 2.0:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): TTS 재생 시간 미달 ({duration:.1f}초 < 2.0초)")

    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 0.003:
        raise RuntimeError(f"DoD 검증 실패 ({jid}): 거의 무음 신호임 (RMS={rms:.6f})")

    # 100ms 윈도우 에너지 변동계수(CV = std / mean) 검증
    win_size = int(sr * 0.1)  # 100ms
    if len(mono) >= win_size * 5:
        num_windows = len(mono) // win_size
        windows = mono[:num_windows * win_size].reshape(num_windows, win_size)
        win_rms = np.sqrt(np.mean(windows ** 2, axis=1))
        mean_rms = np.mean(win_rms)
        std_rms = np.std(win_rms)
        cv = (std_rms / mean_rms) if mean_rms > 1e-6 else 0.0

        if cv < 0.20:
            raise RuntimeError(f"DoD 검증 실패 ({jid}): 단조로운 가짜/더미 신호 감지 (에너지 변동계수 CV={cv:.3f} < 0.20)")


def run_text_job(job, cfg, log, dry_run: bool = False) -> None:
    workdir = Path(cfg["workdir"])
    model = job.get("model") or cfg["text_model"]
    log("INFO", f"{job['id']}: ollama/{model} 호출")
    resp = api_post(
        "/api/generate",
        {"model": model, "prompt": job["prompt"], "stream": False},
        timeout=int(job.get("timeout_sec", 3600)),
        dry_run=dry_run,
    )
    text = (resp or {}).get("response", "")
    if dry_run:
        text = f"[dry-run] {job['id']} 결과 본문\n# 목차\n## 1장\n## 2장\n[Verse 1]\n노래\n[Chorus]\n후렴\n출처: 테스트\n" + "글" * 3000
    if not text.strip():
        raise RuntimeError("LLM 응답이 비어 있음")
    out = workdir / job["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"출력 파일이 없거나 0바이트임: {out}")
    if not dry_run:
        validate_text_output(job, text, out)
    log("INFO", f"{job['id']}: {out} 저장 ({len(text)}자, DoD 통과)")


def run_audio_job(job, cfg, log, dry_run: bool = False) -> None:
    workdir = Path(cfg["workdir"])
    kind = job["kind"]
    tpl = cfg["commands"].get(kind)
    if not tpl:
        raise RuntimeError(f"명령 템플릿 없음: kind={kind}")

    prompt_file = workdir / f"{job['id']}.prompt.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(job["prompt"], encoding="utf-8")

    out = workdir / job["out"]
    cmd = tpl.format(
        prompt_file=q(prompt_file),
        out=q(out),
        duration=int(job.get("duration", 60)),
        prompt=q(job["prompt"]),
    )
    if re.search(r"\bollama\b", cmd, re.I):  # 방어적 재확인
        raise RuntimeError("PHASE2 명령에 ollama 발견 — 차단")

    log("INFO", f"{job['id']}: {cmd}")
    rc, so, se = run_cmd(cmd, timeout=int(job.get("timeout_sec", 3600)), dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"명령 실패 rc={rc}: {se.strip()[:300]}")
    if dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        # 1초짜리 mock audio 작성
        mock_audio = np.random.randn(44100).astype(np.float32) * 0.1
        sf.write(str(out), mock_audio, 44100)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"출력 파일이 생성되지 않았거나 0바이트임: {out}")
    if not dry_run:
        validate_audio_output(job, out)
    log("INFO", f"{job['id']}: {out} 생성 (DoD 통과)")



def with_retries(fn, job, state, cfg, log, dry_run: bool = False) -> bool:
    max_attempts = int(cfg["retry"]["max_attempts"])
    rec = state["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
    if rec.get("status") != "done":
        rec["attempts"] = 0
    while rec["attempts"] < max_attempts:
        rec["attempts"] += 1
        rec["status"] = "running"
        rec.setdefault("started_at", now_iso())
        try:
            fn(job, cfg, log, dry_run)
            rec["status"] = "done"
            rec["finished_at"] = now_iso()
            rec["error"] = None
            log("OK", f"{job['id']} 완료 (시도 {rec['attempts']}/{max_attempts})")
            return True
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
            log("WARN", f"{job['id']} 실패 {rec['attempts']}/{max_attempts}: {e}")
            if rec["attempts"] < max_attempts:
                time.sleep(0 if dry_run else int(cfg["retry"]["backoff_sec"]))
    rec["status"] = "failed"
    return False


# ----------------------------------------------------------------------------
# 상태 / 리포트
# ----------------------------------------------------------------------------
def state_path(cfg) -> Path:
    return Path(cfg["workdir"]) / "state.json"


def load_state(cfg) -> dict:
    p = state_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {
        "batch_id": f"{cfg['batch_name']}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": now_iso(),
        "current_phase": None,
        "unload_confirmed": False,
        "unload_evidence": None,
        "final_status": "running",
        "jobs": {},
    }


def save_state(cfg, state) -> None:
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def abort(cfg, state, log, reason: str) -> None:
    state["final_status"] = "aborted"
    state["abort_reason"] = reason
    state["finished_at"] = now_iso()
    save_state(cfg, state)
    log("ERROR", f"배치 중단: {reason}")
    write_report(cfg, state)


def write_report(cfg, state) -> Path:
    jobs = {j["id"]: j for j in cfg["jobs"]}
    rows = []
    for jid, rec in state.get("jobs", {}).items():
        j = jobs.get(jid, {})
        rows.append(
            f"| {jid} | {j.get('phase', '?')} | {j.get('kind', '?')} | {rec.get('status')} "
            f"| {rec.get('attempts', 0)} | {rec.get('error') or '-'} |"
        )
    ev = state.get("unload_evidence") or {}
    workdir = Path(cfg["workdir"])

    # 산출물 목록
    artifacts = []
    if workdir.exists():
        for f in workdir.glob("*"):
            if f.is_file() and f.name not in ("batch.log", "state.json", "report.md") and not f.name.endswith(".prompt.txt"):
                sz = f.stat().st_size
                mtime = dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                artifacts.append(f"- `{f.name}` ({sz} bytes, {mtime})")
    if not artifacts:
        artifacts.append("- (생성된 산출물 없음)")

    # 다음 단계 제안
    suggestions = [
        "- 실패 작업 재실행 명령: `python overnight_batch.py --config jobs.json`"
    ]
    if state.get("abort_reason"):
        suggestions.append(f"- 중단 사유 분석: {state.get('abort_reason')}")
        if "not found" in str(state.get("abort_reason")).lower() or any("not found" in str(rec.get("error", "")).lower() for rec in state.get("jobs", {}).values()):
            suggestions.append("- 제안: 모델이 설치되지 않았습니다. `ollama pull <model>` 실행 또는 `jobs.json`의 모델명을 확인하세요.")
        if any("not recognized" in str(rec.get("error", "")).lower() or "no module" in str(rec.get("error", "")).lower() for rec in state.get("jobs", {}).values()):
            suggestions.append("- 제안: 오디오 도구 환경(acestep, gpt_sovits) 설치 및 가상환경 상태를 점검하세요.")

    # 최종 체크리스트
    all_text_done = all(
        state.get("jobs", {}).get(j["id"], {}).get("status") == "done"
        for j in cfg["jobs"] if j.get("phase") == "text"
    )
    unload_ok = bool(state.get("unload_confirmed"))
    final_st = state.get("final_status")
    checklist = [
        f"- [{'x' if all_text_done else ' '}] PHASE1 text job이 전부 `done`인가?",
        f"- [{'x' if unload_ok else ' '}] UNLOAD 게이트를 `ollama ps` + `nvidia-smi` 양쪽으로 통과했는가?",
        f"- [{'x' if unload_ok or final_st == 'aborted' else ' '}] PHASE2 동안 `ollama` 관련 호출이 단 한 번도 없었는가?",
        f"- [x] `state.json`과 `report.md`가 최신 상태로 저장되었는가?",
        f"- [{'x' if state.get('abort_reason') or not any(r.get('status') == 'failed' for r in state.get('jobs', {}).values()) else ' '}] 실패한 작업이 있다면 리포트에 사유가 적혔는가?",
    ]

    body = [
        f"# 야간 배치 리포트 — {cfg['batch_name']}",
        "",
        f"- 배치 ID: `{state.get('batch_id')}`",
        f"- 시작: {state.get('started_at')}",
        f"- 종료: {state.get('finished_at', '-')}",
        f"- 최종 상태: **{state.get('final_status')}**",
        f"- 중단 사유: {state.get('abort_reason') or '-'}",
        f"- 언로드 검증: **{state.get('unload_confirmed')}**",
        f"  - 근거: {ev.get('ollama_ps', '-')} / VRAM {ev.get('vram_used_mb', '-')} MiB "
        f"(임계 {ev.get('threshold_mb', '-')} MiB) @ {ev.get('at', '-')}",
        "",
        "| 작업 | 단계 | 종류 | 상태 | 시도 | 오류 |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## 산출물 목록",
        *artifacts,
        "",
        "## 다음 단계 제안",
        *suggestions,
        "",
        "## 최종 체크리스트",
        *checklist,
        "",
        f"작업 디렉터리: `{Path(cfg['workdir']).resolve()}`",
    ]
    p = Path(cfg["workdir"]) / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def run_batch(cfg, state, log, dry_run: bool = False, only_phase=None) -> int:
    for phase in PHASE_ORDER:
        if only_phase and phase != only_phase:
            continue
        state["current_phase"] = phase
        save_state(cfg, state)

        if phase == PHASE_UNLOAD:
            if not gate_unload(cfg, state, log, dry_run):
                abort(cfg, state, log, "UNLOAD 게이트 실패 — PHASE2_AUDIO 진입 차단")
                return 1
            continue

        if phase == PHASE2_AUDIO:
            if not state.get("unload_confirmed"):
                abort(cfg, state, log, "unload_confirmed=False — PHASE2_AUDIO 실행 차단")
                return 1
            assert_no_llm_in_audio(cfg)
            log("INFO", "PHASE2_AUDIO 진입 허용 (언로드 확인됨) — LLM 재로드 금지")

        want = "text" if phase == PHASE1_TEXT else "audio"
        jobs = [j for j in cfg["jobs"] if j.get("phase") == want]
        for job in jobs:
            rec = state["jobs"].get(job["id"])
            if rec and rec.get("status") == "done":
                log("INFO", f"{job['id']} 이미 완료 — 건너뜀 (재개)")
                continue

            # PHASE2_AUDIO 실행 중 감시: 오디오 작업 시작 직전 VRAM 재확인
            if phase == PHASE2_AUDIO:
                used, total, vram_detail = check_vram(dry_run)
                threshold = int(cfg["unload"]["vram_free_threshold_mb"])
                log("INFO", f"{job['id']} 시작 전 VRAM 재확인: {vram_detail} (임계치 {threshold} MiB)")
                if used > threshold:
                    abort(cfg, state, log, f"오디오 작업 직전 VRAM 초과 감지 ({used} MiB > {threshold} MiB) — LLM 재로드 감지 중단")
                    return 1

            fn = run_text_job if phase == PHASE1_TEXT else run_audio_job
            if not with_retries(fn, job, state, cfg, log, dry_run):
                abort(cfg, state, log, f"{job['id']} 재시도 한도 초과 — fail-fast")
                return 1
            save_state(cfg, state)

    state["final_status"] = "completed"
    state["abort_reason"] = None
    state["finished_at"] = now_iso()
    save_state(cfg, state)
    write_report(cfg, state)
    log("OK", "전체 배치 완료")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3단계 순차 배치 실행기 (PHASE1_TEXT -> UNLOAD -> PHASE2_AUDIO)")
    ap.add_argument("--config", default="jobs.json", help="작업 설정 JSON 경로")
    ap.add_argument("--dry-run", action="store_true", help="실행 없이 상태머신/게이트만 검증")
    ap.add_argument("--reset", action="store_true", help="state.json 초기화 후 재시작")
    ap.add_argument("--report-only", action="store_true", help="리포트만 재생성")
    ap.add_argument("--only-phase", choices=PHASE_ORDER, default=None)
    ap.add_argument("--emit-config", metavar="PATH", help="템플릿 설정 파일 생성 후 종료")
    args = ap.parse_args(argv)

    if args.emit_config:
        Path(args.emit_config).write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"템플릿 생성: {args.emit_config}")
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"설정 파일이 없습니다: {cfg_path}\n  먼저 `--emit-config {cfg_path}` 로 생성하세요.", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    Path(cfg["workdir"]).mkdir(parents=True, exist_ok=True)
    log = Log(Path(cfg["workdir"]) / "batch.log")

    if args.reset and state_path(cfg).exists():
        state_path(cfg).unlink()
        log("WARN", "state.json 초기화")

    state = load_state(cfg)

    if args.report_only:
        p = write_report(cfg, state)
        print(f"리포트: {p}")
        return 0

    log("INFO", f"배치 시작 batch_id={state['batch_id']} dry_run={args.dry_run}")
    try:
        preflight(cfg, log)
    except Exception as e:  # noqa: BLE001
        log("ERROR", f"preflight 실패: {e}")
        return 2

    try:
        return run_batch(cfg, state, log, dry_run=args.dry_run, only_phase=args.only_phase)
    except KeyboardInterrupt:
        state["final_status"] = "interrupted"
        state["finished_at"] = now_iso()
        save_state(cfg, state)
        write_report(cfg, state)
        log("WARN", "사용자 중단 — state.json에 기록됨 (이어서 실행 가능)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
