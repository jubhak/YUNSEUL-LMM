#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_engine.py - Ollama 호출 공통 모듈 + 한국어 반복 표현 감지기

- ollama_generate: 반복 억제 샘플링 옵션, qwen3 사고(think) 출력 제거, 스트리밍 취소 지원
- RepetitionGuard: 책 전체에서 거의 같은 문장 제거, 과다 사용 표현 집계, 문단 내 반복 패턴 진단
"""

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Callable, Iterable, List, Optional, Tuple

# 반복을 줄이는 기본 샘플링 값. 알 수 없는 옵션은 Ollama가 무시한다.
BASE_OPTIONS = {
    "num_ctx": 16384,
    "temperature": 0.75,
    "top_p": 0.9,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.12,
    "repeat_last_n": 512,
    "presence_penalty": 0.4,
    "frequency_penalty": 0.4,
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    # 닫히지 않은 <think> 블록(출력 한도에서 잘린 경우)
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip()


class GenerationCancelled(Exception):
    pass


def ollama_generate(
    prompt: str,
    model: str = "qwen3:14b",
    port: int = 11434,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    options: Optional[dict] = None,
    timeout: int = 600,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """Ollama /api/generate 스트리밍 호출. 취소 시 GenerationCancelled."""
    opts = dict(BASE_OPTIONS)
    if options:
        opts.update(options)
    if temperature is not None:
        opts["temperature"] = temperature
    opts["num_predict"] = max_tokens

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": opts,
    }
    if system:
        payload["system"] = system

    def _post(body: dict) -> str:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        parts: List[str] = []
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if cancel_check and cancel_check():
                    raise GenerationCancelled()
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
        return "".join(parts)

    try:
        out = _post(payload)
    except urllib.error.HTTPError as e:
        # 구버전 Ollama 또는 think 미지원 모델
        if e.code == 400:
            payload.pop("think", None)
            out = _post(payload)
        else:
            raise
    return strip_thinking(out)


# ---------------------------------------------------------------------------
# 한국어 문장/표현 반복 감지
# ---------------------------------------------------------------------------
_NORM_RE = re.compile(r"[^0-9A-Za-z가-힣]")

# 한 문단에서 두 번 이상 나오면 어색한 문장 시작어
OPENER_WATCH = {
    "이는", "이것은", "또한", "특히", "따라서", "그러므로", "결론적으로", "이처럼", "이러한",
    "즉", "한편", "물론", "그러나", "하지만", "예를", "실제로", "더불어", "나아가", "이에",
}

# 원고에서 바로 지워야 하는 상투 표현
CLICHES = [
    "라고 해도 과언이 아니", "아무리 강조해도 지나치지 않", "시사하는 바가 크", "주목할 만한",
    "빠르게 변화하는 오늘날", "새로운 패러다임", "패러다임의 전환", "중요한 역할을 하고 있",
    "이 절에서는", "이 장에서는", "살펴보겠습니다", "알아보겠습니다", "요약하자면",
    "결론적으로 말하자면", "앞서 언급했듯이", "앞 장에서 언급했듯이",
]

# 과다 사용 집계에서 제외할 흔한 어절
_STOP_EOJEOL = {
    "있습니다", "합니다", "것입니다", "수", "있는", "있다", "하는", "그", "이", "및", "등",
    "더", "것은", "것이", "있으며", "통해", "위해", "대한", "같은", "때문에", "있고",
}


_LIST_LINE_RE = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def is_list_block(paragraph: str) -> bool:
    return any(_LIST_LINE_RE.match(l) for l in paragraph.splitlines())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize(s: str) -> str:
    return _NORM_RE.sub("", s)


def shingles(s: str, n: int = 3) -> set:
    t = normalize(s)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def eojeol_ngrams(text: str, n: int) -> Iterable[str]:
    words = [normalize(w) for w in text.split()]
    words = [w for w in words if w]
    for i in range(len(words) - n + 1):
        gram = words[i:i + n]
        if all(w in _STOP_EOJEOL for w in gram):
            continue
        yield " ".join(gram)


def sentence_ending(sent: str) -> str:
    words = sent.rstrip(".!?…\"' ").split()
    return normalize(words[-1])[-4:] if words else ""


def sentence_opener(sent: str) -> str:
    words = sent.split()
    return normalize(words[0]) if words else ""


def paragraph_issues(paragraph: str) -> List[str]:
    """문단 안의 반복 패턴을 사람이 읽을 수 있는 지적 목록으로 반환."""
    issues: List[str] = []
    sents = split_sentences(paragraph)
    if not sents:
        return issues

    run, prev = 1, None
    for s in sents:
        end = sentence_ending(s)
        if end and end == prev:
            run += 1
            if run == 3:
                issues.append(f"문장 끝맺음 '~{end}'이(가) 연속 3회 이상 반복됨")
        else:
            run = 1
        prev = end

    openers = Counter(sentence_opener(s) for s in sents)
    for op, c in openers.items():
        if c >= 2 and op in OPENER_WATCH:
            issues.append(f"문장 시작어 '{op}'을(를) 한 문단에서 {c}회 사용")

    grams = Counter(eojeol_ngrams(paragraph, 3))
    for g, c in grams.most_common(3):
        if c >= 2:
            issues.append(f"표현 '{g}'이(가) 한 문단에서 {c}회 반복")

    for cl in CLICHES:
        if cl in paragraph:
            issues.append(f"상투 표현 '{cl}' 사용")
    return issues


_STAT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|퍼센트|배|명|개국|곳|만\s*명|억|조|시간|분|일|주|개월)")
_QUOTED_RE = re.compile(r"[‘'\"“「『]([^’'\"”」』\n]{2,24})[’'\"”」』]")
_SOURCE_RE = re.compile(
    r"([가-힣A-Za-z]+)\s*([가-힣A-Za-z]*(?:리뷰|대학교|대학|연구소|연구원|연구진|협회|재단|위원회|센터|보고서|조사|기구|저널))")
_LATIN_RE = re.compile(r"\b[A-Z][A-Za-z&.-]{2,}(?:\s+[A-Z][A-Za-z&.-]+)*")


def fact_markers(sentence: str) -> set:
    """통계 수치, 인용된 고유명, 출처 기관명 등 '같은 사례/근거'를 식별하는 표지."""
    marks = {re.sub(r"\s+", "", m) for m in _STAT_RE.findall(sentence)}
    marks |= {normalize(m) for m in _QUOTED_RE.findall(sentence) if len(normalize(m)) >= 2}
    marks |= {normalize(a + b) for a, b in _SOURCE_RE.findall(sentence) if len(normalize(a + b)) >= 4}
    marks |= {normalize(m).lower() for m in _LATIN_RE.findall(sentence)}
    return {m for m in marks if m}


class RepetitionGuard:
    """책 전체를 기억하면서 새 원고에서 중복 문장을 제거하고 과다 표현을 알려준다."""

    def __init__(self, sim_threshold: float = 0.55, min_len: int = 18, fact_threshold: float = 0.15):
        self.sim_threshold = sim_threshold
        self.min_len = min_len
        self.fact_threshold = fact_threshold
        self._sentences: List[Tuple[str, set]] = []
        self._phrases: Counter = Counter()
        self._openers: Counter = Counter()
        self._facts: dict = {}

    def _is_duplicate(self, sh: set) -> bool:
        for _, prev in self._sentences:
            if jaccard(sh, prev) >= self.sim_threshold:
                return True
        return False

    def _is_recited_fact(self, sentence: str, sh: set) -> bool:
        """앞에서 쓴 통계/출처/사례를 말만 바꿔 다시 꺼낸 문장인지."""
        reused = [m for m in fact_markers(sentence) if m in self._facts]
        if not reused:
            return False
        for m in reused:
            if any(jaccard(sh, prev) >= self.fact_threshold for prev in self._facts[m]):
                return True
        return any(jaccard(sh, prev) >= 0.2 for _, prev in self._sentences)

    def used_facts(self, top: int = 25) -> List[str]:
        """이미 쓴 수치/출처/고유명 (프롬프트의 재사용 금지 목록용)."""
        return list(self._facts.keys())[-top:]

    def filter_new_text(self, text: str) -> Tuple[str, int]:
        """이미 쓴 문장과 거의 같은 문장을 제거한 텍스트와 제거 개수를 반환. (등록은 하지 않음)"""
        removed = 0
        kept_paragraphs: List[str] = []
        local: List[set] = []
        for para in re.split(r"\n\s*\n", text):
            p = para.strip()
            if not p:
                continue
            if p.startswith("#") or p.startswith("|"):
                kept_paragraphs.append(p)
                continue
            is_list = is_list_block(p)
            units = [l.strip() for l in p.splitlines() if l.strip()] if is_list else split_sentences(p)
            kept_units = []
            for s in units:
                if len(normalize(s)) < self.min_len:
                    kept_units.append(s)
                    continue
                sh = shingles(s)
                if (self._is_duplicate(sh) or self._is_recited_fact(s, sh)
                        or any(jaccard(sh, o) >= self.sim_threshold for o in local)):
                    removed += 1
                    continue
                local.append(sh)
                kept_units.append(s)
            if kept_units:
                kept_paragraphs.append(("\n" if is_list else " ").join(kept_units))
        return "\n\n".join(kept_paragraphs), removed

    def add(self, text: str) -> None:
        for para in re.split(r"\n\s*\n", text):
            p = para.strip()
            if not p or p.startswith("#"):
                continue
            for s in split_sentences(p):
                if len(normalize(s)) >= self.min_len:
                    sh = shingles(s)
                    self._sentences.append((s, sh))
                    for m in fact_markers(s):
                        self._facts.setdefault(m, []).append(sh)
                self._openers[sentence_opener(s)] += 1
            for n in (2, 3):
                self._phrases.update(eojeol_ngrams(p, n))

    def dedupe_document(self, text: str) -> Tuple[str, int]:
        """완성된 문서 전체를 앞에서부터 훑으며, 앞서 나온 문장과 거의 같은 문장을 제거한다."""
        out, removed = [], 0
        for para in re.split(r"\n\s*\n", text):
            if not para.strip():
                continue
            kept, n = self.filter_new_text(para)
            removed += n
            if kept.strip():
                self.add(kept)
                out.append(kept)
        return "\n\n".join(out), removed

    def overused_phrases(self, top: int = 12, min_count: int = 3) -> List[str]:
        out = []
        for g, c in self._phrases.most_common(200):
            if c < min_count:
                break
            if any(g in o for o in out):
                continue
            out.append(g)
            if len(out) >= top:
                break
        openers = [o for o, c in self._openers.most_common(8) if c >= 4 and o in OPENER_WATCH]
        return out + [f"{o} (문장 시작어)" for o in openers]
