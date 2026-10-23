"""Gemini REST lyric client. Keys are never persisted or included in URLs/errors."""
import json
import os
import time
import urllib.error
import urllib.request
import re

DEFAULT_MODEL = "gemini-2.5-flash"


def generate(prompt, system, api_key=None, model=None, temperature=0.8, **_):
    key = (api_key or os.environ.get("GEMINI_API_KEY") or
           os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Gemini API 키를 입력하거나 GEMINI_API_KEY 환경변수를 설정하세요.")
    model = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL).removeprefix("models/")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("Gemini 모델 이름 형식이 올바르지 않습니다.")
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            messages = {400: "요청 또는 모델 설정 확인", 401: "API 키 인증 실패",
                        403: "API 키 권한 확인", 404: "사용 가능한 모델 이름 확인",
                        429: "요청 한도 초과. 잠시 후 다시 시도"}
            raise RuntimeError(f"Gemini HTTP {exc.code}: {messages.get(exc.code, '서비스 오류')}") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Gemini 연결 실패: 네트워크 연결을 확인하세요.") from None
    candidates = result.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    if candidate.get("finishReason") not in (None, "STOP"):
        raise RuntimeError("Gemini 응답이 중단되었습니다. 주제나 출력 분량을 조정하세요.")
    parts = candidate.get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("Gemini에서 가사를 받지 못했습니다. 안전 차단 또는 빈 응답입니다.")
    return text
