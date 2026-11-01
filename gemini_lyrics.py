"""gemini_lyrics.py - Gemini 가사 생성 클라이언트 (최신 모델 대응 REST 호출)

이 모듈이 고친 실제 문제
1) 기본 모델이 gemini-1.5-flash 였다. 1.5 계열은 이미 서비스 종료라 404 만 돌아온다.
   -> 최신 세대 모델을 기본값으로 쓰고, 종료된 모델명은 자동으로 현행 모델로 바꾼다.
2) 종료된 모델이 404 로 실패하면 그대로 예외로 끝났다.
   -> 404/400(모델 미지원) 이면 대체 모델로 한 번 더 시도한다.
3) Gemini 3/4 계열은 temperature 를 1.0 아래로 낮추면 루프/품질 저하가 공식 문서에 명시돼 있다.
   -> 추론 계열에서는 temperature 를 1.0 으로 고정하고, 창의성/정밀도는 thinkingLevel 로 조절한다.
4) 사고(thought) 토큰이 출력 한도를 잡아먹어 MAX_TOKENS 로 잘리던 문제.
   -> 출력 한도를 넉넉히 주고, thinkingLevel 을 명시하며, 잘리면 원인을 알려준다.

키는 디스크/URL/오류 메시지에 남기지 않는다.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

DEFAULT_API_VERSION = "v1beta"

# 현행 모델 (Google Generative Language API 실서비스 기준)
DEFAULT_MODEL = "gemini-2.0-flash"
# 기본 모델이 404/400 이면 순서대로 시도
FALLBACK_MODELS = ("gemini-1.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro", "gemini-3.8-flash")

# 레거시 모델명 치환
RETIRED_MODELS = {
    "gemini-pro": DEFAULT_MODEL,
    "gemini-1.0-pro": DEFAULT_MODEL,
    "gemini-1.5-flash-latest": "gemini-1.5-flash",
    "gemini-1.5-pro-latest": "gemini-1.5-pro",
}
MODEL_ALIASES = RETIRED_MODELS  # 이전 버전과의 호환 이름

THINKING_LEVELS = ("minimal", "low", "medium", "high")
MAX_OUTPUT_TOKENS = 16384

# 실제로 응답을 돌려준 모델 (UI 로그/증거 기록용)
LAST_MODEL = DEFAULT_MODEL


def api_version() -> str:
    return (os.environ.get("GEMINI_API_VERSION") or DEFAULT_API_VERSION).strip().strip("/")


def resolve_model(model=None) -> str:
    """모델명 정규화: models/ 접두사 제거, 종료 모델은 현행 모델로 치환."""
    name = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL)
    name = str(name).strip().removeprefix("models/").strip()
    return RETIRED_MODELS.get(name, name)


def _is_reasoning_model(model: str) -> bool:
    """Gemini 3 세대 이상은 temperature 기본값(1.0)을 쓰도록 권고된다."""
    return bool(re.match(r"gemini-(2\.5|3|4)", model or ""))


def _effective_temperature(model: str, temperature=None) -> float:
    if temperature is None:
        return 1.0 if _is_reasoning_model(model) else 0.8
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        value = 1.0 if _is_reasoning_model(model) else 0.8
    value = max(0.0, min(2.0, value))
    # 문서 권고: Gemini 3 계열에서 1.0 미만은 루프/품질 저하를 부를 수 있다.
    if _is_reasoning_model(model) and value < 1.0:
        return 1.0
    return value


def _thinking_level(model: str, level) -> str:
    if not _is_reasoning_model(model):
        return ""
    level = str(level or "").strip().lower()
    if level not in THINKING_LEVELS:
        return "high"
    if model == "gemini-3.1-pro-preview" and level == "minimal":
        return "medium"  # 3.1 Pro 는 minimal 미지원
    return level


def _build_body(prompt, system, model, temperature, max_output_tokens,
                thinking_level, seed, presence_penalty, frequency_penalty):
    generation_config = {
        "maxOutputTokens": int(max_output_tokens),
        "temperature": _effective_temperature(model, temperature),
    }
    level = _thinking_level(model, thinking_level)
    if level:
        # generateContent(Gemini 3) 는 thinkingConfig.thinkingLevel 을 enum 문자열로 받는다.
        generation_config["thinkingConfig"] = {"thinkingLevel": level.upper()}
    if seed is not None:
        generation_config["seed"] = int(seed)
    if presence_penalty is not None:
        generation_config["presencePenalty"] = float(presence_penalty)
    if frequency_penalty is not None:
        generation_config["frequencyPenalty"] = float(frequency_penalty)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def _error_message(exc, key: str) -> str:
    err_msg = ""
    try:
        raw_bytes = exc.read() if hasattr(exc, "read") else b""
        if raw_bytes:
            err_json = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            err_msg = (err_json.get("error") or {}).get("message", "") or ""
    except Exception:  # noqa: BLE001
        pass
    if key and err_msg:
        err_msg = err_msg.replace(key, "******")
    return err_msg


def _request(model: str, body: dict, key: str, timeout: int):
    url = (f"https://generativelanguage.googleapis.com/{api_version()}"
           f"/models/{model}:generateContent")
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def generate(prompt, system, api_key=None, model=None, temperature=None,
             thinking_level="high", max_output_tokens=MAX_OUTPUT_TOKENS,
             seed=None, presence_penalty=None, frequency_penalty=None,
             timeout=180, **_):
    global LAST_MODEL
    key = (api_key or os.environ.get("GEMINI_API_KEY") or
           os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Gemini API 키를 입력하거나 GEMINI_API_KEY 환경변수를 설정하세요.")
    primary = resolve_model(model)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", primary):
        raise ValueError("Gemini 모델 이름 형식이 올바르지 않습니다.")

    chain = [primary] + [m for m in FALLBACK_MODELS if m != primary]
    body = _build_body(prompt, system, primary, temperature, max_output_tokens,
                       thinking_level, seed, presence_penalty, frequency_penalty)
    body_plain = _build_body(prompt, system, primary, temperature, max_output_tokens,
                             "", seed, presence_penalty, frequency_penalty)

    last_error = None
    for model_name in chain:
        payload = dict(body)
        for attempt in range(3):
            try:
                result = _request(model_name, payload, key, timeout)
                LAST_MODEL = model_name
                return _extract_text(result, model_name)
            except urllib.error.HTTPError as exc:
                message = _error_message(exc, key)
                # thinkingConfig 미지원 모델: 파라미터를 빼고 같은 모델로 한 번 더
                if exc.code == 400 and "thinkingConfig" in json.dumps(payload) \
                        and "thinking" in (message or "").lower():
                    payload = dict(body_plain)
                    continue
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep((2 ** attempt) + 0.5)
                    continue
                hints = {400: "요청 또는 모델 설정 확인", 401: "API 키 인증 실패",
                         403: "API 키 권한 확인", 404: "사용 가능한 모델 이름 확인",
                         429: "요청 한도 초과. 잠시 후 다시 시도"}
                if exc.code in (400, 404):
                    last_error = (f"Gemini HTTP {exc.code} ({model_name})"
                                  + (f": {message}" if message else f": {hints.get(exc.code)}"))
                    break  # 다음 대체 모델로
                detail = f" ({message})" if message else f": {hints.get(exc.code, '서비스 오류')}"
                raise RuntimeError(f"Gemini HTTP {exc.code}{detail}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < 1:
                    time.sleep(1.0)
                    continue
                raise RuntimeError("Gemini 연결 실패: 네트워크 연결을 확인하세요.") from None
        else:
            continue
        continue
    raise RuntimeError(last_error or "Gemini 호출에 실패했습니다. 모델 이름과 키 권한을 확인하세요.")


def _extract_text(result: dict, model: str) -> str:
    candidates = result.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    finish = candidate.get("finishReason")
    if finish not in (None, "STOP"):
        if finish == "MAX_TOKENS":
            raise RuntimeError(f"Gemini 응답이 출력 한도에서 잘렸습니다 ({model}). "
                               "주제를 좁히거나 분량을 줄여 다시 시도하세요.")
        raise RuntimeError(f"Gemini 응답이 중단되었습니다 (finishReason={finish}). "
                           "주제나 출력 분량을 조정하세요.")
    parts = candidate.get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    if not text:
        raise RuntimeError("Gemini에서 가사를 받지 못했습니다. 안전 차단 또는 빈 응답입니다.")
    return text


def describe(model=None) -> str:
    """UI 표시용 한 줄 설명."""
    name = resolve_model(model)
    extra = " (temperature 1.0 고정, thinkingLevel 사용)" if _is_reasoning_model(name) else ""
    return f"{name}{extra}"
