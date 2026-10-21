#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lyrics_writer.py - 한국어 가사 작사 엔진 (ACE-Step 1.5 가창용)

1. 장르별 작법 가이드(템포, 줄당 음절, 구조 태그)를 프롬프트에 주입
2. LLM은 고유 섹션(Verse, Pre-Chorus, Chorus, Bridge...)만 한 번씩 쓰고,
   후렴 반복과 곡 배치는 코드가 조립 -> 모든 후렴이 정확히 같은 훅으로 반복된다
3. 후보 여러 개를 음절 수, 상투어, 줄 중복 기준으로 채점해 최선을 고르고, 지적 사항만 퇴고
"""

import re
from typing import Dict, List, Optional, Tuple

from text_engine import ollama_generate, strip_thinking

GENRE_LYRIC_GUIDE: Dict[str, dict] = {
    "ballad_acoustic": {
        "name": "감성 어쿠스틱 발라드",
        "bpm": 68, "keyscale": "C Major", "syllables": (6, 9),
        "tone": "담담하게 시작해 후렴에서 감정이 차오르는 고백. 일상의 작은 장면으로 이별/그리움을 보여준다",
        "english": "영어를 쓰지 않는다",
        "tags": {"Intro": "Intro - acoustic guitar", "Bridge": "Bridge - soft",
                 "Final Chorus": "Final Chorus - powerful", "Outro": "Outro - fade out"},
    },
    "ballad_orchestral": {
        "name": "오케스트라 팝 발라드",
        "bpm": 72, "keyscale": "D Major", "syllables": (6, 9),
        "tone": "영화 같은 장면 전환, 절정에서 크게 터지는 고음 후렴",
        "english": "영어를 쓰지 않는다",
        "tags": {"Intro": "Intro - piano", "Bridge": "Bridge - building energy",
                 "Final Chorus": "Final Chorus - powerful belting", "Outro": "Outro - piano"},
    },
    "kpop_dance": {
        "name": "K-POP 댄스 팝",
        "bpm": 122, "keyscale": "A Minor", "syllables": (5, 9),
        "tone": "자신감 있고 리드미컬한 말맛. 짧게 끊어 치는 줄, 따라 부르기 쉬운 반복 훅",
        "english": "후렴 훅에 짧은 영어 구절 1~2개까지 허용. 나머지는 한국어",
        "tags": {"Intro": "Intro - synth", "Bridge": "Breakdown",
                 "Final Chorus": "Final Chorus - high energy", "Outro": "Outro"},
    },
    "kpop_rnb": {
        "name": "K-POP R&B / 어반 그루브",
        "bpm": 90, "keyscale": "F Minor", "syllables": (6, 9),
        "tone": "늦은 밤의 은근한 긴장감. 속삭이듯 말하는 화자, 감각적인 디테일",
        "english": "짧은 영어 구절을 각 섹션에 최대 1줄까지 허용",
        "tags": {"Intro": "Intro - rhodes piano", "Bridge": "Bridge - whispered",
                 "Final Chorus": "Final Chorus - harmonies", "Outro": "Outro - fade out"},
    },
    "kpop_rock": {
        "name": "K-POP 모던 록 발라드",
        "bpm": 80, "keyscale": "E Minor", "syllables": (6, 10),
        "tone": "무너졌다가 다시 일어서는 의지. 거칠고 직선적인 말투",
        "english": "영어는 마지막 한 줄 정도만 허용",
        "tags": {"Intro": "Intro - electric guitar", "Bridge": "Guitar Solo",
                 "Final Chorus": "Final Chorus - powerful", "Outro": "Outro"},
    },
}
GENRE_LYRIC_GUIDE.update({
    "ballad_piano": {
        "name": "피아노 발라드",
        "bpm": 66, "keyscale": "Eb Major", "syllables": (6, 9),
        "tone": "조용한 독백으로 시작해 후렴에서 참았던 말을 꺼내는 고백. 방 안, 창가, 버스 정류장 같은 작은 장면",
        "english": "영어를 쓰지 않는다",
        "tags": {"Intro": "Intro - solo piano", "Bridge": "Bridge - quiet then rising",
                 "Final Chorus": "Final Chorus - powerful high notes", "Outro": "Outro - solo piano"},
    },
    "ballad_rnb": {
        "name": "R&B 발라드",
        "bpm": 74, "keyscale": "Ab Major", "syllables": (6, 10),
        "tone": "담백하게 속삭이다 후렴에서 감정을 실어 꺾는 창법. 대화하듯 2인칭으로 말한다",
        "english": "후렴에 짧은 영어 구절 1개까지 허용",
        "tags": {"Intro": "Intro - electric piano", "Bridge": "Bridge - vocal runs",
                 "Final Chorus": "Final Chorus - ad-libs", "Outro": "Outro - fade out"},
    },
    "ost_drama": {
        "name": "드라마 OST 발라드",
        "bpm": 70, "keyscale": "D Major", "syllables": (6, 9),
        "tone": "두 사람의 엇갈림과 기다림. 장면이 눈앞에 그려지는 서사, 후렴은 누구나 따라 부를 쉬운 말",
        "english": "영어를 쓰지 않는다",
        "tags": {"Intro": "Intro - piano and strings", "Bridge": "Bridge - building strings",
                 "Final Chorus": "Final Chorus - full orchestra", "Outro": "Outro - strings fade"},
    },
    "indie_folk": {
        "name": "인디 포크",
        "bpm": 92, "keyscale": "G Major", "syllables": (6, 10),
        "tone": "일기처럼 솔직하고 소소한 일상. 계절, 골목, 자전거 같은 구체적 사물",
        "english": "영어를 쓰지 않는다",
        "tags": {"Intro": "Intro - acoustic guitar", "Bridge": "Bridge - humming",
                 "Final Chorus": "Final Chorus - group vocals", "Outro": "Outro - acoustic guitar"},
    },
    "citypop": {
        "name": "시티팝",
        "bpm": 104, "keyscale": "F Major", "syllables": (6, 9),
        "tone": "도시의 밤과 드라이브, 아련하지만 경쾌한 그리움",
        "english": "후렴 훅에 짧은 영어 구절 1~2개까지 허용",
        "tags": {"Intro": "Intro - funky guitar", "Bridge": "Bridge - synth solo",
                 "Final Chorus": "Final Chorus", "Outro": "Outro - fade out"},
    },
})
GENRE_LYRIC_GUIDE["custom"] = dict(GENRE_LYRIC_GUIDE["ballad_acoustic"], name="자유 장르")

# ACE-Step 캡션: 장르별 편곡/음색 묘사 ({vocal} 자리에 female/male vocals 등이 들어간다)
QUALITY_TAIL = ("smooth legato {vocal}, in-tune singing, natural phrasing, clear Korean diction, "
                "gentle vibrato, breath control, studio mix, no shouting, no robotic timing")

GENRE_CAPTION = {
    "ballad_piano": "Korean piano ballad, intimate emotional {vocal}, grand piano, soft strings, slow tempo, close-mic verses, soaring but controlled chorus",
    "ballad_acoustic": "Korean acoustic ballad, warm {vocal}, fingerpicked acoustic guitar, soft piano, light strings, gentle and sentimental",
    "ballad_orchestral": "Korean orchestral pop ballad, powerful yet controlled {vocal}, grand piano, cinematic strings, big drums in final chorus",
    "ballad_rnb": "Korean R&B ballad, smooth soulful {vocal}, electric piano, warm bass, soft drums, tasteful vocal runs, late night mood",
    "ost_drama": "Korean drama OST ballad, heartfelt {vocal}, piano and strings, cinematic, emotional climax, crystal-clear diction",
    "kpop_rock": "Korean rock ballad, passionate {vocal}, overdrive electric guitar, live drums, emotional soaring climax",
    "indie_folk": "Korean indie folk, gentle {vocal}, acoustic guitar, soft percussion, warm and intimate",
    "citypop": "Korean city pop, smooth {vocal}, groovy bass, clean electric guitar, synth pads, nostalgic 80s",
    "kpop_rnb": "Korean urban R&B, smooth {vocal}, groovy bassline, Rhodes piano, crisp snaps, late night mood",
    "kpop_dance": "Korean K-pop dance pop, bright {vocal}, punchy drums, synth bass, catchy hook, high energy",
    "custom": "Korean pop song, expressive {vocal}, polished band arrangement, catchy melodic chorus",
}

# 화면에 보여줄 장르 순서 (그룹, 키)
GENRE_MENU = [
    ("발라드", ["ballad_piano", "ballad_acoustic", "ballad_orchestral", "ballad_rnb", "ost_drama", "kpop_rock"]),
    ("그 밖의 장르", ["indie_folk", "citypop", "kpop_rnb", "kpop_dance"]),
]

MOOD_CAPTION = {
    "애절한": "heartbreaking", "그리운": "nostalgic", "쓸쓸한": "lonely melancholic", "담담한": "calm and restrained",
    "따뜻한": "warm and tender", "설레는": "fluttering romantic", "희망찬": "hopeful uplifting", "웅장한": "epic and grand",
}


def build_caption(genre_key: str, gender: str = "", moods=(), extra: str = "") -> str:
    vocal = f"{gender} vocals" if gender else "lead vocals"
    cap = GENRE_CAPTION.get(genre_key, GENRE_CAPTION["custom"]).format(vocal=vocal)
    mood_words = [MOOD_CAPTION[m] for m in moods if m in MOOD_CAPTION]
    if mood_words:
        cap += ", " + ", ".join(mood_words)
    if extra.strip():
        cap += ", " + extra.strip().strip(",")
    cap += ", " + QUALITY_TAIL.format(vocal=vocal)
    return cap

# (고유 섹션, 줄 수) - LLM이 쓰는 단위
UNIQUE_SECTIONS = {
    "short": [("Verse 1", 4), ("Pre-Chorus", 2), ("Chorus", 4), ("Outro", 1)],
    "medium": [("Verse 1", 4), ("Pre-Chorus", 2), ("Chorus", 4), ("Verse 2", 4), ("Outro", 1)],
    "full": [("Verse 1", 4), ("Pre-Chorus", 2), ("Chorus", 4), ("Verse 2", 4), ("Bridge", 2),
             ("Final Line", 1), ("Outro", 1)],
}

# 곡 배치 (표시 태그 키, 가사 출처 섹션)
ARRANGEMENT = {
    "short": [("Intro", None), ("Verse 1", "Verse 1"), ("Pre-Chorus", "Pre-Chorus"),
              ("Chorus", "Chorus"), ("Outro", "Outro")],
    "medium": [("Intro", None), ("Verse 1", "Verse 1"), ("Pre-Chorus", "Pre-Chorus"),
               ("Chorus", "Chorus"), ("Verse 2", "Verse 2"), ("Pre-Chorus", "Pre-Chorus"),
               ("Chorus", "Chorus"), ("Outro", "Outro")],
    "full": [("Intro", None), ("Verse 1", "Verse 1"), ("Pre-Chorus", "Pre-Chorus"),
             ("Chorus", "Chorus"), ("Verse 2", "Verse 2"), ("Pre-Chorus", "Pre-Chorus"),
             ("Chorus", "Chorus"), ("Bridge", "Bridge"), ("Final Chorus", "Final Chorus"),
             ("Outro", "Outro")],
}

CLICHE_WORDS = ["그리움", "눈물", "추억", "영원히", "영원한", "운명", "기적", "별빛", "아픔",
                "슬픔", "행복", "사랑해", "가슴이", "심장이", "너만을", "아련", "시린"]

SECTION_ALIASES = {
    "intro": "Intro", "인트로": "Intro",
    "verse1": "Verse 1", "verse": "Verse 1", "1절": "Verse 1", "벌스1": "Verse 1",
    "verse2": "Verse 2", "2절": "Verse 2", "벌스2": "Verse 2",
    "prechorus": "Pre-Chorus", "프리코러스": "Pre-Chorus", "프리후렴": "Pre-Chorus",
    "chorus": "Chorus", "후렴": "Chorus", "코러스": "Chorus", "hook": "Chorus", "훅": "Chorus",
    "bridge": "Bridge", "브릿지": "Bridge", "브리지": "Bridge",
    "finalline": "Final Line", "마지막줄": "Final Line", "finalchorus": "Final Chorus",
    "outro": "Outro", "아웃트로": "Outro",
}

LYRICIST_SYSTEM = """당신은 한국 대중가요 히트곡을 여럿 쓴 작사가입니다. 소리 내어 불렀을 때 자연스러운 가사를 씁니다.

[작사 원칙]
- 한 곡에는 중심 이미지(은유) 하나만 쓰고, 그 이미지를 섹션마다 다른 각도로 변주한다. 이미지를 섞지 않는다.
- 그리움, 슬픔, 운명, 영원 같은 추상어를 나열하지 말고 구체적인 장면, 사물, 행동으로 감정을 보여준다.
- 형용사를 쌓지 않는다.
- 한 줄은 한 숨에 자연스럽게 부를 길이로 쓰고, 같은 섹션 안의 줄은 음절 수를 비슷하게 맞춘다. 혀가 꼬이는 말, 막히는 받침 연속은 피한다.
- 줄 끝 모음이나 받침을 맞춰 운율을 만든다.
- Verse는 이야기를 진행하고, Pre-Chorus는 긴장을 쌓고, Chorus는 곡의 제목이 될 한 줄 훅을 첫 줄이나 마지막 줄에 둔다.
- Verse 2는 Verse 1과 다른 시간, 다른 장면으로 이야기를 전진시키고 Verse 1의 줄을 재사용하지 않는다.
- 마침표와 쉼표는 쓰지 않는다. 배경 코러스는 (괄호)로 짧게만 쓴다.
- 구조 태그와 가사 줄만 출력한다. 제목, 해설, 번호, 이모지, 음절 수 표기를 쓰지 않는다."""


def get_guide(genre_key: str) -> dict:
    return GENRE_LYRIC_GUIDE.get(genre_key) or GENRE_LYRIC_GUIDE["ballad_acoustic"]


def song_form(duration: Optional[float]) -> str:
    if duration and duration <= 75:
        return "short"
    if duration and duration <= 150:
        return "medium"
    return "full"


def count_syllables(line: str) -> int:
    hangul = len(re.findall(r"[가-힣]", line))
    latin = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())))
                for w in re.findall(r"[A-Za-z']+", line))
    return hangul + latin


def _canonical_section(label: str) -> Optional[str]:
    base = re.split(r"\s+[-–]\s*|\s*[–:]\s*", label.strip())[0]
    key = re.sub(r"[\s\-_]", "", base).lower()
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key]
    m = re.match(r"(verse|벌스)(\d)", key)
    if m:
        return f"Verse {m.group(2)}"
    return None


_TAG_RE = re.compile(r"^[\[\(（]?\s*\**\s*([A-Za-z가-힣\- 0-9]+?)\s*\**\s*[\]\)）]?\s*:?\s*(\(\d+줄\))?$")


def parse_sections(raw: str) -> Dict[str, List[str]]:
    """LLM 출력에서 섹션별 가사 줄을 추출한다. 같은 섹션이 반복되면 첫 번째만 사용."""
    text = strip_thinking(raw)
    text = re.sub(r"```[a-z]*", "", text)
    sections: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        s = line.strip().strip("*#> ").strip()
        if not s:
            continue
        m = _TAG_RE.match(s)
        if m and (s.startswith(("[", "(", "（")) or s.endswith(":") or len(s) <= 16):
            name = _canonical_section(m.group(1))
            if name:
                current = name if name not in sections else None
                if current:
                    sections[current] = []
                continue
        if current is None:
            continue
        if re.match(r"^(※|참고|설명|해설|제목|title|note)", s, re.I) or "음절" in s or "BPM" in s:
            continue
        if re.fullmatch(r"[\(（].{12,}[\)）]", s):
            continue
        s = re.sub(r"^\d+[.)]\s*", "", s)
        s = re.sub(r"^[-•]\s*", "", s)
        s = re.sub(r"[.,。、]+$", "", s).replace(",", "")
        if s:
            sections[current].append(s)
    return {k: v for k, v in sections.items() if v}


def _line_key(line: str) -> str:
    return re.sub(r"[\s,.!?~…'\"]+", "", line)


def _near_same(a: str, b: str) -> bool:
    ka, kb = _line_key(a), _line_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ga = {ka[i:i + 2] for i in range(len(ka) - 1)}
    gb = {kb[i:i + 2] for i in range(len(kb) - 1)}
    return bool(ga and gb) and len(ga & gb) / len(ga | gb) >= 0.7


def drop_repeated_lines(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """같은 섹션 안에서 바로 앞줄과 (거의) 같은 줄, Outro 가 후렴 줄을 그대로 되풀이하는 줄을 걷어낸다."""
    cleaned: Dict[str, List[str]] = {}
    chorus = sections.get("Chorus", []) + sections.get("Final Line", [])
    for name, lines in sections.items():
        kept: List[str] = []
        for ln in lines:
            if kept and _near_same(kept[-1], ln):
                continue
            if name == "Outro" and any(_near_same(ln, c) for c in chorus):
                continue
            kept.append(ln)
        cleaned[name] = kept
    return cleaned


def assemble(sections: Dict[str, List[str]], form: str, guide: dict) -> str:
    out: List[str] = []
    tags = guide.get("tags", {})
    sections = drop_repeated_lines(sections)
    for tag_key, src in ARRANGEMENT[form]:
        label = tags.get(tag_key, tag_key)
        if src is None:
            out.append(f"[{label}]")
            out.append("")
            continue
        if src == "Final Chorus":
            lines = list(sections.get("Chorus", []))
            final = sections.get("Final Line", [])
            if lines and final:
                lines[-1] = final[0]
        else:
            lines = sections.get(src, [])
        if not lines:
            if src == "Outro":
                out += [f"[{label}]", ""]
            continue
        out.append(f"[{label}]")
        out.extend(lines)
        out.append("")
    return "\n".join(out).strip() + "\n"


def score_sections(sections: Dict[str, List[str]], form: str, guide: dict) -> Tuple[int, List[str]]:
    score, notes = 100, []
    lo, hi = guide["syllables"]
    for name, n_lines in UNIQUE_SECTIONS[form]:
        lines = sections.get(name)
        if not lines:
            penalty = 40 if name in ("Verse 1", "Chorus") else 10
            score -= penalty
            notes.append(f"[{name}] 섹션 누락")
            continue
        if abs(len(lines) - n_lines) > 0:
            score -= 2 * abs(len(lines) - n_lines)
            notes.append(f"[{name}]은(는) {n_lines}줄이어야 하는데 {len(lines)}줄")
        for ln in lines:
            syl = count_syllables(ln)
            if syl < lo - 1 or syl > hi + 2:
                score -= 3
                notes.append(f"'{ln}' -> {syl}음절 ({lo}~{hi}음절로 조정)")

    body = "\n".join(l for k, v in sections.items() for l in v)
    cliche_hits = [w for w in CLICHE_WORDS if w in body]
    if len(cliche_hits) > 2:
        score -= 3 * (len(cliche_hits) - 2)
        notes.append("상투어 과다: " + ", ".join(cliche_hits) + " -> 구체적 장면/사물로 교체")

    seen = {}
    for name, lines in sections.items():
        for i, ln in enumerate(lines):
            key = _line_key(ln)
            if key in seen:
                where = f"[{name}] 안에서" if seen[key] == name else f"[{seen[key]}]와 [{name}]에"
                score -= 6
                notes.append(f"'{ln}'이(가) {where} 중복 -> 새 줄로 바꿀 것")
            elif i > 0 and _near_same(lines[i - 1], ln):
                score -= 4
                notes.append(f"'{lines[i - 1]}' / '{ln}' 연속된 두 줄이 거의 같음 -> 하나를 바꿀 것")
            seen.setdefault(key, name)

    v1, v2 = sections.get("Verse 1", []), sections.get("Verse 2", [])
    if v1 and v2 and len(set(v1) & set(v2)) > 0:
        score -= 5
        notes.append("Verse 2가 Verse 1의 줄을 재사용함")

    if "영어를 쓰지 않는다" in guide.get("english", ""):
        eng = [l for l in body.splitlines() if re.search(r"[A-Za-z]{3,}", l)]
        if eng:
            score -= 3 * len(eng)
            notes.append("영어 줄을 한국어로 바꿀 것: " + " / ".join(eng[:3]))
    return score, notes


def build_prompt(topic: str, guide: dict, form: str, extra: str = "") -> str:
    lo, hi = guide["syllables"]
    spec_lines = []
    for name, n in UNIQUE_SECTIONS[form]:
        if name == "Chorus":
            spec_lines.append(f"[Chorus]  ({n}줄, 곡의 훅. 이 후렴이 곡 전체에서 그대로 반복된다)")
        elif name == "Final Line":
            spec_lines.append("[Final Line]  (1줄. 마지막 후렴의 마지막 줄을 대신해 감정을 최고조로 끌어올리는 한 줄)")
        elif name == "Bridge":
            spec_lines.append(f"[Bridge]  ({n}줄. 시점이나 감정이 뒤집히는 전환)")
        else:
            spec_lines.append(f"[{name}]  ({n}줄)")
    spec = "\n".join(spec_lines)
    return f"""[주제] {topic}
[장르] {guide['name']} (약 {guide['bpm']} BPM)
[분위기와 화자] {guide['tone']}
[영어 사용] {guide['english']}
[줄당 음절] 한 줄 {lo}~{hi}음절
{('[추가 요청] ' + extra) if extra else ''}

아래 섹션을 이 순서대로 한 번씩만 쓰세요. 태그 줄은 대괄호 그대로 쓰고, 괄호 안 설명은 출력하지 마세요.
{spec}

출력 예시 형식:
[Verse 1]
가사 줄
가사 줄"""


def _revise_prompt(raw_sections: Dict[str, List[str]], notes: List[str], form: str, guide: dict) -> str:
    current = "\n\n".join(f"[{k}]\n" + "\n".join(v) for k, v in raw_sections.items())
    lo, hi = guide["syllables"]
    return f"""아래 가사를 퇴고하세요. 지적된 문제만 고치고 나머지 좋은 줄은 그대로 둡니다.

[지적 사항]
{chr(10).join('- ' + n for n in notes[:12])}

[규칙] 한 줄 {lo}~{hi}음절, 섹션 태그와 순서와 줄 수는 그대로, 가사만 출력.

[현재 가사]
{current}"""


def generate_lyrics(topic: str, genre_key: str = "ballad_acoustic", duration: Optional[float] = None,
                    model: str = "qwen3:14b", port: int = 11434, candidates: int = 2,
                    extra: str = "", log=print) -> str:
    guide = get_guide(genre_key)
    form = song_form(duration)
    prompt = build_prompt(topic, guide, form, extra)
    # 후렴 반복을 코드가 처리하므로 반복 억제 페널티는 낮게 둔다
    opts = {"repeat_penalty": 1.05, "presence_penalty": 0.2, "frequency_penalty": 0.2}

    best = None
    for i in range(max(1, candidates)):
        log(f"✍️ 가사 초안 {i + 1}/{candidates} 작성 중...")
        raw = ollama_generate(prompt, model=model, port=port, system=LYRICIST_SYSTEM,
                              max_tokens=1500, temperature=0.9, options=opts)
        secs = parse_sections(raw)
        sc, notes = score_sections(secs, form, guide)
        log(f"   초안 {i + 1} 점수 {sc}")
        if best is None or sc > best[0]:
            best = (sc, secs, notes)

    sc, secs, notes = best
    if notes and sc < 90:
        log("🔧 지적 사항 퇴고 중...")
        raw = ollama_generate(_revise_prompt(secs, notes, form, guide), model=model, port=port,
                              system=LYRICIST_SYSTEM, max_tokens=1500, temperature=0.6, options=opts)
        new_secs = parse_sections(raw)
        new_sc, _ = score_sections(new_secs, form, guide)
        log(f"   퇴고본 점수 {new_sc} (기존 {sc})")
        if new_sc >= sc:
            secs, sc = new_secs, new_sc

    if not secs.get("Chorus") and not secs.get("Verse 1"):
        raise RuntimeError("가사 생성 결과에서 Verse/Chorus 구조를 찾지 못했습니다. 다시 시도해 주세요.")
    lyrics = assemble(secs, form, guide)
    log(f"✅ 가사 완성 (점수 {sc}, 예상 길이 약 {estimate_duration_sec(lyrics, guide['bpm'])}초)")
    return lyrics


def estimate_duration_sec(lyrics: str, bpm: int = 80) -> int:
    """가사 줄 수와 템포로 곡 길이를 추정한다. 한 줄 = 2마디(8박), 무가사 섹션 = 4마디."""
    beat = 60.0 / max(40, min(200, bpm or 80))
    lines = sections = empty_sections = 0
    current_has_lines = True
    for raw in lyrics.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("["):
            if not current_has_lines:
                empty_sections += 1
            sections += 1
            current_has_lines = False
        else:
            lines += 1
            current_has_lines = True
    if not current_has_lines:
        empty_sections += 1
    total_beats = lines * 8 + empty_sections * 16 + sections * 2
    return int(max(30, min(300, total_beats * beat)))
