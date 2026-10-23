# -*- coding: utf-8 -*-
"""
ebook_persona.py - 전자책 집필용 '장르 페르소나' + 장편(최대 300장) 구조 설계

근거
 - Shanahan et al., "Role play with large language models", Nature 2023
   → 페르소나는 시스템 프롬프트로 부여한 '역할'이며, 역할이 구체적일수록 어휘·문체가 일관된다.
 - Li et al., "High-Dimensional Programmable Personas" (IEEE Access 2026)
   → 페르소나는 (직업/시대/가치관/금기/문체)처럼 **다차원 속성**으로 명시할 때
     비구조적 한 줄 지시보다 통제가 잘 된다. → 아래 PERSONA 스키마는 그 6개 축을 쓴다.
 - Yang et al., "Re3: Generating Longer Stories with Recursive Reprompting and Revision" (EMNLP 2022)
   → 계획(plan)을 고정하고, 매 장마다 (계획 + 지금까지의 상태)로 다시 프롬프트하며,
     끝에 수정(revision) 패스를 둔다. → plan/state/revise 3단 구조.
 - Yang et al., "DOC: Improving Long Story Coherence with Detailed Outline Control" (ACL 2023)
   → 상세 아웃라인(장별 비트)이 장문 일관성을 크게 올린다. → BEAT 단위 계획.
 - "Scaling Long-Form Story Generation via Narrative State Tracking" (arXiv 2026)
   → 인물/장소/소지품/시간을 'typed state'로 추적하면 설정 충돌이 줄어든다. → StoryState.
 - "Integrating planning into single-turn long-form text generation" (arXiv 2024)
   → 집필 전(pre-writing) 단계 분리가 품질을 올린다.

이 모듈은 순수 데이터/프롬프트 빌더다(모델 호출 없음) → 그대로 테스트 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- 페르소나 축
PERSONA_AXES = ("role", "era", "values", "voice", "taboo", "reader_promise")


@dataclass
class Persona:
    """6축 페르소나. 모든 필드는 '한 문장' 지시로 쓰인다."""
    name: str
    role: str                 # 직업/정체성 : "30년 경력의 경제 기자"
    era: str                  # 시대/공간  : "2026년 서울"
    values: str               # 가치관      : "숫자보다 사람의 선택을 먼저 본다"
    voice: str                # 문체        : "짧은 문장, 구체적 숫자, 마지막에 질문"
    taboo: str                # 금기        : "근거 없는 전망, 감탄 위주의 수사"
    reader_promise: str = ""  # 독자 약속   : "읽고 나면 ~할 수 있다"

    def system_prompt(self) -> str:
        lines = [
            f"당신은 '{self.name}'입니다.",
            f"[정체성] {self.role}. [시대/공간] {self.era}.",
            f"[가치관] {self.values}.",
            f"[문체] {self.voice}.",
            f"[금기] {self.taboo}는 절대 쓰지 않습니다.",
        ]
        if self.reader_promise:
            lines.append(f"[독자 약속] 이 책을 덮은 독자는 {self.reader_promise}.")
        lines.append("[규칙] 이 페르소나를 벗어난 설명조·교과서체 문장은 쓰지 않습니다.")
        return "\n".join(lines)


# ---------------------------------------------------------------- 장르 프리셋
#   beats: 장 하나를 구성하는 필수 비트(순서대로)
#   style: 문체 지시
#   chapter_target: 300p 목표에서 장당 대략 분량(자)
GENRES = {
    "fantasy": dict(
        label="판타지 소설",
        persona=Persona(
            name="서사 설계자", role="세계관을 먼저 세우고 인물을 그 안에 던지는 장편 소설가",
            era="마법이 제도로 관리되는 가상 왕국",
            values="인물의 욕망이 사건을 만든다. 편의를 위한 우연은 쓰지 않는다",
            voice="감각적 묘사 3문장 → 대사 → 행동. 문단마다 장면이 움직인다",
            taboo="설정 설명을 위한 대화, 갑작스러운 능력 각성, 결말의 기적",
            reader_promise="세계의 규칙을 이해하고 주인공의 선택에 납득한다"),
        beats=["장면 진입(감각)", "욕망/걸림", "갈등 심화", "전환점", "장 말미 훅"],
        style="1인칭 제한 시점. 현재형 감각 묘사. 마법 규칙은 행동으로만 보여준다.",
        chapter_target=4200,
    ),
    "poetry": dict(
        label="시집",
        persona=Persona(
            name="시인", role="일상의 장면에서 한 줄의 긴장을 남기는 시인",
            era="지금의 도시와 방",
            values="설명하지 않고 보여준다. 추상어보다 사물",
            voice="짧은 행갈이, 반복과 변주, 마지막 행에서 시선 이동",
            taboo="'그리움/영원/운명' 같은 상투 추상어 남용, 감정 직접 진술",
            reader_promise="한 편을 소리 내어 읽고 싶어진다"),
        beats=["장면", "응시", "전환", "여백"],
        style="연작 형태. 1부/2부/3부로 정조를 이동. 각 시 8~20행.",
        chapter_target=1400,
    ),
    "economy": dict(
        label="경제 교양서",
        persona=Persona(
            name="현장 경제 기자", role="데이터와 현장 인터뷰를 함께 다루는 경제 저널리스트",
            era="2020년대 이후 한국과 세계 경제",
            values="숫자는 맥락과 함께. 인과와 상관을 구분한다",
            voice="한 문단 한 논점. 통계 → 사례 → 반론 → 적용",
            taboo="출처 없는 수치, '무조건/반드시' 단정, 투자 권유",
            reader_promise="뉴스의 경제 기사를 스스로 해석한다"),
        beats=["질문", "데이터", "사례", "반론", "적용"],
        style="논픽션. 모든 수치는 '누가 언제 조사했는지' 형태로 표기. 투자 조언 금지.",
        chapter_target=4200,
    ),
    "essay": dict(
        label="에세이",
        persona=Persona(
            name="생활 관찰자", role="일상의 장면에서 생각의 실마리를 찾는 에세이스트",
            era="평범한 오늘",
            values="거창한 결론보다 정확한 장면",
            voice="장면 → 생각 → 다시 장면. 자기 비하와 유머 허용",
            taboo="교훈 강요, 남의 삶 평가",
            reader_promise="내 하루를 다르게 본다"),
        beats=["장면", "기억", "생각", "되돌아봄"],
        style="1인칭. 시제를 오가되 단락마다 하나의 장면만.",
        chapter_target=3000,
    ),
    "selfhelp": dict(
        label="실용/자기계발",
        persona=Persona(
            name="실행 코치", role="독자가 오늘 바로 시작할 수 있게 설계하는 코치",
            era="지금",
            values="작게 시작, 측정 가능, 실패 비용 낮추기",
            voice="원리 → 3단계 실행 → 체크리스트 → 흔한 실수",
            taboo="근거 없는 성공담, 과장된 보장",
            reader_promise="오늘 한 가지를 실제로 실행한다"),
        beats=["문제", "원리", "3단계", "체크리스트", "흔한 실수"],
        style="명령형보다 제안형. 모든 장에 실행 항목 1개 이상.",
        chapter_target=3600,
    ),
    "scifi": dict(
        label="SF",
        persona=Persona(
            name="기술 사변 소설가", role="기술 하나를 극단까지 밀어 사회를 보는 작가",
            era="근미래",
            values="기술은 중립이 아니다. 제도가 결과를 만든다",
            voice="차가운 관찰 + 결정적 장면. 설명은 인물의 업무 언어로",
            taboo="편의를 위한 설정 변경, 결말의 초능력",
            reader_promise="지금의 기술 선택을 다시 생각한다"),
        beats=["일상 속 기술", "균열", "제도적 갈등", "선택", "여파"],
        style="3인칭 제한. 기술 용어는 인물이 쓰는 말로만.",
        chapter_target=4200,
    ),
    "history": dict(
        label="역사/인물",
        persona=Persona(
            name="사료 해설자", role="1차 사료를 읽어 주는 역사 저술가",
            era="해당 시대",
            values="사료에 없는 것은 쓰지 않는다. 추정은 추정이라고 밝힌다",
            voice="사료 인용 → 해석 → 당대 맥락 → 현재적 질문",
            taboo="확인되지 않은 일화의 단정, 시대착오적 평가",
            reader_promise="사료와 해석의 차이를 구분한다"),
        beats=["사료", "해석", "맥락", "논쟁", "질문"],
        style="인용은 출처 표기. 연대·인명은 표기 통일.",
        chapter_target=4200,
    ),
    "business": dict(
        label="비즈니스/전략",
        persona=Persona(
            name="전략 컨설턴트", role="의사결정 구조를 세우는 전략가",
            era="현재 산업",
            values="프레임 → 증거 → 반대 시나리오 → 결정 기준",
            voice="표와 목록을 적극 사용. 결론을 먼저",
            taboo="일반론, 검증 불가한 베스트프랙티스",
            reader_promise="다음 회의에서 쓸 판단 기준을 갖는다"),
        beats=["결론", "프레임", "증거", "반대 시나리오", "결정 기준"],
        style="두괄식. 각 장 끝에 '결정 체크리스트'.",
        chapter_target=3900,
    ),
}
GENRE_ORDER = ["fantasy", "poetry", "economy", "essay", "selfhelp", "scifi", "history", "business"]


def genre_options() -> list:
    return [(GENRES[k]["label"], k) for k in GENRE_ORDER]


def persona_system(genre_key: str, extra_persona: str = "") -> str:
    """장르 페르소나 시스템 프롬프트 (+ 사용자가 덧붙이는 외부 페르소나)."""
    g = GENRES.get(genre_key, GENRES["economy"])
    sysmsg = g["persona"].system_prompt()
    sysmsg += f"\n[장르] {g['label']}\n[장르 문체 규칙] {g['style']}"
    if extra_persona.strip():
        sysmsg += ("\n[사용자 지정 페르소나 - 위 장르 규칙보다 우선]\n"
                   + extra_persona.strip())
    return sysmsg


# ---------------------------------------------------------------- 장편 구조
@dataclass
class StoryState:
    """Re3 / Narrative State Tracking 의 'typed state' 축소판."""
    characters: dict = field(default_factory=dict)   # 이름 -> {"want","fear","secret","voice"}
    places: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)     # ["3일차: ...", ...]
    open_threads: list = field(default_factory=list)  # 회수해야 할 복선
    resolved: list = field(default_factory=list)

    def to_prompt(self, max_items: int = 14) -> str:
        def fmt(d):
            return "; ".join(f"{k}: {v}" for k, v in list(d.items())[:max_items]) or "(없음)"
        return (
            f"[인물] {fmt(self.characters)}\n"
            f"[장소] {fmt(self.places)}\n"
            f"[중요 소지품] {fmt(self.items)}\n"
            f"[연표] {' | '.join(self.timeline[-8:]) or '(없음)'}\n"
            f"[미회수 복선] {', '.join(self.open_threads[-10:]) or '(없음)'}\n"
            f"[이미 회수] {', '.join(self.resolved[-10:]) or '(없음)'}"
        )


def plan_chapters(total_pages: int, genre_key: str, chapter_target: int | None = None) -> dict:
    """300장(페이지) 목표를 장/절로 분해한다.

    반환: {"chapters":[{no,title,pages,sections:[{no,title,beats,target_chars}]}], ...}
    페이지당 약 450자(국문 단행본 기준)로 환산한다.
    """
    g = GENRES.get(genre_key, GENRES["economy"])
    per_chapter = chapter_target or g["chapter_target"]
    total = max(1, int(total_pages))
    # 장 수: 장당 8~14p 로 나눈다(시집은 더 얇게)
    pages_per_chapter = 10 if genre_key != "poetry" else 6
    n_ch = max(1, min(60, round(total / pages_per_chapter)))
    sections_per_chapter = max(2, min(6, round(per_chapter / 1400)))
    sec_chars = int(per_chapter / sections_per_chapter)
    chapters = []
    remaining = total
    for i in range(n_ch):
        pages = max(1, round(remaining / (n_ch - i)))
        remaining -= pages
        secs = []
        for s in range(sections_per_chapter):
            beat = g["beats"][s % len(g["beats"])]
            secs.append({"no": s + 1, "title": f"{beat}", "beats": [beat],
                         "target_chars": sec_chars})
        chapters.append({"no": i + 1, "title": f"제{i + 1}장", "pages": pages,
                         "sections": secs})
    return {
        "genre": genre_key, "genre_label": g["label"], "total_pages": total,
        "chars_per_page": 450, "target_chars": total * 450,
        "chapters": chapters, "chapter_target_chars": per_chapter,
    }


def outline_prompt(topic: str, genre_key: str, total_pages: int, persona_extra: str = "") -> str:
    """상세 아웃라인(DOC 스타일) 생성 프롬프트."""
    g = GENRES.get(genre_key, GENRES["economy"])
    plan = plan_chapters(total_pages, genre_key)
    ch = plan["chapters"]
    return f"""다음 조건으로 {g['label']} 전체 아웃라인을 설계하세요.

[주제/전제] {topic}
[분량] {total_pages}쪽 (약 {plan['target_chars']:,}자), 장 {len(ch)}개
[장르] {g['label']} / {g['style']}
[장 필수 비트] {', '.join(g['beats'])}
{persona_extra and f'[페르소나] {persona_extra}'}

[규칙]
- 각 장은 '무엇이 달라지는가'가 한 줄로 보여야 한다(상태 변화).
- 같은 사건·같은 통계를 두 장에 배치하지 않는다.
- 인물/개념은 첫 등장 장을 명시한다.
- 마지막 장은 앞의 복선을 최소 3개 회수한다.

JSON만 출력:
{{"title":"책 제목","logline":"한 문장","state_seed":{{"characters":{{"이름":"욕망/두려움"}},
 "places":{{}},"items":{{}},"open_threads":["복선1"]}},
 "chapters":[{{"no":1,"title":"...","pages":10,"state_change":"...",
   "sections":[{{"title":"...","beats":["..."],"target_chars":1400}}]}}]}}"""


def revision_prompt(chapter_text: str, state_prompt: str, genre_key: str) -> str:
    """Re3 의 revision 패스: 일관성/설정 충돌/반복만 고친다."""
    g = GENRES.get(genre_key, GENRES["economy"])
    return f"""아래 장을 '수정'하세요. 새로 쓰지 말고 문제만 고칩니다.

[현재 상태]
{state_prompt}

[고칠 것 - 우선순위]
1) 설정 충돌(인물 이름·관계·시간·장소·능력 규칙)
2) 같은 정보의 반복 설명
3) 장르 문체 위반: {g['style']} / 금기: {g['persona'].taboo}
4) 장 말미에 다음 장으로 넘어가는 힘이 없는 경우 → 마지막 2문장만 다시 씀

[유지] 사건 순서, 인물의 욕망, 이미 확정된 사실.
[출력] 수정된 장 본문만.

[장 본문]
{chapter_text}"""


def state_update_prompt(chapter_text: str, state_prompt: str) -> str:
    """장 집필 후 상태(인물/복선/연표)를 갱신하는 프롬프트."""
    return f"""아래 장을 읽고 '이야기 상태'만 JSON 으로 갱신하세요.

[현재 상태]
{state_prompt}

[갱신 규칙]
- 새 인물/장소/소지품만 추가한다. 기존 항목은 삭제하지 않는다.
- 새로 심은 복선은 open_threads 에, 회수한 복선은 resolved 로 옮긴다.
- timeline 에는 '무슨 일이 있었는지' 한 줄만 추가한다(최대 3줄).

JSON만 출력:
{{"characters":{{}},"places":{{}},"items":{{}},"timeline":["..."],
 "open_threads":["..."],"resolved":["..."]}}

[장 본문]
{chapter_text}"""


def consistency_report(chapters: list, state: StoryState) -> dict:
    """로컬(모델 없이) 일관성 점검: 이름 표기 흔들림, 미회수 복선, 길이 편차."""
    import re
    issues = []
    names = list(state.characters.keys())
    for nm in names:
        variants = set()
        for c in chapters:
            for m in re.findall(r"[가-힣]{2,4}", c):
                if m != nm and len(m) == len(nm) and sum(a != b for a, b in zip(m, nm)) == 1:
                    variants.add(m)
        if variants:
            issues.append(f"'{nm}'와 유사한 표기 발견: {', '.join(sorted(variants)[:3])} "
                          f"(오타이거나 다른 인물이면 확인 필요)")
    if state.open_threads:
        unresolved = [t for t in state.open_threads if t not in state.resolved]
        if unresolved:
            issues.append(f"회수되지 않은 복선 {len(unresolved)}개: {', '.join(unresolved[:4])}")
    lens = [len(c) for c in chapters if c]
    if len(lens) >= 3:
        avg = sum(lens) / len(lens)
        thin = [i + 1 for i, L in enumerate(lens) if L < avg * 0.55]
        if thin:
            issues.append(f"분량이 평균의 55% 미만인 장: {thin[:5]} (확장 집필 권장)")
    return {"issues": issues, "checked_chapters": len(chapters),
            "open_threads": len(state.open_threads), "resolved": len(state.resolved)}


def persona_dict(genre_key: str) -> dict:
    g = GENRES.get(genre_key, GENRES["economy"])
    d = asdict(g["persona"])
    d.update({"genre": g["label"], "style": g["style"], "beats": g["beats"]})
    return d
