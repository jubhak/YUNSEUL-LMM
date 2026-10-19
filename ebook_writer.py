#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ebook_writer.py - 윤슬 LMM 전문 전자책 집필 및 자동 아카이빙 엔진
- books/YYYY-MM-DD/ 폴더 자동 생성
- 당일 중복 생성 시 ebook-01.html, ebook-02.html, ebook-03.html 자동 순번 채번
- 고품격 출판용 HTML (Noto Serif KR, 목차 앵커, A4 인쇄/PDF 서식) 변환 및 Markdown 동시 저장
"""

import os
import sys
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# 콘솔 UTF-8 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    base = Path(__file__).resolve().parent
    if (base / "books").exists() or (base / "widget_app.py").exists():
        return base
    dev_path = Path(r"D:\workspace\YUNSEUL-LMM")
    if dev_path.exists():
        return dev_path
    return base

# 기본 디렉터리 설정
BASE_DIR = get_base_dir()
BOOKS_DIR = BASE_DIR / "books"
BATCH_OUT = BASE_DIR / "batch_out"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)


def markdown_to_book_html(title: str, markdown_content: str, date_str: str = None) -> str:
    """마크다운 원고를 출판용 전문 HTML 서식으로 변환"""
    if not date_str:
        date_str = datetime.now().strftime("%Y년 %m월 %d일")

    body_lines = []
    toc_items = []
    ch_counter = 0

    lines = markdown_content.splitlines()
    for line in lines:
        line_str = line.strip()
        if not line_str:
            body_lines.append("<div class='spacer'></div>")
            continue

        if line_str.startswith("# ") and not line_str.startswith("## "):
            # 대제목(책 표지/제목)
            book_title = line_str[2:].strip()
            body_lines.append(f"""
            <div class='cover-wrapper'>
                <div class='cover-badge'>YUNSEUL DIGITAL PUBLISHING</div>
                <h1 class='book-title'>{book_title}</h1>
                <div class='book-meta'>
                    <span>발행일: {date_str}</span> &nbsp;|&nbsp;
                    <span>형식: 전자책 단행본 (HTML/PDF)</span>
                </div>
            </div>
            """)
        elif line_str.startswith("## "):
            # 챕터 제목
            ch_title = line_str[3:].strip()
            if "목차" in ch_title:
                body_lines.append("<div class='toc-title'>목 차 (Contents)</div>")
                continue
            ch_counter += 1
            ch_id = f"chapter-{ch_counter}"
            toc_items.append((ch_id, ch_title))
            body_lines.append(f"<h2 class='chapter-title' id='{ch_id}'><span class='ch-num'>CHAPTER {ch_counter:02d}</span>{ch_title}</h2>")
        elif line_str.startswith("### "):
            # 소제목
            sec_title = line_str[4:].strip()
            body_lines.append(f"<h3 class='section-title'>{sec_title}</h3>")
        elif line_str.startswith("---"):
            body_lines.append("<hr class='chapter-divider'>")
        elif line_str.startswith("- ") or line_str.startswith("* "):
            body_lines.append(f"<li class='book-list-item'>{line_str[2:]}</li>")
        elif re.match(r'^\d+\.\s', line_str):
            body_lines.append(f"<li class='book-list-item-num'>{line_str}</li>")
        else:
            # 본문 단락
            body_lines.append(f"<p class='body-text'>{line_str}</p>")

    formatted_body = "\n".join(body_lines)

    # 목차 HTML 생성
    toc_html = ""
    if toc_items:
        toc_entries = "".join([f"<li><a href='#{cid}'>{ctitle}</a></li>" for cid, ctitle in toc_items])
        toc_html = f"""
        <div class='toc-box'>
            <div class='toc-title'>목 차 (Contents)</div>
            <ul class='toc-list'>
                {toc_entries}
            </ul>
        </div>
        """

    html_template = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - 전자책 단행본</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300;400;600;700&family=Pretendard:wght@400;600;700&display=swap');

:root {{
    --bg-page: #faf8f5;
    --text-main: #27272a;
    --text-title: #09090b;
    --accent: #0284c7;
    --border: #e4e4e7;
    --line-height: 1.85;
}}

body {{
    font-family: 'Noto Serif KR', serif;
    font-size: 16.5px;
    line-height: var(--line-height);
    background-color: var(--bg-page);
    color: var(--text-main);
    margin: 0;
    padding: 60px 20px;
    -webkit-font-smoothing: antialiased;
}}

.book-container {{
    max-width: 780px;
    margin: 0 auto;
    background: #ffffff;
    padding: 70px 65px;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    border: 1px solid #edebe8;
}}

/* 표지 영역 */
.cover-wrapper {{
    text-align: center;
    padding: 60px 0 50px 0;
    border-bottom: 2px solid #18181b;
    margin-bottom: 50px;
}}
.cover-badge {{
    font-family: 'Pretendard', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent);
    margin-bottom: 25px;
}}
.book-title {{
    font-size: 2.3em;
    font-weight: 700;
    line-height: 1.35;
    color: var(--text-title);
    margin: 0 0 25px 0;
    word-break: keep-all;
}}
.book-meta {{
    font-family: 'Pretendard', sans-serif;
    font-size: 13px;
    color: #71717a;
}}

/* 목차 */
.toc-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 30px 35px;
    margin: 40px 0 60px 0;
}}
.toc-title {{
    font-family: 'Pretendard', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 1px;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid #cbd5e1;
}}
.toc-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}
.toc-list li {{
    margin: 10px 0;
    font-size: 14.5px;
    font-family: 'Pretendard', sans-serif;
}}
.toc-list a {{
    color: #334155;
    text-decoration: none;
    transition: color 0.2s;
}}
.toc-list a:hover {{
    color: var(--accent);
    text-decoration: underline;
}}

/* 챕터 & 본문 서식 */
.chapter-title {{
    font-family: 'Pretendard', sans-serif;
    font-size: 1.55em;
    font-weight: 700;
    color: #0c4a6e;
    margin-top: 60px;
    margin-bottom: 24px;
    padding-top: 25px;
    border-top: 2px solid #e0f2fe;
    page-break-before: always;
}}
.ch-num {{
    display: block;
    font-size: 12px;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 6px;
}}
.section-title {{
    font-family: 'Pretendard', sans-serif;
    font-size: 1.22em;
    font-weight: 600;
    color: #1e293b;
    margin-top: 35px;
    margin-bottom: 14px;
}}
.body-text {{
    margin-bottom: 1.45em;
    text-align: justify;
    text-indent: 1.2em;
    word-break: keep-all;
}}
.book-list-item, .book-list-item-num {{
    margin-left: 28px;
    margin-bottom: 0.6em;
    word-break: keep-all;
}}
.chapter-divider {{
    border: none;
    border-top: 1px dashed #cbd5e1;
    margin: 50px 0;
}}
.spacer {{ height: 16px; }}

/* 인쇄 및 PDF 저장용 스타일 */
@media print {{
    body {{
        background: white;
        padding: 0;
        color: #000;
    }}
    .book-container {{
        box-shadow: none;
        border: none;
        padding: 0;
        max-width: 100%;
    }}
    .chapter-title {{
        page-break-before: always;
    }}
    .toc-box {{
        page-break-after: always;
    }}
    a {{
        text-decoration: none !important;
        color: inherit !important;
    }}
}}
</style>
</head>
<body>
<div class='book-container'>
{formatted_body}
</div>
</body>
</html>
"""
    return html_template


def save_ebook_auto_increment(title: str, markdown_content: str, books_dir: Path = None, date_str: str = None) -> tuple:
    """
    books/YYYY-MM-DD/ 폴더를 자동 생성하고
    당일 1개 이상 집필 시 ebook-01.html, ebook-02.html, ebook-03.html 순번을 자동 계산하여 저장
    Returns: (html_path, md_path, index)
    """
    if books_dir is None:
        books_dir = BOOKS_DIR
    books_dir = Path(books_dir).resolve()
    books_dir.mkdir(parents=True, exist_ok=True)

    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # 년월일 서브 폴더 생성 (예: books/2026-10-03)
    target_date_dir = books_dir / date_str
    target_date_dir.mkdir(parents=True, exist_ok=True)

    # 순번 자동 채번 (ebook-01.html, ebook-02.html, ebook-03.html ...)
    idx = 1
    while True:
        html_candidate = target_date_dir / f"ebook-{idx:02d}.html"
        md_candidate = target_date_dir / f"ebook-{idx:02d}.md"
        if not html_candidate.exists() and not md_candidate.exists():
            break
        idx += 1

    final_html_path = target_date_dir / f"ebook-{idx:02d}.html"
    final_md_path = target_date_dir / f"ebook-{idx:02d}.md"

    # HTML 변환 및 저장
    html_content = markdown_to_book_html(title, markdown_content)
    final_html_path.write_text(html_content, encoding="utf-8")
    final_md_path.write_text(markdown_content, encoding="utf-8")

    # 기존 batch_out/ebook-01 호환성 동기화
    try:
        BATCH_OUT.mkdir(parents=True, exist_ok=True)
        (BATCH_OUT / "ebook-01.html").write_text(html_content, encoding="utf-8")
        (BATCH_OUT / "ebook-01.md").write_text(markdown_content, encoding="utf-8")
    except Exception:
        pass

    return final_html_path, final_md_path, idx


# =============================================================================
# 집필 엔진
# =============================================================================
from text_engine import (  # noqa: E402
    GenerationCancelled, RepetitionGuard, is_list_block, ollama_generate, paragraph_issues,
    split_sentences,
)

META_PHRASES = ("이 절에서는", "이 장에서는", "살펴보겠습니다", "알아보겠습니다", "요약하자면")

# mode -> (챕터 수, 챕터당 소제목 수, 소제목당 목표 글자 수)
MODE_SPEC = {
    "full": (7, 3, 1300),
    "chapter": (1, 4, 1300),
    "summary": (1, 3, 550),
}

WRITER_SYSTEM = """당신은 한국어 논픽션 전문 저술가입니다. 편집자가 교정 없이 바로 넘길 수 있는 원고를 씁니다.

[문체 원칙]
- 한 문단에는 하나의 생각만 담고, 문단은 3~5문장으로 쓴다.
- 같은 문장 끝맺음(~입니다, ~것입니다, ~보여줍니다, ~있습니다)을 세 문장 연속 쓰지 않는다. 짧은 문장, 질문, 인용, 장면 묘사를 섞어 리듬을 만든다.
- '이는', '또한', '특히', '따라서', '이처럼', '결론적으로' 같은 문장 시작어를 한 문단에서 두 번 쓰지 않는다.
- 앞에서 이미 쓴 문장, 통계, 사례, 비유를 다시 쓰지 않는다. 같은 개념을 다시 다뤄야 하면 새로운 각도(반론, 실행 방법, 다른 사례)로만 다룬다.
- 형용사를 쌓지 말고 구체적인 사람, 장소, 연도, 행동으로 보여준다.
- 확인할 수 없는 수치를 지어내지 않는다. '기관명 + 연도 + 퍼센트'를 묶은 통계는 널리 알려진 사실일 때만 쓰고, 아니면 '예를 들어 한 직장인이…'처럼 예시임이 드러나는 장면으로 설명한다.
- '이 절에서는 ~을 살펴보겠습니다', '요약하자면', '결론적으로 말하자면' 같은 메타 발언을 쓰지 않는다.
- 제목, 소제목, 머리말 없이 본문 문단만 출력한다."""

EDITOR_SYSTEM = """당신은 출판사 시니어 교정 편집자입니다. 원문의 의미와 사실 정보는 그대로 두고 문장만 다듬습니다.
결과 문단 하나만 출력하고, 설명이나 따옴표, 머리말을 붙이지 않습니다."""


def _parse_json_block(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _clean_section_body(text: str, titles: list) -> str:
    """LLM이 붙인 제목/구분선/메타 문장을 걷어낸다."""
    title_norms = {re.sub(r"\s+", "", t) for t in titles if t}
    lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("#") or s in ("---", "***"):
            continue
        if re.sub(r"[\s*#:]", "", s) in title_norms:
            continue
        lines.append(line.rstrip())
    body = "\n".join(lines).strip()
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


class EbookWriter:
    """
    기획(챕터/소제목/고유 사례 배정) -> 소제목 단위 집필 -> 중복 문장 제거 -> 반복 문단 윤문.
    이미 다룬 내용(ledger)과 과다 사용 표현을 매 소제목 프롬프트에 넣어 반복을 원천 차단한다.
    """

    def __init__(self, topic: str, model: str = "qwen3:14b", port: int = 11434, mode: str = "full",
                 log=print, on_progress=None, on_chapter=None, cancel_check=None):
        self.topic = topic
        self.model = model
        self.port = port
        self.mode = mode if mode in MODE_SPEC else "full"
        self.log = log or (lambda *_: None)
        self.on_progress = on_progress or (lambda *_: None)
        self.on_chapter = on_chapter or (lambda *_: None)
        self.cancel_check = cancel_check
        self.guard = RepetitionGuard()
        self.ledger = []
        self.partial_text = ""
        self.stats = {"dup_sentences_removed": 0, "paragraphs_rewritten": 0}

    def _gen(self, prompt, system=WRITER_SYSTEM, **kw):
        return ollama_generate(prompt, model=self.model, port=self.port, system=system,
                               cancel_check=self.cancel_check, **kw)

    # ------------------------------------------------------------------ 기획
    def plan(self) -> dict:
        n_ch, n_sec, _ = MODE_SPEC[self.mode]
        if self.mode == "full":
            shape = ("프롤로그, 제1장~제5장, 에필로그 순서의 7개 챕터. "
                     f"본문 챕터는 소제목 {n_sec}개, 프롤로그와 에필로그는 소제목 2개")
        else:
            shape = f"챕터 1개, 소제목 {n_sec}개"
        prompt = f"""주제 '{self.topic}'로 전자책을 기획하세요.

[구성] {shape}

[조건]
- 챕터끼리, 소제목끼리 다루는 내용이 겹치면 안 된다. 같은 사례나 통계를 두 곳에 배정하지 않는다.
- 소제목의 point에는 그 절에서만 다룰 구체적 논점을 1~2문장으로 쓴다.
- 챕터의 examples에는 그 장에서만 쓸 실존 사례, 인물, 사건, 연구를 2~3개 넣는다.
- 책 전체의 흐름은 문제 제기 -> 원리 -> 현장 변화 -> 갈등 -> 해법 -> 전망으로 점점 깊어지게 한다.

아래 JSON 형식으로만 응답하세요.
{{"book_title": "책 제목", "core_message": "책이 독자에게 남길 한 문장", "reader": "주 독자층",
  "chapters": [{{"title": "프롤로그: ...", "summary": "장 요약", "examples": ["..."],
                 "sections": [{{"title": "소제목", "point": "이 절의 고유 논점"}}]}}]}}"""

        plan = None
        for temp in (0.6, 0.3):
            raw = self._gen(prompt, system=None, max_tokens=3000, temperature=temp,
                            options={"presence_penalty": 0.0, "frequency_penalty": 0.0})
            plan = _parse_json_block(raw)
            if plan and len(plan.get("chapters") or []) >= n_ch:
                break
            self.log("⚠️ 목차 JSON 해석 실패 -> 재시도")
        chapters = (plan or {}).get("chapters") or []
        chapters = [c for c in chapters if isinstance(c, dict) and c.get("title")][:n_ch]
        if len(chapters) < n_ch:
            chapters = self._fallback_chapters(n_ch, n_sec)
        for ch in chapters:
            secs = [s for s in (ch.get("sections") or []) if isinstance(s, dict) and s.get("title")]
            if len(secs) < 2:
                secs = [{"title": "핵심 원리", "point": ch.get("summary", "")},
                        {"title": "현장의 사례", "point": "이 장의 고유 사례를 깊이 분석"},
                        {"title": "실천과 시사점", "point": "독자가 적용할 방법"}][:max(2, n_sec)]
            ch["sections"] = secs[:max(n_sec, 2)]
            ch.setdefault("examples", [])
        return {
            "book_title": ((plan or {}).get("book_title") or self.topic).strip(),
            "core_message": (plan or {}).get("core_message", ""),
            "reader": (plan or {}).get("reader", ""),
            "chapters": chapters,
        }

    def _fallback_chapters(self, n_ch, n_sec):
        if n_ch == 1:
            titles = [f"{self.topic}의 핵심"]
        else:
            titles = [
                f"프롤로그: 왜 지금 {self.topic}인가",
                f"제1장: {self.topic}의 본질과 작동 원리",
                "제2장: 산업과 일상에서 일어나는 변화",
                "제3장: 갈등과 숨은 위험",
                "제4장: 대응 전략과 실천 해법",
                "제5장: 10년 후의 전망과 기회",
                "에필로그: 독자에게 남기는 질문",
            ][:n_ch]
        base_secs = [("배경과 문제의식", "왜 이 주제가 중요한가"), ("구체적 사례", "현장에서 일어난 일"),
                     ("쟁점과 반론", "다른 시각과 한계"), ("실천 방법", "독자가 적용할 수 있는 방법")]
        return [{"title": t, "summary": "", "examples": [],
                 "sections": [{"title": a, "point": b} for a, b in base_secs[:n_sec]]} for t in titles]

    # ------------------------------------------------------------------ 집필
    def _section_prompt(self, plan, ch, sec, sec_idx, next_title, tail, target_chars):
        ledger = "\n".join(self.ledger[-45:]) or "(아직 없음)"
        avoid = self.guard.overused_phrases()
        avoid_txt = ", ".join(f"'{a}'" for a in avoid) if avoid else "(없음)"
        ch_examples = [e for e in (ch.get("examples") or []) if e]
        n_secs = len(ch["sections"])
        mine = [e for i, e in enumerate(ch_examples) if i % n_secs == sec_idx]
        others = [e for e in ch_examples if e not in mine]
        examples = ", ".join(mine) or "(이 절에서는 새 사례를 하나 고르되, 앞에서 쓴 사례와 수치는 제외)"
        if others:
            examples += f"\n[다른 소제목에 배정된 사례 - 이번 절에서는 언급하지 말 것] {', '.join(others)}"
        facts = self.guard.used_facts()
        facts_txt = ", ".join(facts) if facts else "(없음)"
        nxt = f"\n[다음 소제목 - 이 내용은 미리 쓰지 말 것] {next_title}" if next_title else ""
        tail_txt = (f"\n[직전 원고의 마지막 부분 - 흐름만 이어받고, 그 내용을 요약하거나 다시 인용하지 말 것]\n{tail}\n"
                    if tail else "")
        return f"""[책 주제] {self.topic}
[책 제목] {plan['book_title']}
[핵심 메시지] {plan.get('core_message', '')}
[주 독자] {plan.get('reader', '') or '일반 교양 독자'}

[현재 위치] {ch['title']} > {sec['title']} ({sec_idx + 1}/{len(ch['sections'])})
[이 장의 요약] {ch.get('summary', '')}
[이 절에서 다룰 고유 논점] {sec.get('point', '')}
[이 절에서 쓸 사례] {examples}

[이미 다룬 내용 - 다시 설명하거나 같은 사례, 수치를 재사용하지 말 것]
{ledger}

[이미 인용한 수치/출처/고유명 - 이번 절에서 다시 꺼내지 말 것] {facts_txt}

[이미 여러 번 쓴 표현 - 이번 절에서는 쓰지 말 것] {avoid_txt}
{tail_txt}{nxt}

위 조건으로 '{sec['title']}' 절의 본문을 약 {target_chars}자(공백 포함) 분량으로 쓰세요. 본문 문단만 출력하세요."""

    def _rewrite_paragraph(self, para, issues):
        avoid = self.guard.overused_phrases(top=8)
        prompt = (
            "다음 문단을 윤문하세요. 의미와 사실 정보는 유지하고 아래 문제만 고칩니다.\n\n"
            "[문제]\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
            + (f"[쓰지 말 표현] {', '.join(avoid)}\n\n" if avoid else "")
            + "[규칙] 길이는 원문과 비슷하게(±20%). 문단 하나만 출력.\n\n[문단]\n" + para
        )
        new = self._gen(prompt, system=EDITOR_SYSTEM, max_tokens=int(len(para) * 1.8) + 200,
                        temperature=0.6)
        new = _clean_section_body(new, [])
        if not new or not (0.6 <= len(new) / max(1, len(para)) <= 1.5):
            return None
        if len(paragraph_issues(new)) >= len(issues):
            return None
        return new

    def _polish(self, body: str, max_rewrites: int = 3) -> str:
        paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        avoid = set(self.guard.overused_phrases(top=12, min_count=4))
        done = 0
        for i, p in enumerate(paras):
            if done >= max_rewrites:
                break
            if is_list_block(p):
                continue
            issues = paragraph_issues(p)
            hits = [a for a in avoid if a in p]
            if len(hits) >= 2:
                issues.append("책 전체에서 이미 많이 쓴 표현 반복: " + ", ".join(hits))
            if not issues:
                continue
            new = self._rewrite_paragraph(p, issues)
            if new:
                paras[i] = new
                done += 1
        self.stats["paragraphs_rewritten"] += done
        cleaned = []
        for p in paras:
            if is_list_block(p):
                cleaned.append(p)
                continue
            sents = [s for s in split_sentences(p) if not any(m in s for m in META_PHRASES)]
            if sents:
                cleaned.append(" ".join(sents))
        return "\n\n".join(cleaned)

    def _digest(self, ch_title, sec_title, body):
        try:
            raw = self._gen(
                "아래 원고에서 다룬 핵심 주장, 사례, 수치를 '- '로 시작하는 한 줄씩 최대 5개로 정리하세요. "
                "목록만 출력하세요.\n\n" + body,
                system=None, max_tokens=400, temperature=0.2,
                options={"presence_penalty": 0.0, "frequency_penalty": 0.0})
        except GenerationCancelled:
            raise
        except Exception:
            return
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith(("-", "•", "*")):
                self.ledger.append(f"- [{sec_title}] {s.lstrip('-•* ').strip()}")

    def write_section(self, plan, ch, sec_idx, tail, target_chars) -> str:
        sec = ch["sections"][sec_idx]
        next_title = ch["sections"][sec_idx + 1]["title"] if sec_idx + 1 < len(ch["sections"]) else ""
        prompt = self._section_prompt(plan, ch, sec, sec_idx, next_title, tail, target_chars)
        body = self._gen(prompt, max_tokens=int(target_chars * 1.8) + 500)
        body = _clean_section_body(body, [ch["title"], sec["title"]])
        body, removed = self.guard.filter_new_text(body)

        if len(body) < target_chars * 0.75:
            need = target_chars - len(body)
            more = self._gen(
                prompt + f"\n\n[이미 쓴 본문]\n{body}\n\n위 본문에 이어서, 아직 다루지 않은 측면으로 약 {need}자를 더 쓰세요. "
                         "이미 쓴 문장을 반복하지 말고 이어지는 문단만 출력하세요.",
                max_tokens=int(need * 1.8) + 400)
            more, r2 = self.guard.filter_new_text(_clean_section_body(more, [ch["title"], sec["title"]]))
            removed += r2
            body = (body + "\n\n" + more).strip()

        self.stats["dup_sentences_removed"] += removed
        body = self._polish(body)
        self.guard.add(body)
        self._digest(ch["title"], sec["title"], body)
        return body

    def write(self) -> str:
        _, _, target_chars = MODE_SPEC[self.mode]
        self.on_progress("기획: 챕터와 소제목, 장별 사례 배정 중...", 0, 1, 0)
        plan = self.plan()
        chapters = plan["chapters"]
        total_steps = sum(len(c["sections"]) for c in chapters)
        self.log(f"📋 기획 완료: '{plan['book_title']}' / {len(chapters)}개 챕터, {total_steps}개 소제목")

        head = f"# {plan['book_title']}\n\n"
        if len(chapters) > 1:
            head += "## [목차 (Contents)]\n\n"
            for i, ch in enumerate(chapters):
                head += f"{i + 1}. {ch['title']}\n"
            head += "\n---\n"
        full = head
        self.partial_text = full

        step = 0
        tail = ""
        for ch in chapters:
            ch_md = f"\n\n## {ch['title']}\n"
            for si, sec in enumerate(ch["sections"]):
                self.on_progress(f"[{step + 1}/{total_steps}] {ch['title']} > {sec['title']} 집필 중...",
                                 step, total_steps, len(full + ch_md))
                body = self.write_section(plan, ch, si, tail, target_chars)
                ch_md += f"\n### {sec['title']}\n\n{body}\n"
                tail = body[-300:]
                step += 1
                self.partial_text = full + ch_md
            ch_md += "\n---\n"
            full += ch_md
            self.partial_text = full
            self.on_chapter(ch["title"], ch_md, full)
            self.log(f"   ✓ '{ch['title']}' 완료 ({len(ch_md):,}자)")

        self.on_progress("집필 완료", total_steps, total_steps, len(full))
        self.log(f"🧹 중복 문장 {self.stats['dup_sentences_removed']}개 제거, "
                 f"반복 문단 {self.stats['paragraphs_rewritten']}개 윤문")
        return full


def generate_and_save_ebook(
    topic: str = "인공지능의 현대 사회에 미치는 영향과 인간-AI 공존의 미래",
    model: str = "qwen3:14b",
    port: int = 11434,
    mode: str = "full",
    log_func=print,
) -> tuple:
    """전자책을 집필하고 books/YYYY-MM-DD/ebook-XX.html 로 자동 저장"""
    start_time = time.time()
    log_func(f"🚀 [전자책 집필 시작] 주제: '{topic}' | 모델: {model} | 모드: {mode}")
    writer = EbookWriter(topic, model=model, port=port, mode=mode, log=log_func,
                         on_progress=lambda msg, *_: log_func(f"✍️ {msg}"))
    full_text = writer.write()

    elapsed = int(time.time() - start_time)
    em, es = divmod(elapsed, 60)
    log_func(f"🎉 [집필 완료] 총 {len(full_text):,}자 / 소요 시간: {em}분 {es}초")

    html_path, md_path, file_idx = save_ebook_auto_increment(topic, full_text)
    log_func("💾 [자동 아카이빙 성공]")
    log_func(f"   • HTML 파일: {html_path}")
    log_func(f"   • MD 원고:   {md_path}")
    log_func(f"   • 당일 채번:  ebook-{file_idx:02d}.html")
    return full_text, html_path, md_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="윤슬 전자책 집필 및 books 자동 아카이빙 엔진")
    parser.add_argument("--topic", default="인공지능의 현대 사회에 미치는 영향과 인간-AI 공존의 미래", help="전자책 주제")
    parser.add_argument("--mode", default="full", choices=["full", "chapter", "summary"], help="집필 모드")
    parser.add_argument("--model", default="qwen3:14b", help="Ollama 모델명")
    parser.add_argument("--port", type=int, default=11434, help="Ollama 포트")
    args = parser.parse_args()

    generate_and_save_ebook(topic=args.topic, model=args.model, port=args.port, mode=args.mode)
