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


def call_ollama(prompt: str, model: str = "qwen3:14b", port: int = 11434, max_tokens: int = 8192) -> str:
    """Ollama 로컬 LLM API 호출"""
    url = f"http://127.0.0.1:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "num_predict": max_tokens,
            "temperature": 0.7
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=360) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def generate_and_save_ebook(
    topic: str = "인공지능의 현대 사회에 미치는 영향과 인간-AI 공존의 미래",
    model: str = "qwen3:14b",
    port: int = 11434,
    mode: str = "full",
    log_func = print
) -> tuple:
    """
    전자책 집필을 시작하고 완료 즉시 books/YYYY-MM-DD/ebook-XX.html 로 자동 저장
    """
    start_time = time.time()
    log_func(f"🚀 [전자책 집필 시작] 주제: '{topic}' | 모델: {model}")

    if mode == "summary":
        log_func("📖 [1/1] 핵심 요약본 작성 중...")
        prompt = f"다음 주제로 전문 전자책 원고(1,500자 이상)를 작성하라.\n\n주제: '{topic}'\n\n1. 목차와 체계적 구조 포함\n2. 사실 검증된 구체적 통계와 사례 서술"
        content = call_ollama(prompt, model=model, port=port, max_tokens=3000)
        full_text = f"# {topic}\n\n{content}\n"

    elif mode == "chapter":
        log_func("📖 [1/1] 심층 단행 챕터 작성 중...")
        prompt = f"다음 주제로 전문 전자책 원고(5,000자 이상)를 작성하라.\n\n주제: '{topic}'\n\n1. 체계적 소제목 구조 포함\n2. 통계와 역사적/현대적 사례를 풍부히 담아 심층 서술"
        content = call_ollama(prompt, model=model, port=port, max_tokens=6000)
        full_text = f"# {topic}\n\n{content}\n"

    else:
        # Full Book (7개 챕터 정규 단행본)
        log_func("📋 [기획 단계] 7개 챕터 정규 목차 설계 중...")
        toc_prompt = f"""당신은 출판계 최고의 기획자이자 저술가입니다.
주제: '{topic}'

이 주제로 총 7개 챕터(프롤로그, 제1장~제5장, 에필로그)로 구성된 상세 목차를 기획하세요.
반드시 아래 JSON 배열 형식으로만 응답해야 합니다:
[
  {{"title": "프롤로그: ...", "summary": "..."}},
  {{"title": "제1장: ...", "summary": "..."}},
  {{"title": "제2장: ...", "summary": "..."}},
  {{"title": "제3장: ...", "summary": "..."}},
  {{"title": "제4장: ...", "summary": "..."}},
  {{"title": "제5장: ...", "summary": "..."}},
  {{"title": "에필로그: ...", "summary": "..."}}
]"""
        toc_raw = call_ollama(toc_prompt, model=model, port=port, max_tokens=2048)
        chapters = []
        try:
            m = re.search(r'\[.*\]', toc_raw, re.DOTALL)
            if m:
                chapters = json.loads(m.group(0))
        except Exception:
            pass

        if not chapters or len(chapters) < 5:
            chapters = [
                {"title": f"프롤로그: 새로운 문명의 서막과 {topic}의 태동", "summary": "시대적 배경과 핵심 화두"},
                {"title": f"제1장: {topic}의 본질적 기제와 구조 분석", "summary": "원리와 핵심 개념"},
                {"title": f"제2장: 산업 생태계와 사회 구조의 대변혁", "summary": "산업별 파급 효과와 구체적 변화"},
                {"title": f"제3장: 현실적 갈등과 숨겨진 위기 요인", "summary": "부작용과 윤리적·경제적 쟁점"},
                {"title": f"제4장: 지속 가능한 공존을 위한 해법과 전략", "summary": "실천 가이드라인과 대응 전략"},
                {"title": f"제5장: 10년 후 미래 전망과 차세대 기회", "summary": "새로운 기회와 역량"},
                {"title": f"에필로그: 인간다움의 가치와 미래를 여는 지혜", "summary": "총평과 성찰"}
            ]

        total_steps = len(chapters)
        full_text = f"# {topic}\n\n"
        full_text += "## [목차 (Contents)]\n\n"
        for i, ch in enumerate(chapters):
            full_text += f"{i+1}. {ch.get('title')}\n"
        full_text += "\n---\n\n"

        # 실행 요약(running summary) — 챕터 간 일관성 유지용 메모리
        story_memory = ""

        for idx, ch in enumerate(chapters):
            ch_title = ch.get("title", f"제{idx+1}장")
            ch_summary = ch.get("summary", "")
            log_func(f"✍️ [{idx+1}/{total_steps}] '{ch_title}' 집필 중...")

            memory_block = ""
            if story_memory:
                memory_block = f"\n[지금까지 집필된 내용 요약 - 앞 내용과 모순되지 않게 이어서 쓸 것]\n{story_memory}\n"

            prompt = f"""당신은 '{topic}' 분야의 최고 권위자이자 전문 저술가입니다.
전체 전자책 기획에 맞춰, 아래 챕터의 원고를 최소 3,000자 이상으로 깊이 있고 풍부하게 작성하십시오.

[현재 집필할 챕터]
제목: {ch_title}
핵심 요강: {ch_summary}
{memory_block}
[전체 주제]
{topic}

[필수 작성 지침]
1. 실제 통계, 역사적/현대적 사례, 구체적인 예시와 비유를 풍부하게 포함할 것.
2. 단순 나열을 피하고 기승전결이 있는 서술형 문장으로 깊이 있게 전개할 것.
3. 세부 소제목(### 1. ..., ### 2. ..., ### 3. ...)을 명확히 나누어 독자가 몰입할 수 있도록 할 것.
4. 책의 한 장이므로 챕터 제목(## {ch_title})으로 시작할 것.
5. 앞 챕터에서 이미 설명한 내용을 반복하지 말고, 필요하면 '앞 장에서 언급했듯이'로 연결할 것."""

            ch_content = call_ollama(prompt, model=model, port=port, max_tokens=8192)
            full_text += f"\n\n## {ch_title}\n\n{ch_content}\n\n---\n"
            log_func(f"   ✓ '{ch_title}' 완료 ({len(ch_content):,}자)")

            # 다음 챕터를 위한 실행 요약 갱신
            try:
                sum_prompt = (
                    "아래 챕터 원고의 핵심 논지, 인물/사례, 결론을 3문장 이내로 요약하십시오. "
                    "요약만 출력하십시오.\n\n" + ch_content
                )
                ch_digest = call_ollama(sum_prompt, model=model, port=port, max_tokens=512)
                if ch_digest:
                    story_memory = (story_memory + f"\n- [{ch_title}] {ch_digest.strip()}").strip()
            except Exception:
                pass

    elapsed = int(time.time() - start_time)
    em, es = divmod(elapsed, 60)
    log_func(f"🎉 [집필 완료] 총 {len(full_text):,}자 / 소요 시간: {em}분 {es}초")

    # 자동 아카이빙 (books/YYYY-MM-DD/ebook-XX.html)
    html_path, md_path, file_idx = save_ebook_auto_increment(topic, full_text)
    log_func(f"💾 [자동 아카이빙 성공]")
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
