# -*- coding: utf-8 -*-
"""
song_studio.py - 노래 만들기 스튜디오 화면

왼쪽: 곡 설정(장르, 분위기, 보컬, 템포, 구성, 학습한 목소리)
가운데: 노래 아이디어 -> AI 가사 -> 가사 편집 -> 노래 만들기(진행률)
오른쪽: 내가 만든 노래 목록 + 플레이어(여러 버전 비교, 가사/설정 다시 불러오기)
"""
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QTextEdit, QVBoxLayout, QWidget,
)

from mp3_meta import embed_mp3_meta, read_cover_bytes

from lyrics_writer import (
    GENRE_LYRIC_GUIDE, GENRE_MENU, MOOD_CAPTION, build_caption, count_syllables,
    estimate_duration_sec, generate_lyrics, get_guide,
)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

TEMPO_FACTOR = {"느리게": 0.88, "보통": 1.0, "빠르게": 1.12}
FORMS = [("짧게 · 1절 + 후렴", 60), ("보통 · 2절 + 후렴 두 번", 120), ("풀버전 · 브릿지 + 마지막 후렴", 180)]
SECTION_BUTTONS = [("1절", "[Verse]"), ("프리코러스", "[Pre-Chorus]"), ("후렴", "[Chorus]"),
                   ("브릿지", "[Bridge]"), ("간주", "[Instrumental Break]"), ("아웃트로", "[Outro]")]
MAX_MOODS = 2
SHORT_NAMES = {
    "ballad_piano": "피아노 발라드", "ballad_acoustic": "어쿠스틱 발라드", "ballad_orchestral": "오케스트라 발라드",
    "ballad_rnb": "R&&B 발라드", "ost_drama": "드라마 OST", "kpop_rock": "록 발라드",
    "indie_folk": "인디 포크", "citypop": "시티팝", "kpop_rnb": "R&&B 그루브", "kpop_dance": "K-POP 댄스",
}

STUDIO_CSS = """
QWidget#studioRoot { background-color: #0b1120; }
QFrame[class="card"] { background-color: #111a2e; border: 1px solid #1e293b; border-radius: 12px; }
QLabel { color: #cbd5e1; font-size: 12px; }
QLabel[class="h1"] { color: #f8fafc; font-size: 15px; font-weight: bold; }
QLabel[class="h2"] { color: #94a3b8; font-size: 11px; font-weight: bold; margin-top: 6px; }
QLabel[class="hint"] { color: #64748b; font-size: 11px; }
QPushButton[class="chip"] {
    background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 14px;
    padding: 6px 10px; font-size: 12px;
}
QPushButton[class="chip"]:hover { border-color: #64748b; }
QPushButton[class="chip"]:checked { background-color: #be185d; color: white; border-color: #ec4899; font-weight: bold; }
QPushButton[class="tool"] {
    background-color: #1e293b; color: #93c5fd; border: 1px solid #334155; border-radius: 6px;
    padding: 4px 9px; font-size: 11px;
}
QPushButton[class="tool"]:hover { background-color: #334155; }
QPushButton[class="ghost"] {
    background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px;
}
QPushButton[class="ghost"]:hover { background-color: #334155; }
QPushButton[class="ghost"]:disabled { color: #475569; }
QPushButton#btnLyrics { background-color: #7c3aed; color: white; font-weight: bold; border-radius: 8px; padding: 8px 18px; }
QPushButton#btnLyrics:disabled { background-color: #334155; color: #94a3b8; }
QPushButton#btnSong { background-color: #ec4899; color: white; font-weight: bold; font-size: 14px; border-radius: 8px; padding: 10px 22px; }
QPushButton#btnSong:disabled { background-color: #334155; color: #94a3b8; }
QPushButton#btnPlay { background-color: #0284c7; color: white; font-weight: bold; border-radius: 18px; }
QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #0f172a; color: #f1f5f9; border: 1px solid #334155; border-radius: 8px;
    padding: 8px; font-size: 13px; selection-background-color: #be185d;
}
QComboBox {
    background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 5px 8px; font-size: 12px;
}
QComboBox QAbstractItemView { background-color: #0f172a; color: white; selection-background-color: #be185d; }
QListWidget { background-color: #0f172a; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 8px; font-size: 12px; }
QListWidget::item { padding: 8px 6px; border-bottom: 1px solid #1e293b; }
QListWidget::item:selected { background-color: #3b0764; color: white; }
QProgressBar { background-color: #1e293b; border: none; border-radius: 4px; height: 8px; color: transparent; }
QProgressBar::chunk { background-color: #ec4899; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #1e293b; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #ec4899; border-radius: 2px; }
QSlider::handle:horizontal { background: #f9a8d4; width: 12px; margin: -4px 0; border-radius: 6px; }
QScrollArea { border: none; background: transparent; }
"""


def _label(text, cls=None, wrap=False):
    lb = QLabel(text)
    if cls:
        lb.setProperty("class", cls)
    lb.setWordWrap(wrap)
    return lb


def _card():
    f = QFrame()
    f.setProperty("class", "card")
    return f


def _fmt_time(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def suggest_title(lyrics: str) -> str:
    lines = lyrics.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\[(chorus|후렴)", ln.strip(), re.I):
            for nxt in lines[i + 1:]:
                s = re.sub(r"\(.*?\)", "", nxt).strip()
                if s and not s.startswith("["):
                    return s[:24]
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith("["):
            return s[:24]
    return ""


# =============================================================================
# 작업 스레드
# =============================================================================
class LyricsWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)

    def __init__(self, port, model, feeling, genre_key, duration, extra):
        super().__init__()
        self.args = (port, model, feeling, genre_key, duration, extra)

    def run(self):
        port, model, feeling, genre_key, duration, extra = self.args
        try:
            lyrics = generate_lyrics(feeling, genre_key, duration=duration, model=model, port=port,
                                     extra=extra, log=self.progress.emit)
            self.finished.emit(True, lyrics, "")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, "", str(e))


class SongWorker(QThread):
    """acestep_official_runner.py 를 ACE-Step 전용 파이썬으로 실행하고 진행 단계를 알린다."""
    stage = pyqtSignal(int, int, str)        # 현재 %, 이 단계에서 다가갈 최대 %, 안내 문구
    finished = pyqtSignal(bool, str, str)

    STAGES = [
        ("DiT 핸들러 초기화", 8, 20, "작곡 엔진 불러오는 중"),
        ("학습된 어댑터 로드", 20, 23, "학습한 목소리 입히는 중"),
        ("5Hz LM 초기화", 23, 31, "멜로디 설계 모델 불러오는 중"),
        ("곡 생성 시작", 32, 88, "멜로디 · 반주 · 보컬 만드는 중"),
        ("생성 완료", 88, 96, "버전별로 가사가 또렷한지 비교하는 중"),
        ("최종 출력", 97, 99, "음량 다듬고 저장하는 중"),
    ]

    def __init__(self, cmd, cwd, text_model):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.text_model = text_model
        self.proc = None
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)

    def run(self):
        out_path = self.cmd[self.cmd.index("--output") + 1]
        try:
            self.stage.emit(2, 8, "그래픽카드 메모리 정리 중")
            subprocess.run(["ollama", "stop", self.text_model], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", env=env, cwd=self.cwd,
                                         creationflags=CREATE_NO_WINDOW)
            tail = []
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail = (tail + [line])[-30:]
                for key, lo, hi, msg in self.STAGES:
                    if key in line:
                        self.stage.emit(lo, hi, msg)
                        break
            self.proc.wait()
            if self.cancelled:
                self.finished.emit(False, "", "사용자가 중지했습니다.")
                return
            rc = self.proc.returncode
            if rc == 4:
                raise RuntimeError("학습한 목소리(LoRA)를 불러오지 못했습니다. 목소리를 '기본 목소리'로 바꾸거나 다시 학습해 주세요.")
            if rc == 3:
                raise RuntimeError("만든 버전이 모두 잡음/무음으로 판정됐습니다. 다시 시도하거나 가사를 줄여 보세요.")
            if rc != 0 or not Path(out_path).exists():
                raise RuntimeError("작곡 엔진 오류:\n" + "\n".join(tail[-10:]))
            self.finished.emit(True, out_path, "")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, "", str(e))


# =============================================================================
# 화면
# =============================================================================
class SongStudio(QWidget):
    def __init__(self, host, base_dir: Path, songs_dir: Path, trained_dir: Path, acestep_py: Path):
        super().__init__()
        self.host = host
        self.base_dir = Path(base_dir)
        self.songs_dir = Path(songs_dir)
        self.songs_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir = Path(trained_dir)
        self.acestep_py = Path(acestep_py)
        self.player = host.player
        self.audio_output = host.audio_output
        self.lyrics_worker = None
        self.song_worker = None
        self._lyrics_busy = False
        self._song_busy = False
        self.current_song = None
        self._pct = 0.0
        self._pct_cap = 0.0
        self._gen_started = 0.0

        self.setObjectName("studioRoot")
        self.setStyleSheet(STUDIO_CSS)
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)
        root.addWidget(self._build_settings_panel())
        root.addWidget(self._build_editor_panel(), 1)
        root.addWidget(self._build_library_panel())

        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._tick_progress)
        self.stats_timer = QTimer(self)
        self.stats_timer.setSingleShot(True)
        self.stats_timer.timeout.connect(self._update_lyrics_stats)

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda d: self.slider_seek.setRange(0, d))
        self.player.playbackStateChanged.connect(self._on_play_state)

        self.refresh_voices()
        self.refresh_library()
        self._update_lyrics_stats()
        self._update_buttons()

    # ------------------------------------------------------------------ 왼쪽: 곡 설정
    def _build_settings_panel(self):
        card = _card()
        card.setFixedWidth(312)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(6)
        lay.addWidget(_label("곡 설정", "h1"))

        self.genre_group = QButtonGroup(self)
        self.genre_group.setExclusive(True)
        for group, keys in GENRE_MENU:
            lay.addWidget(_label(group, "h2"))
            grid = QGridLayout()
            grid.setSpacing(6)
            for i, key in enumerate(keys):
                b = QPushButton(SHORT_NAMES.get(key, GENRE_LYRIC_GUIDE[key]["name"]))
                b.setProperty("class", "chip")
                b.setCheckable(True)
                b.setProperty("genre", key)
                b.setToolTip(GENRE_LYRIC_GUIDE[key]["tone"])
                self.genre_group.addButton(b)
                grid.addWidget(b, i // 2, i % 2)
            lay.addLayout(grid)
        self.genre_group.buttons()[0].setChecked(True)

        lay.addWidget(_label(f"분위기 (최대 {MAX_MOODS}개)", "h2"))
        mood_grid = QGridLayout()
        mood_grid.setSpacing(6)
        self.mood_buttons = []
        self._mood_order = []
        for i, mood in enumerate(MOOD_CAPTION):
            b = QPushButton(mood)
            b.setProperty("class", "chip")
            b.setCheckable(True)
            b.toggled.connect(lambda on, m=mood: self._on_mood_toggled(m, on))
            self.mood_buttons.append(b)
            mood_grid.addWidget(b, i // 4, i % 4)
        lay.addLayout(mood_grid)

        lay.addWidget(_label("템포", "h2"))
        self.tempo_group, row = self._segmented([(k, k) for k in TEMPO_FACTOR], default=1)
        lay.addLayout(row)

        lay.addWidget(_label("곡 구성", "h2"))
        self.combo_form = QComboBox()
        for label, sec in FORMS:
            self.combo_form.addItem(label, sec)
        self.combo_form.setCurrentIndex(1)
        lay.addWidget(self.combo_form)

        lay.addWidget(_label("목소리 (여성 · 남성 · 별칭)", "h2"))
        self.combo_voice = QComboBox()
        self.combo_voice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_voice.setToolTip("여성, 남성, 또는 따라 읽어 저장한 별칭 목소리입니다.")
        lay.addWidget(self.combo_voice)

        lay.addWidget(_label("학습 모델", "h2"))
        self.combo_model = QComboBox()
        self.combo_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_model.setToolTip("가수 노래로 학습한 모델입니다. 목소리 별칭과는 따로 고릅니다.")
        lay.addWidget(self.combo_model)

        vrow = QHBoxLayout()
        self.combo_strength = QComboBox()
        for label, val in [("반영 100%", 1.0), ("반영 80%", 0.8), ("반영 60%", 0.6), ("반영 120%", 1.2)]:
            self.combo_strength.addItem(label, val)
        self.combo_strength.setToolTip("발음이 뭉개지면 낮추고, 학습한 느낌이 약하면 높이세요.")
        vrow.addWidget(self.combo_strength, 1)
        btn_refresh = QPushButton("🔄")
        btn_refresh.setProperty("class", "ghost")
        btn_refresh.setToolTip("목소리와 학습 모델 목록 새로고침")
        btn_refresh.clicked.connect(self.refresh_voices)
        vrow.addWidget(btn_refresh)
        lay.addLayout(vrow)
        self.lbl_voice_info = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_voice_info)
        self.combo_voice.currentIndexChanged.connect(self._on_voice_changed)
        self.combo_model.currentIndexChanged.connect(self._on_voice_changed)

        lay.addWidget(_label("한 번에 만들 버전 수", "h2"))
        self.combo_takes = QComboBox()
        for n in (1, 2, 3):
            self.combo_takes.addItem(f"{n}개" + (" (추천)" if n == 2 else ""), n)
        self.combo_takes.setCurrentIndex(1)
        self.combo_takes.setToolTip("여러 버전을 만들어 가사가 가장 또렷하게 들리는 것을 자동으로 고릅니다.\n버전이 많을수록 오래 걸립니다.")
        lay.addWidget(self.combo_takes)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return card

    def _segmented(self, items, default=0):
        group = QButtonGroup(self)
        group.setExclusive(True)
        row = QHBoxLayout()
        row.setSpacing(6)
        for i, (label, val) in enumerate(items):
            b = QPushButton(label)
            b.setProperty("class", "chip")
            b.setCheckable(True)
            b.setProperty("value", val)
            group.addButton(b)
            row.addWidget(b)
            if i == default:
                b.setChecked(True)
        return group, row

    def _on_mood_toggled(self, mood, on):
        if on:
            self._mood_order.append(mood)
            while len(self._mood_order) > MAX_MOODS:
                old = self._mood_order.pop(0)
                for b in self.mood_buttons:
                    if b.text() == old:
                        b.blockSignals(True)
                        b.setChecked(False)
                        b.blockSignals(False)
        elif mood in self._mood_order:
            self._mood_order.remove(mood)

    # ------------------------------------------------------------------ 가운데: 아이디어 -> 가사 -> 노래
    def _build_editor_panel(self):
        card = _card()
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        lay.addWidget(_label("① 노래 아이디어", "h1"))
        self.edit_idea = QTextEdit()
        self.edit_idea.setAcceptRichText(False)
        self.edit_idea.setFixedHeight(78)
        self.edit_idea.setPlaceholderText(
            "어떤 이야기를, 어떤 마음으로 부르는 노래인가요? 자유롭게 적어주세요.\n"
            "예) 이별 후 1년, 그 사람이 좋아하던 카페 앞을 지나며 이제는 웃으며 보내주는 마음")
        lay.addWidget(self.edit_idea)

        row = QHBoxLayout()
        self.btn_lyrics = QPushButton("✨ 가사 만들기")
        self.btn_lyrics.setObjectName("btnLyrics")
        self.btn_lyrics.clicked.connect(self.start_lyrics)
        row.addWidget(self.btn_lyrics)
        self.lbl_lyrics_status = _label("", "hint")
        row.addWidget(self.lbl_lyrics_status, 1)
        lay.addLayout(row)

        title_row = QHBoxLayout()
        lay.addWidget(_label("② 가사 다듬기", "h1"))
        title_row.addWidget(QLabel("제목"))
        self.edit_title = QLineEdit()
        self.edit_title.setPlaceholderText("비워 두면 후렴 첫 줄로 정해집니다")
        title_row.addWidget(self.edit_title, 1)
        lay.addLayout(title_row)

        tools = QHBoxLayout()
        tools.setSpacing(4)
        tools.addWidget(_label("구간 넣기:", "hint"))
        for label, tag in SECTION_BUTTONS:
            b = QPushButton(label)
            b.setProperty("class", "tool")
            b.setToolTip(f"커서 위치에 {tag} 표시를 넣습니다")
            b.clicked.connect(lambda _=False, t=tag: self._insert_section(t))
            tools.addWidget(b)
        tools.addStretch()
        lay.addLayout(tools)

        self.edit_lyrics = QPlainTextEdit()
        self.edit_lyrics.setPlaceholderText(
            "[가사 만들기]를 누르면 여기에 가사가 나타나고, 마음대로 고칠 수 있습니다. 직접 써도 됩니다.\n\n"
            "• [Verse] 1절, [Chorus] 후렴, [Bridge] 브릿지처럼 대괄호는 노래 구조를 알려줍니다.\n"
            "• 한 줄은 한 숨에 부를 길이(6~10글자)가 좋습니다.\n"
            "• 같은 후렴을 여러 번 부르게 하려면 [Chorus] 구간을 그대로 복사해 두세요.")
        self.edit_lyrics.textChanged.connect(lambda: (self.stats_timer.start(300), self._update_buttons()))
        lay.addWidget(self.edit_lyrics, 1)
        self.lbl_stats = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_stats)

        lay.addWidget(_label("③ 노래 만들기", "h1"))
        self.edit_extra = QLineEdit()
        self.edit_extra.setPlaceholderText("(선택) 더 넣고 싶은 악기·느낌을 영어로 — 예: rainy night, cello solo, soft choir")
        lay.addWidget(self.edit_extra)

        gen_row = QHBoxLayout()
        self.btn_song = QPushButton("🎵 이 가사로 노래 만들기")
        self.btn_song.setObjectName("btnSong")
        self.btn_song.clicked.connect(self.start_song)
        gen_row.addWidget(self.btn_song)
        self.btn_cancel = QPushButton("중지")
        self.btn_cancel.setProperty("class", "ghost")
        self.btn_cancel.clicked.connect(self.cancel_song)
        self.btn_cancel.hide()
        gen_row.addWidget(self.btn_cancel)
        prog_col = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        prog_col.addWidget(self.progress)
        self.lbl_progress = _label("가사가 준비되면 노래를 만들 수 있습니다.", "hint")
        prog_col.addWidget(self.lbl_progress)
        gen_row.addLayout(prog_col, 1)
        lay.addLayout(gen_row)
        return card

    def _insert_section(self, tag):
        cur = self.edit_lyrics.textCursor()
        block_text = cur.block().text()
        prefix = "\n" if block_text.strip() else ""
        cur.insertText(f"{prefix}\n{tag}\n" if self.edit_lyrics.toPlainText().strip() else f"{tag}\n")
        self.edit_lyrics.setFocus()

    def _update_lyrics_stats(self):
        text = self.edit_lyrics.toPlainText()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        sections = [l for l in lines if l.startswith("[")]
        sung = [l for l in lines if not l.startswith("[")]
        if not sung:
            self.lbl_stats.setText("")
            return
        long_lines = [l for l in sung if count_syllables(re.sub(r"\(.*?\)", "", l)) > 13]
        dur = estimate_duration_sec(text, self._bpm())
        msg = f"구간 {len(sections)}개 · 가사 {len(sung)}줄 · 예상 길이 약 {_fmt_time(dur)}"
        if not sections:
            msg += "   ⚠️ [Verse], [Chorus] 같은 구간 표시가 없으면 노래 구조가 흐트러질 수 있어요"
        elif long_lines:
            msg += f"   ⚠️ 너무 긴 줄 {len(long_lines)}개 (13글자 넘음) — 나눠 쓰면 더 또렷하게 부릅니다"
        self.lbl_stats.setText(msg)

    # ------------------------------------------------------------------ 오른쪽: 내 노래
    def _build_library_panel(self):
        card = _card()
        card.setFixedWidth(320)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        lay.addWidget(_label("내 노래", "h1"))

        self.list_songs = QListWidget()
        self.list_songs.currentItemChanged.connect(self._on_song_selected)
        self.list_songs.itemDoubleClicked.connect(lambda _: self.toggle_play())
        lay.addWidget(self.list_songs, 1)

        row = QHBoxLayout()
        self.btn_reuse = QPushButton("가사·설정 불러오기")
        self.btn_reuse.setProperty("class", "ghost")
        self.btn_reuse.setToolTip("이 노래의 가사와 설정을 가운데 편집기로 가져옵니다 (고쳐서 다시 만들 때)")
        self.btn_reuse.clicked.connect(self.reuse_selected)
        row.addWidget(self.btn_reuse, 1)
        btn_folder = QPushButton("📁")
        btn_folder.setProperty("class", "ghost")
        btn_folder.setToolTip("노래 폴더 열기")
        btn_folder.clicked.connect(lambda: os.startfile(str(self.songs_dir)))
        row.addWidget(btn_folder)
        self.btn_delete = QPushButton("🗑")
        self.btn_delete.setProperty("class", "ghost")
        self.btn_delete.setToolTip("선택한 노래 삭제")
        self.btn_delete.clicked.connect(self.delete_selected)
        row.addWidget(self.btn_delete)
        self.btn_cover = QPushButton("🖼 커버")
        self.btn_cover.setProperty("class", "ghost")
        self.btn_cover.setToolTip("선택한 노래 MP3에 커버 이미지를 넣습니다")
        self.btn_cover.clicked.connect(self.set_cover)
        row.addWidget(self.btn_cover)
        lay.addLayout(row)

        player = _card()
        p = QVBoxLayout(player)
        p.setContentsMargins(12, 10, 12, 10)
        p.setSpacing(6)
        now_row = QHBoxLayout()
        self.lbl_cover = QLabel()
        self.lbl_cover.setFixedSize(56, 56)
        self.lbl_cover.setStyleSheet("background-color: #0f172a; border-radius: 6px;")
        self.lbl_cover.setScaledContents(True)
        now_row.addWidget(self.lbl_cover)
        self.lbl_now = _label("노래를 선택하세요", wrap=True)
        self.lbl_now.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 13px;")
        now_row.addWidget(self.lbl_now, 1)
        p.addLayout(now_row)
        self.combo_take = QComboBox()
        self.combo_take.setToolTip("같은 가사로 함께 만들어진 다른 버전")
        self.combo_take.currentIndexChanged.connect(self._on_take_changed)
        self.combo_take.hide()
        p.addWidget(self.combo_take)
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.sliderMoved.connect(self.player.setPosition)
        p.addWidget(self.slider_seek)
        ctl = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("btnPlay")
        self.btn_play.setFixedSize(36, 36)
        self.btn_play.clicked.connect(self.toggle_play)
        ctl.addWidget(self.btn_play)
        self.lbl_time = _label("0:00 / 0:00", "hint")
        ctl.addWidget(self.lbl_time)
        ctl.addStretch()
        ctl.addWidget(_label("🔊", "hint"))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setFixedWidth(80)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(int(self.audio_output.volume() * 100))
        self.slider_vol.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        ctl.addWidget(self.slider_vol)
        p.addLayout(ctl)
        lay.addWidget(player)
        return card

    # ------------------------------------------------------------------ 설정 값
    def _genre(self):
        b = self.genre_group.checkedButton()
        return b.property("genre") if b else "ballad_piano"

    def _gender(self):
        d = self.combo_voice.currentData()
        if isinstance(d, dict) and d.get("gender") in ("female", "male"):
            return d["gender"]
        return "female"

    def _tempo(self):
        b = self.tempo_group.checkedButton()
        return b.property("value") if b else "보통"

    def _moods(self):
        return list(self._mood_order)

    def _bpm(self):
        return int(round(get_guide(self._genre())["bpm"] * TEMPO_FACTOR.get(self._tempo(), 1.0)))

    def _set_checked(self, group, prop, value):
        for b in group.buttons():
            if b.property(prop) == value:
                b.setChecked(True)
                return

    def _load_trained_metas(self):
        aliases, models, legacy = [], [], []
        if not self.trained_dir.exists():
            return aliases, models, legacy
        for mdir in sorted(self.trained_dir.iterdir()):
            meta_path = mdir / "meta.json"
            if not mdir.is_dir() or not meta_path.exists() or mdir.name.startswith("_"):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            adapter = meta.get("adapter_path")
            if not adapter or not (Path(adapter) / "adapter_config.json").exists():
                legacy.append(mdir.name)
                continue
            meta["name"] = meta.get("name") or mdir.name
            kind = meta.get("kind") or "song_lora"
            meta["kind"] = kind
            (aliases if kind == "speech_clone" else models).append(meta)
        return aliases, models, legacy

    def _restore_combo(self, combo, prev_name, default=0):
        idx = default
        if prev_name:
            for i in range(combo.count()):
                d = combo.itemData(i)
                if isinstance(d, dict) and d.get("name") == prev_name:
                    idx = i
                    break
        combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ 목소리 / 학습 모델
    def refresh_voices(self):
        prev_voice = self.combo_voice.currentData()
        prev_model = self.combo_model.currentData() if hasattr(self, "combo_model") else None
        prev_voice_name = prev_voice.get("name") if isinstance(prev_voice, dict) else None
        prev_model_name = prev_model.get("name") if isinstance(prev_model, dict) else None
        aliases, models, legacy = self._load_trained_metas()
        self.combo_voice.blockSignals(True)
        self.combo_model.blockSignals(True)
        self.combo_voice.clear()
        self.combo_voice.addItem("여성", {"kind": "builtin", "gender": "female"})
        self.combo_voice.addItem("남성", {"kind": "builtin", "gender": "male"})
        for meta in aliases:
            self.combo_voice.addItem(f"🎤 {meta['name']}", meta)
        self.combo_model.clear()
        self.combo_model.addItem("사용 안 함", None)
        for meta in models:
            self.combo_model.addItem(meta["name"], meta)
        self._restore_combo(self.combo_voice, prev_voice_name, 0)
        self._restore_combo(self.combo_model, prev_model_name, 0)
        self.combo_voice.blockSignals(False)
        self.combo_model.blockSignals(False)
        self._legacy_voices = legacy
        self._on_voice_changed()

    def _active_adapter(self):
        """별칭 목소리가 있으면 그 LoRA를, 없으면 학습 모델을 쓴다. 둘은 동시에 겹치지 않는다."""
        voice = self.combo_voice.currentData()
        if isinstance(voice, dict) and voice.get("kind") == "speech_clone":
            return voice
        model = self.combo_model.currentData()
        if isinstance(model, dict) and model.get("adapter_path"):
            return model
        return None

    def _on_voice_changed(self):
        voice = self.combo_voice.currentData()
        model = self.combo_model.currentData()
        g = "여성" if self._gender() == "female" else "남성"
        parts = []
        if isinstance(voice, dict) and voice.get("kind") == "speech_clone":
            n = voice.get("clips_count") or voice.get("songs_count") or "?"
            parts.append(f"별칭 '{voice.get('name')}' ({n}문장)")
            if isinstance(model, dict):
                parts.append("학습 모델은 별칭과 겹쳐 이번 곡에는 별칭만 적용")
        else:
            parts.append(f"기본 {g} 보컬")
            if isinstance(model, dict):
                parts.append(f"학습 모델 {model.get('name')} · {model.get('songs_count', '?')}곡")
        if getattr(self, "_legacy_voices", None):
            parts.append("다시 학습 필요: " + ", ".join(self._legacy_voices))
        self.lbl_voice_info.setText(" · ".join(parts))

    # ------------------------------------------------------------------ 가사 만들기
    def _ollama_alive(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.host.port}/api/version", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def start_lyrics(self):
        idea = self.edit_idea.toPlainText().strip()
        if not idea:
            QMessageBox.information(self, "노래 아이디어", "①에 어떤 노래인지 한두 문장으로 적어주세요.")
            self.edit_idea.setFocus()
            return
        if self.edit_lyrics.toPlainText().strip():
            if QMessageBox.question(self, "가사 새로 만들기", "지금 가사를 새 가사로 바꿀까요?") != QMessageBox.StandardButton.Yes:
                return
        if not self._ollama_alive():
            QMessageBox.warning(self, "글쓰기 AI 꺼짐", "가사를 쓰는 AI(Ollama)가 꺼져 있습니다.\n[⚡ 서버 & VRAM] 탭에서 서버를 켠 뒤 다시 눌러주세요.")
            return
        ok, msg = self.host.check_vram_for_task("ebook")
        if not ok:
            QMessageBox.warning(self, "메모리 부족", msg)
            return
        gender_ko = "여성" if self._gender() == "female" else "남성"
        extra = f"{gender_ko} 보컬이 부른다."
        if self._moods():
            extra += f" 분위기: {', '.join(self._moods())}."
        extra += f" 템포: {self._tempo()}."
        self.lyrics_worker = LyricsWorker(self.host.port, self.host.cfg.get("text_model", "qwen3:14b"), idea,
                                          self._genre(), self.combo_form.currentData(), extra)
        self.lyrics_worker.progress.connect(lambda m: self.lbl_lyrics_status.setText(m.strip()))
        self.lyrics_worker.finished.connect(self._on_lyrics_done)
        self.lbl_lyrics_status.setText("가사를 쓰는 중... (30초~1분)")
        self._lyrics_busy = True
        self.lyrics_worker.start()
        self._update_buttons()

    def _on_lyrics_done(self, ok, lyrics, err):
        self._lyrics_busy = False
        self._update_buttons()
        if not ok or not lyrics.strip():
            self.lbl_lyrics_status.setText("")
            QMessageBox.critical(self, "가사 만들기 실패", err or "빈 결과가 나왔습니다. 다시 시도해 주세요.")
            return
        self.edit_lyrics.setPlainText(lyrics.strip() + "\n")
        self.edit_title.setText(suggest_title(lyrics))
        self.lbl_lyrics_status.setText("✅ 가사 완성 — 마음에 안 드는 줄은 바로 고치세요")

    # ------------------------------------------------------------------ 노래 만들기
    def start_song(self):
        lyrics = self.edit_lyrics.toPlainText().strip()
        if not lyrics:
            return
        if not self.acestep_py.exists():
            QMessageBox.critical(self, "작곡 엔진 없음", f"ACE-Step 실행 환경을 찾지 못했습니다:\n{self.acestep_py}")
            return
        ok, msg = self.host.check_vram_for_task("music")
        if not ok:
            QMessageBox.warning(self, "메모리 부족", msg)
            return

        voice = self._active_adapter()
        caption = build_caption(self._genre(), self._gender(), self._moods(), self.edit_extra.text())
        if voice and voice.get("trigger_tag"):
            caption = f"{voice['trigger_tag']}, {caption}"
        bpm = self._bpm()
        duration = max(estimate_duration_sec(lyrics, bpm), 30)
        takes = self.combo_takes.currentData() or 2

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = self.songs_dir / f"song-{stamp}.mp3"
        work = self.songs_dir / ".work"
        work.mkdir(exist_ok=True)
        prompt_file, lyrics_file = work / f"{stamp}.caption.txt", work / f"{stamp}.lyrics.txt"
        prompt_file.write_text(caption, encoding="utf-8")
        lyrics_file.write_text(lyrics, encoding="utf-8")
        cmd = [str(self.acestep_py), "-u", str(self.base_dir / "audio_engines" / "acestep_official_runner.py"),
               "--prompt-file", str(prompt_file), "--lyrics-file", str(lyrics_file), "--output", str(out_path),
               "--duration", str(int(duration)), "--steps", "10", "--shift", "3.0", "--bpm", str(bpm),
               "--candidates", str(takes), "--vocal-language", "ko",
               "--title", self.edit_title.text().strip() or suggest_title(lyrics) or "제목 없음"]
        if voice:
            cmd += ["--lora-path", voice["adapter_path"], "--lora-scale", f"{self.combo_strength.currentData():.2f}"]

        self._pending_meta = {
            "title": self.edit_title.text().strip() or suggest_title(lyrics) or "제목 없음",
            "idea": self.edit_idea.toPlainText().strip(), "genre": self._genre(),
            "genre_name": get_guide(self._genre())["name"], "moods": self._moods(), "gender": self._gender(),
            "tempo": self._tempo(), "bpm": bpm, "form": self.combo_form.currentIndex(),
            "voice": (self.combo_voice.currentData() or {}).get("name")
                     or ("여성" if self._gender() == "female" else "남성"),
            "voice_kind": (self.combo_voice.currentData() or {}).get("kind") or "builtin",
            "model": (self.combo_model.currentData() or {}).get("name") if isinstance(self.combo_model.currentData(), dict) else None,
            "voice_strength": self.combo_strength.currentData(),
            "extra": self.edit_extra.text().strip(), "caption": caption, "lyrics": lyrics,
            "duration": duration, "created_at": time.strftime("%Y-%m-%d %H:%M"),
        }
        self.player.stop()
        self.player.setSource(QUrl())
        self.song_worker = SongWorker(cmd, str(self.base_dir / "audio_engines"), self.host.cfg.get("text_model", "qwen3:14b"))
        self.song_worker.stage.connect(self._on_stage)
        self.song_worker.finished.connect(self._on_song_done)
        self._pct, self._pct_cap = 0.0, 5.0
        self._gen_started = time.time()
        self._stage_msg = "준비 중"
        self.progress_timer.start(500)
        self.btn_cancel.show()
        self._song_busy = True
        self.song_worker.start()
        self._update_buttons()

    def cancel_song(self):
        if self._song_busy and self.song_worker:
            self.lbl_progress.setText("중지하는 중...")
            self.song_worker.cancel()

    def _on_stage(self, lo, hi, msg):
        self._pct = max(self._pct, float(lo))
        self._pct_cap = float(hi)
        self._stage_msg = msg

    def _tick_progress(self):
        self._pct += max(0.0, self._pct_cap - self._pct) * 0.012
        self.progress.setValue(int(self._pct * 10))
        self.lbl_progress.setText(f"{self._stage_msg}... {int(self._pct)}%  ·  {_fmt_time(time.time() - self._gen_started)} 경과")

    def _on_song_done(self, ok, path, err):
        self.progress_timer.stop()
        self.btn_cancel.hide()
        self._song_busy = False
        self._update_buttons()
        if not ok:
            self.progress.setValue(0)
            self.lbl_progress.setText("노래 만들기 실패" if "중지" not in err else "중지했습니다.")
            if "중지" not in err:
                QMessageBox.critical(self, "노래 만들기 실패", err)
            return
        self.progress.setValue(1000)
        self.lbl_progress.setText(f"✅ 완성! ({_fmt_time(time.time() - self._gen_started)} 걸림)")
        meta = dict(self._pending_meta)
        run_json = Path(path + ".run.json")
        if run_json.exists():
            try:
                run = json.loads(run_json.read_text(encoding="utf-8"))
                meta["seed"] = run.get("best_seed")
            except Exception:
                pass
        Path(path).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            embed_mp3_meta(path, title=meta.get("title"), lyrics=meta.get("lyrics"))
        except Exception:
            pass
        self.refresh_library(select=path)
        self.toggle_play(force_play=True)

    def _update_buttons(self):
        busy_l, busy_s = self._lyrics_busy, self._song_busy
        self.btn_lyrics.setEnabled(not busy_l and not busy_s)
        self.btn_lyrics.setText("⏳ 가사 쓰는 중..." if busy_l else "✨ 가사 만들기")
        self.btn_song.setEnabled(not busy_l and not busy_s and bool(self.edit_lyrics.toPlainText().strip()))
        self.btn_song.setText("⏳ 노래 만드는 중..." if busy_s else "🎵 이 가사로 노래 만들기")
        if not busy_s and not self.edit_lyrics.toPlainText().strip():
            self.lbl_progress.setText("가사가 준비되면 노래를 만들 수 있습니다.")
        elif not busy_s and self.lbl_progress.text().startswith("가사가 준비되면"):
            self.lbl_progress.setText("준비 완료 — 1~3분 정도 걸립니다.")

    # ------------------------------------------------------------------ 라이브러리
    def _song_files(self):
        return sorted((p for p in self.songs_dir.glob("song-*.mp3") if "_take" not in p.stem),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def _load_meta(self, mp3: Path) -> dict:
        jp = mp3.with_suffix(".json")
        if jp.exists():
            try:
                return json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                pass
        meta = {"title": mp3.stem, "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(mp3.stat().st_mtime))}
        lp = mp3.with_suffix(".lyrics.txt")
        if lp.exists():
            meta["lyrics"] = lp.read_text(encoding="utf-8")
        return meta

    def refresh_library(self, select=None):
        self.list_songs.blockSignals(True)
        self.list_songs.clear()
        for mp3 in self._song_files():
            meta = self._load_meta(mp3)
            sub = " · ".join(x for x in [meta.get("genre_name"), meta.get("created_at", "")[5:],
                                         _fmt_time(meta["duration"]) if meta.get("duration") else None] if x)
            item = QListWidgetItem(f"{meta.get('title') or mp3.stem}\n{sub}")
            item.setData(Qt.ItemDataRole.UserRole, str(mp3))
            self.list_songs.addItem(item)
        self.list_songs.blockSignals(False)
        if self.list_songs.count() == 0:
            empty = QListWidgetItem("아직 만든 노래가 없습니다.\n가운데에서 첫 노래를 만들어 보세요.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_songs.addItem(empty)
        target = 0
        for i in range(self.list_songs.count()):
            if select and self.list_songs.item(i).data(Qt.ItemDataRole.UserRole) == str(select):
                target = i
        if self.list_songs.item(target) and self.list_songs.item(target).data(Qt.ItemDataRole.UserRole):
            self.list_songs.setCurrentRow(target)
            self._on_song_selected(self.list_songs.item(target), None)
        has = self.current_song is not None
        self.btn_reuse.setEnabled(has)
        self.btn_delete.setEnabled(has)
        self.btn_cover.setEnabled(has)

    def _selected_path(self):
        item = self.list_songs.currentItem()
        p = item.data(Qt.ItemDataRole.UserRole) if item else None
        return Path(p) if p else None

    def _on_song_selected(self, item, _prev):
        p = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not p:
            return
        mp3 = Path(p)
        self.current_song = mp3
        meta = self._load_meta(mp3)
        self.lbl_now.setText(f"🎧 {meta.get('title') or mp3.stem}")
        takes = [mp3] + sorted(mp3.parent.glob(mp3.stem + "_take*.mp3"))
        self.combo_take.blockSignals(True)
        self.combo_take.clear()
        for i, t in enumerate(takes):
            self.combo_take.addItem("버전 1 (가장 또렷함)" if i == 0 else f"버전 {i + 1}", str(t))
        self.combo_take.setVisible(len(takes) > 1)
        self.combo_take.blockSignals(False)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(mp3)))
        self.btn_reuse.setEnabled(True)
        self.btn_delete.setEnabled(True)
        self.btn_cover.setEnabled(True)
        self._show_cover(mp3)

    def _on_take_changed(self, _idx):
        path = self.combo_take.currentData()
        if not path:
            return
        playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        pos = self.player.position()
        self.player.setSource(QUrl.fromLocalFile(path))
        if playing:
            self.player.setPosition(pos)
            self.player.play()

    def toggle_play(self, force_play=False):
        if self.player.source().isEmpty():
            p = self._selected_path()
            if not p:
                return
            self.player.setSource(QUrl.fromLocalFile(str(p)))
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState and not force_play:
            self.player.pause()
        else:
            self.player.play()

    def _on_play_state(self, state):
        self.btn_play.setText("⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

    def _on_position(self, pos):
        if not self.slider_seek.isSliderDown():
            self.slider_seek.setValue(pos)
        self.lbl_time.setText(f"{_fmt_time(pos / 1000)} / {_fmt_time(self.player.duration() / 1000)}")

    def reuse_selected(self):
        p = self._selected_path()
        if not p:
            return
        meta = self._load_meta(p)
        if self.edit_lyrics.toPlainText().strip() and \
                QMessageBox.question(self, "불러오기", "편집 중인 가사를 이 노래의 가사로 바꿀까요?") != QMessageBox.StandardButton.Yes:
            return
        self.edit_lyrics.setPlainText(meta.get("lyrics", ""))
        self.edit_title.setText(meta.get("title", ""))
        self.edit_idea.setPlainText(meta.get("idea", ""))
        self.edit_extra.setText(meta.get("extra", ""))
        if meta.get("genre"):
            self._set_checked(self.genre_group, "genre", meta["genre"])
        if meta.get("tempo"):
            self._set_checked(self.tempo_group, "value", meta["tempo"])
        if isinstance(meta.get("form"), int):
            self.combo_form.setCurrentIndex(meta["form"])
        for b in self.mood_buttons:
            b.setChecked(False)
        for m in meta.get("moods", []):
            for b in self.mood_buttons:
                if b.text() == m:
                    b.setChecked(True)
        voice_name = meta.get("voice")
        if meta.get("gender") in ("female", "male") and meta.get("voice_kind") != "speech_clone":
            self.combo_voice.setCurrentIndex(0 if meta["gender"] == "female" else 1)
        if voice_name and meta.get("voice_kind") == "speech_clone":
            self._restore_combo(self.combo_voice, voice_name, self.combo_voice.currentIndex())
        model_name = meta.get("model")
        if not model_name and meta.get("voice_kind") == "song_lora":
            model_name = voice_name
        self._restore_combo(self.combo_model, model_name, 0)
        self.lbl_lyrics_status.setText("이전 노래의 가사와 설정을 불러왔습니다. 고친 뒤 다시 만들어 보세요.")

    def _show_cover(self, mp3: Path):
        data = read_cover_bytes(mp3)
        if data:
            pix = QPixmap()
            pix.loadFromData(data)
            self.lbl_cover.setPixmap(pix)
        else:
            self.lbl_cover.setPixmap(QPixmap())
            self.lbl_cover.setText("")

    def set_cover(self):
        p = self._selected_path()
        if not p:
            return
        img, _ = QFileDialog.getOpenFileName(
            self, "커버 이미지 선택", "", "이미지 (*.jpg *.jpeg *.png *.webp);;All Files (*)")
        if not img:
            return
        self.player.stop()
        ok = False
        targets = [p, *p.parent.glob(p.stem + "_take*.mp3")]
        for f in targets:
            try:
                ok = embed_mp3_meta(f, cover_path=img) or ok
            except Exception as e:
                QMessageBox.warning(self, "커버 넣기 실패", str(e))
                return
        if not ok:
            QMessageBox.warning(self, "커버 넣기 실패", "이미지를 MP3에 넣지 못했습니다.")
            return
        meta = self._load_meta(p)
        meta["cover"] = img
        p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._show_cover(p)
        self.player.setSource(QUrl.fromLocalFile(str(p)))
        QMessageBox.information(self, "커버 넣기", "선택한 노래 MP3에 커버 이미지를 넣었습니다.\n파일 정보나 음악 앱에서 앨범 표지로 보입니다.")

    def delete_selected(self):
        p = self._selected_path()
        if not p:
            return
        meta = self._load_meta(p)
        if QMessageBox.question(self, "노래 삭제", f"'{meta.get('title') or p.stem}'와 다른 버전들을 모두 삭제할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.player.stop()
        self.player.setSource(QUrl())
        for f in [p, *p.parent.glob(p.stem + "_take*.mp3"), p.with_suffix(".json"), p.with_suffix(".lyrics.txt"),
                  Path(str(p) + ".run.json")]:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        self.current_song = None
        self.lbl_now.setText("노래를 선택하세요")
        self.refresh_library()
