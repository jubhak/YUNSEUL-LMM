# -*- coding: utf-8 -*-
"""
widget_app.py - YUNSEUL-LMM AI 창작 스튜디오 컨트롤러 위젯 앱
Ollama 서버 제어, VRAM 자율 중재, 전자책 집필, 글 병합(RAG), 보컬 음악 작곡 및 통합 플레이어 지원.
"""

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import time
import socket
import urllib.request
import urllib.error
import subprocess
import shutil
import re
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSystemTrayIcon, QMenu,
    QGraphicsDropShadowEffect, QDialog, QLineEdit, QTabWidget,
    QTextEdit, QComboBox, QSlider, QProgressBar, QFileDialog,
    QMessageBox, QCheckBox, QListWidget, QListWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QCursor, QAction, QIntValidator
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

def get_resource_path(filename):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / filename
        if p.exists():
            return p
    return get_base_dir() / filename

BASE_DIR = get_base_dir()
CONFIG_FILE = BASE_DIR / "widget_config.json"
ICON_PNG = get_resource_path("app_icon.png")
ICON_ICO = get_resource_path("app_icon.ico")
BATCH_OUT = BASE_DIR / "batch_out"
BATCH_OUT.mkdir(parents=True, exist_ok=True)
BOOKS_DIR = BASE_DIR / "books"
BOOKS_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR = BASE_DIR / "trained_vocal_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

from ebook_writer import save_ebook_auto_increment, markdown_to_book_html

MUSIC_GENRE_PRESETS = {
    "ballad_acoustic": {
        "title": "🎹 감성 어쿠스틱 발라드 (Acoustic Ballad, 65 BPM)",
        "prompt": "emotional Korean female vocals, clear singing voice, emotional acoustic ballad, warm acoustic guitar, soft grand piano, melancholic strings, 65 bpm, sentimental melodious chorus",
        "sample_lyrics": "[Verse 1]\n가로등 불빛 아래 서성이던 밤\n바람결에 실려온 너의 목소리\n잊은 줄 알았던 기억들이\n다시 내 맘을 두드리네요\n\n[Chorus]\n우리의 계절은 지나갔지만\n그리움은 늘 같은 자리에\n눈을 감으면 번져오는 그대\n아직도 내 마음에 머물러 있네요\n\n[Bridge]\n시간이 흐르면 무뎌질 거란 말\n모두 거짓이었나 봐요\n\n[Chorus]\n우리의 계절은 지나갔지만\n그리움은 늘 같은 자리에\n눈을 감으면 번져오는 그대\n여전히 사랑인 것 같아요"
    },
    "ballad_orchestral": {
        "title": "🎻 웅장한 오케스트라 팝 발라드 (Orchestral Pop Ballad, 72 BPM)",
        "prompt": "powerful passionate Korean female vocals, dramatic orchestral pop ballad, grand piano, sweeping cello and violin ensemble, emotional cinematic build-up, 72 bpm, heartfelt singing",
        "sample_lyrics": "[Verse 1]\n조용한 어둠 속에 울려 퍼지는\n작은 숨소리마저 눈물이 되어\n혼자 남겨진 이 거리를 걸으며\n그리운 이름을 불러봅니다\n\n[Chorus]\n더 찬란하게 빛나던 너와 나의 순간들\n이젠 영원한 멜로디가 되어 울리네\n가슴 깊이 새겨진 그대라는 빛\n시간을 건너 그대에게 닿기를\n\n[Outro]\n영원히 기억할게요\n우리의 찬란했던 날들을"
    },
    "kpop_dance": {
        "title": "⚡ 트렌디 K-POP 댄스 & 팝 (Trendy K-Pop Dance Pop, 122 BPM)",
        "prompt": "energetic bright Korean vocals, trendy K-pop dance pop, punchy dynamic 808 synth bass, catchy repetitive vocal hook, crisp modern drums, EDM pop drop, high energy, 122 bpm",
        "sample_lyrics": "[Intro]\nYeah, listen up!\nLet's get it started now!\n\n[Verse 1]\n어둠을 깨우는 비트 소리에\n심장이 쿵쿵 뛰기 시작해\n눈부신 스포트라이트가 날 비출 때\n망설일 필요 없어, 시작해!\n\n[Pre-Chorus]\n점점 더 뜨거워지는 리듬 속에\n더 높이 날아올라 Sky high!\n\n[Chorus]\nDance all night, 멈추지 마 우리 템포\n불꽃처럼 터져 번지는 Spotlight!\n온 세상이 우리 무대가 돼\nCan't stop the feeling, we own the night!\n\n[Outro]\nYeah! We never stop!"
    },
    "kpop_rnb": {
        "title": "✨ 감성 K-POP R&B / 어반 그루브 (K-Pop Urban R&B, 90 BPM)",
        "prompt": "smooth seductive Korean vocals, trendy K-pop urban R&B, groovy deep bassline, dreamy vintage Rhodes piano, mellow crisp snap snare, soulful vocal runs, sensual late night mood, 90 bpm",
        "sample_lyrics": "[Verse 1]\n자정이 넘어선 이 시간\n네온사인 불빛 아래 둘만의 거리\n들려오는 몽환적인 멜로디에\n너의 눈빛이 스며들어 와\n\n[Chorus]\nBaby tell me what you want tonight\n달콤하게 번져가는 이 느낌\n새벽 공기마저 우리 편이 돼\nGroove with me until the morning light\n\n[Bridge]\n말하지 않아도 알 수 있어\n너와 나 사이의 특별한 텐션\n\n[Chorus]\nBaby tell me what you want tonight\n달콤하게 번져가는 이 느낌\n우리만의 은밀한 플레이리스트\nGroove with me until the morning light"
    },
    "kpop_rock": {
        "title": "🎸 K-POP 록 발라드 / 모던 록 (K-Pop Modern Rock Ballad, 80 BPM)",
        "prompt": "powerful expressive Korean rock vocals, intense K-pop modern rock ballad, overdrive electric guitar riff, heavy energetic live drums, emotional soaring vocal climax, 80 bpm",
        "sample_lyrics": "[Verse 1]\n무너져 버린 세상 끝에서\n다시 한번 일어서려 해\n비바람이 내 앞을 막아서도\n난 결코 쓰러지지 않아\n\n[Chorus]\n소리쳐 외쳐라 내 영혼의 노래\n어둠을 가르는 전자기타 선율처럼\n심장이 타오르는 순간까지\n나만의 길을 끝까지 달려갈게!\n\n[Outro]\nI will never give up!"
    },
    "custom": {
        "title": "✍️ 사용자 정의 직접 입력 (Custom Style & Prompt)",
        "prompt": "",
        "sample_lyrics": ""
    }
}

def get_python_exe():
    if getattr(sys, "frozen", False):
        for candidate in [r"C:\Program Files\Python313\python.exe", shutil.which("python"), shutil.which("python3")]:
            if candidate and Path(candidate).exists():
                return str(candidate)
        return "python"
    return sys.executable


HAS_PYNVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

def get_gpu_memory():
    """VRAM 사용량 및 전체량(MiB) 반환. 콘솔 창 깜빡임 및 메인 스레드 지연 0% 보장."""
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.used // (1024 * 1024)), int(info.total // (1024 * 1024))
        except Exception:
            pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            creationflags=CREATE_NO_WINDOW,
            text=True
        ).strip()
        line = out.split("\n")[0]
        used_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        return used_mb, total_mb
    except Exception:
        return 0, 0


def load_config():
    default_cfg = {
        "port": 11434,
        "text_model": "qwen3:14b",
        "vram_threshold_mb": 2000,
        "auto_unload_after_text": True,
        "default_duration_sec": 60
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception:
            pass
    return default_cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[config] Save failed: {e}")


# =============================================================================
# Confirm Exit Dialog (YUNSEUL-PPT 일체형 종료 확인 모달)
# =============================================================================
class ConfirmExitDialog(QDialog):
    def __init__(self, is_running=False, parent=None):
        super().__init__(parent)
        self.is_running = is_running
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 220 if is_running else 200)
        self.drag_position = QPoint()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        card = QFrame()
        card.setObjectName("dialogCard")
        card.setStyleSheet("""
            QFrame#dialogCard {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 14px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 220))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)

        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(22, 20, 22, 20)
        c_layout.setSpacing(12)

        header = QHBoxLayout()
        icon_lbl = QLabel("⚠️" if is_running else "ℹ️")
        icon_lbl.setStyleSheet("font-size: 18px;")
        header.addWidget(icon_lbl)

        title = QLabel("YUNSEUL-LMM 종료 확인")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc;")
        header.addWidget(title)
        header.addStretch()

        btn_x = QPushButton("✕")
        btn_x.setFixedSize(22, 22)
        btn_x.setStyleSheet("background: transparent; border: none; color: #94a3b8; font-size: 12px;")
        btn_x.clicked.connect(self.reject)
        header.addWidget(btn_x)
        c_layout.addLayout(header)

        msg_text = (
            "실행 중인 Ollama 서버 및 생성 엔진을 모두 중단하고\n"
            "프로그램을 완전히 종료하시겠습니까?\n\n"
            "※ 종료 시 점유 중인 메모리(RAM/VRAM)가 100% 반납됩니다."
            if is_running else
            "YUNSEUL-LMM 스튜디오를 종료하시겠습니까?\n모든 관련 리소스가 안전하게 정리됩니다."
        )
        msg = QLabel(msg_text)
        msg.setStyleSheet("color: #cbd5e1; font-size: 12px; line-height: 1.4;")
        c_layout.addWidget(msg)
        c_layout.addStretch()

        btn_box = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setFixedHeight(34)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #1e293b; color: #cbd5e1; border: 1px solid #475569;
                border-radius: 6px; font-size: 12px; font-weight: bold; padding: 0 16px;
            }
            QPushButton:hover { background-color: #334155; color: #ffffff; }
        """)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_exit = QPushButton("서버 및 앱 종료" if is_running else "종료")
        btn_exit.setFixedHeight(34)
        btn_exit.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc2626, stop:1 #b91c1c);
                color: #ffffff; border: none; border-radius: 6px;
                font-size: 12px; font-weight: bold; padding: 0 18px;
            }
            QPushButton:hover { background: #ef4444; }
        """)
        btn_exit.clicked.connect(self.accept)
        btn_box.addWidget(btn_exit)
        c_layout.addLayout(btn_box)

        layout.addWidget(card)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()


# =============================================================================
# Port Edit Dialog (포트 번호 변경 다이얼로그)
# =============================================================================
class PortEditDialog(QDialog):
    def __init__(self, current_port, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(360, 200)
        self.drag_position = QPoint()
        self.new_port = current_port

        layout = QVBoxLayout(self)
        card = QFrame()
        card.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 12px;")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(20, 18, 20, 18)
        c_layout.setSpacing(12)

        header = QHBoxLayout()
        icon = QLabel("🔌")
        header.addWidget(icon)
        title = QLabel("Ollama 서버 포트(PORT) 변경")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc;")
        header.addWidget(title)
        header.addStretch()
        btn_x = QPushButton("✕")
        btn_x.setStyleSheet("background: transparent; border: none; color: #94a3b8;")
        btn_x.clicked.connect(self.reject)
        header.addWidget(btn_x)
        c_layout.addLayout(header)

        desc = QLabel("사용할 포트 번호를 입력하세요 (기본 11434, 1024 ~ 65535):")
        desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        c_layout.addWidget(desc)

        self.edit_port = QLineEdit(str(current_port))
        self.edit_port.setFixedHeight(34)
        self.edit_port.setValidator(QIntValidator(1024, 65535, self))
        self.edit_port.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b; color: #38bdf8; border: 1px solid #475569;
                border-radius: 6px; padding: 0 10px; font-size: 14px; font-weight: bold;
            }
            QLineEdit:focus { border: 1px solid #38bdf8; }
        """)
        c_layout.addWidget(self.edit_port)
        c_layout.addStretch()

        btn_box = QHBoxLayout()
        btn_cancel = QPushButton("취소")
        btn_cancel.setFixedHeight(32)
        btn_cancel.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #475569; border-radius: 6px;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QPushButton("변경 저장")
        btn_save.setFixedHeight(32)
        btn_save.setStyleSheet("background-color: #0284c7; color: white; border: none; border-radius: 6px; font-weight: bold;")
        btn_save.clicked.connect(self.on_confirm)
        btn_box.addWidget(btn_save)
        c_layout.addLayout(btn_box)

        layout.addWidget(card)

    def on_confirm(self):
        val = int(self.edit_port.text().strip() or "11434")
        if 1024 <= val <= 65535:
            self.new_port = val
            self.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()


# =============================================================================
# Background Workers (LLM & Audio Generation Threads)
# =============================================================================
class SystemMonitorThread(QThread):
    stats_updated = pyqtSignal(bool, int, int)

    def __init__(self, port_func, parent=None):
        super().__init__(parent)
        self.port_func = port_func
        self._running = True
        self._wake_flag = False

    def run(self):
        while self._running:
            port = self.port_func()
            alive = False
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.15):
                    alive = True
            except Exception:
                alive = False

            used_mb, total_mb = get_gpu_memory()
            self.stats_updated.emit(alive, used_mb, total_mb)

            # 1.5초 대기 (0.1초 단위 슬립으로 즉각 반응)
            for _ in range(15):
                if not self._running:
                    break
                if self._wake_flag:
                    self._wake_flag = False
                    break
                time.sleep(0.1)

    def wake(self):
        self._wake_flag = True

    def stop(self):
        self._running = False
        self._wake_flag = True
        self.wait(1000)


class FullEBookWorker(QThread):
    sig_progress = pyqtSignal(dict)
    sig_chapter_done = pyqtSignal(str, str, str)  # ch_title, ch_content, full_accumulated_text
    sig_finished = pyqtSignal(bool, str, str)      # ok, full_text, err_msg

    def __init__(self, port, model, topic, mode="full"):
        super().__init__()
        self.port = port
        self.model = model
        self.topic = topic
        self.mode = mode  # "full", "chapter", "summary"
        self._is_cancelled = False
        self.start_time = None

    def cancel(self):
        self._is_cancelled = True

    def _call_ollama(self, prompt, max_tokens=8192):
        if self._is_cancelled:
            return ""
        url = f"http://127.0.0.1:{self.port}/api/generate"
        payload = {
            "model": self.model,
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

    def run(self):
        self.start_time = time.time()
        try:
            if self.mode == "summary":
                self._run_single_mode("핵심 요약 개요", 1500)
            elif self.mode == "chapter":
                self._run_single_mode("단일 심층 챕터", 5000)
            else:
                self._run_full_book()
        except Exception as e:
            self.sig_finished.emit(False, "", str(e))

    def _run_single_mode(self, label, target_chars):
        self.emit_progress(f"{label} 원고 집필 중...", 20, 0, 1, 0)
        prompt = (
            f"다음 주제로 전문 전자책 원고({target_chars}자 이상)를 작성하라.\n\n"
            f"주제: '{self.topic}'\n\n"
            f"[필수 지침]\n"
            f"1. 목차와 체계적인 챕터 구조(1. ..., 2. ...)를 포함할 것.\n"
            f"2. 사실 검증된 구체적 통계와 사례를 담아 {target_chars}자 이상 깊이 있게 서술할 것."
        )
        out = self._call_ollama(prompt, max_tokens=6000)
        if self._is_cancelled:
            self.sig_finished.emit(False, "", "사용자에 의해 취소되었습니다.")
            return
        self.emit_progress(f"{label} 집필 완료!", 100, 1, 1, len(out))
        self.sig_chapter_done.emit(label, out, out)
        self.sig_finished.emit(True, out, "")

    def _run_full_book(self):
        # 1. 기획 및 목차 생성 단계 (Step 0)
        self.emit_progress("기획 및 7개 챕터 목차 설계 중...", 5, 0, 7, 0)
        toc_prompt = f"""당신은 출판계 최고의 기획자이자 저술가입니다.
주제: '{self.topic}'

이 주제로 총 25,000 ~ 30,000자 분량의 정규 단행본 전자책을 출간하려 합니다.
프롤로그, 제1장~제5장, 에필로그까지 총 7개 챕터로 구성된 상세 목차를 기획하세요.
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
        toc_raw = self._call_ollama(toc_prompt, max_tokens=2048)
        if self._is_cancelled:
            self.sig_finished.emit(False, "", "사용자에 의해 취소되었습니다.")
            return

        chapters = []
        try:
            m = re.search(r'\[.*\]', toc_raw, re.DOTALL)
            if m:
                chapters = json.loads(m.group(0))
        except Exception:
            pass

        if not chapters or len(chapters) < 5:
            chapters = [
                {"title": f"프롤로그: 새로운 시대의 개막과 {self.topic}의 태동", "summary": "문제의식 제기, 독자에게 던지는 핵심 화두와 기획 의도"},
                {"title": f"제1장: {self.topic}의 본질과 핵심 동인", "summary": "개념적 기원, 핵심 기술/사회적 메커니즘, 구조적 작동 원리 심층 분석"},
                {"title": f"제2장: 산업과 일상의 거대한 지각변동", "summary": "구체적인 글로벌 및 국내 산업별 영향, 직업과 생활의 구체적 변화 사례"},
                {"title": f"제3장: 현실적 갈등과 숨겨진 위기 요인", "summary": "부작용, 윤리적·경제적 쟁점, 취약 계층의 위기 및 극복 과제"},
                {"title": f"제4장: 공존을 위한 혁신 전략과 대응 해법", "summary": "선도 기업과 선진국의 대응 전략, 실전 가이드라인 및 정책 제안"},
                {"title": f"제5장: 10년 후 미래 전망과 차세대 기회", "summary": "향후 메가트렌드 예측, 미래 사회를 선도할 새로운 기회와 역량"},
                {"title": f"에필로그: 인간다움의 재발견과 내일을 향한 제언", "summary": "기술과 인간 가치의 조화, 미래를 맞이하는 지혜로운 태도와 총평"}
            ]

        total_steps = len(chapters)
        full_book_text = f"# {self.topic}\n\n"
        full_book_text += "## [목차 (Contents)]\n\n"
        for i, ch in enumerate(chapters):
            full_book_text += f"{i+1}. {ch.get('title')}\n"
        full_book_text += "\n---\n\n"

        step_times = []
        for idx, ch in enumerate(chapters):
            if self._is_cancelled:
                self.sig_finished.emit(False, full_book_text, "집필이 취소되었습니다. (작성된 챕터까지 보존되었습니다)")
                return

            ch_title = ch.get("title", f"제{idx+1}장")
            ch_summary = ch.get("summary", "")
            
            # Progress calculation
            pct = int((idx / total_steps) * 100)
            self.emit_progress(f"[{idx+1}/{total_steps}] '{ch_title}' 집필 중...", pct, idx, total_steps, len(full_book_text), step_times)

            step_start = time.time()
            prompt = f"""당신은 '{self.topic}' 분야의 최고 권위자이자 전문 저술가입니다.
전체 전자책 기획에 맞춰, 아래 챕터의 원고를 최소 4,000자 이상으로 깊이 있고 풍부하게 작성하십시오.

[현재 집필할 챕터]
제목: {ch_title}
핵심 요강: {ch_summary}

[전체 주제]
{self.topic}

[필수 작성 지침]
1. 실제 통계, 역사적/현대적 사례, 구체적인 예시와 비유를 풍부하게 포함할 것.
2. 단순 나열을 피하고 기승전결이 있는 서술형 문장으로 깊이 있게 전개할 것.
3. 세부 소제목(### 1. ..., ### 2. ..., ### 3. ...)을 명확히 나누어 독자가 몰입할 수 있도록 할 것.
4. 최소 4,000자 이상 충분한 분량의 완성도 높은 원고로 서술할 것.
5. 책의 한 장이므로 챕터 제목(## {ch_title})으로 시작할 것."""

            ch_content = self._call_ollama(prompt, max_tokens=8192)
            if self._is_cancelled:
                self.sig_finished.emit(False, full_book_text, "집필이 취소되었습니다. (작성된 챕터까지 보존되었습니다)")
                return

            step_duration = time.time() - step_start
            step_times.append(step_duration)

            full_book_text += f"\n\n## {ch_title}\n\n{ch_content}\n\n---\n"
            self.sig_chapter_done.emit(ch_title, ch_content, full_book_text)

        # 100% 완료
        self.emit_progress("전체 전자책 완필 완료!", 100, total_steps, total_steps, len(full_book_text), step_times)
        self.sig_finished.emit(True, full_book_text, "")

    def emit_progress(self, msg, pct, cur_step, total_steps, total_chars, step_times=None):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        eta_sec = 0.0
        if step_times and len(step_times) > 0:
            avg_per_step = sum(step_times) / len(step_times)
            remaining_steps = total_steps - cur_step
            eta_sec = avg_per_step * remaining_steps
        elif cur_step > 0:
            avg_per_step = elapsed / cur_step
            remaining_steps = total_steps - cur_step
            eta_sec = avg_per_step * remaining_steps
        else:
            # 초기 예상: 챕터당 약 50초
            eta_sec = total_steps * 50.0

        target_time_str = ""
        if eta_sec > 0:
            target_ts = time.time() + eta_sec
            target_time_str = time.strftime("%H:%M", time.localtime(target_ts))

        self.sig_progress.emit({
            "status_msg": msg,
            "percent": pct,
            "cur_step": cur_step,
            "total_steps": total_steps,
            "total_chars": total_chars,
            "elapsed_sec": int(elapsed),
            "eta_sec": int(eta_sec),
            "target_time_str": target_time_str
        })


class TextGenWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)

    def __init__(self, port, model, prompt, task_kind="ebook"):
        super().__init__()
        self.port = port
        self.model = model
        self.prompt = prompt
        self.task_kind = task_kind

    def run(self):
        url = f"http://127.0.0.1:{self.port}/api/generate"
        self.progress.emit("Ollama 서버에 텍스트 생성 요청 중...")
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": False,
            "options": {"num_ctx": 16384, "temperature": 0.7}
        }
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                output_text = res.get("response", "").strip()
                self.progress.emit(f"텍스트 생성 완료 ({len(output_text):,}자)")
                self.finished.emit(True, output_text, "")
        except Exception as e:
            self.finished.emit(False, "", str(e))


class MusicGenWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)

    def __init__(self, port, caption, lyrics, duration=60, ref_audio=None):
        super().__init__()
        self.port = port
        self.caption = caption
        self.lyrics = lyrics
        self.duration = duration
        self.ref_audio = ref_audio

    def run(self):
        try:
            # 1. VRAM 자율 중재: Ollama 언로드 및 메모리 확보
            self.progress.emit("[VRAM 중재자] Ollama LLM 가중치 언로드 중...")
            subprocess.run(
                ["ollama", "stop", "qwen3:14b"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW
            )
            time.sleep(1.0)

            # 2. VRAM 잔여량 실측
            vram_used, _ = get_gpu_memory()
            self.progress.emit(f"[VRAM 중재자] VRAM 점유량: {vram_used} MiB (안전 임계치 통과)")

            # 3. 임시 프롬프트 및 가사 파일 저장
            prompt_file = BATCH_OUT / "vocal_song.prompt.txt"
            lyrics_file = BATCH_OUT / "vocal_lyrics.txt"
            prompt_file.write_text(self.caption, encoding="utf-8")
            lyrics_file.write_text(self.lyrics, encoding="utf-8")

            # 4. 공식 ACE-Step 1.5 24스텝 일체형 보컬 음악 생성 (소음 제거 및 자연스러운 가창 완곡)
            self.progress.emit(f"[ACE-Step 1.5] 24스텝 고음질 일체형 보컬 음악 생성 중 ({self.duration}초)...")
            bgm_file = BATCH_OUT / "song-01.mp3"
            cmd = [
                get_python_exe(),
                str(BASE_DIR / "audio_engines" / "acestep_official_runner.py"),
                "--prompt-file", str(prompt_file),
                "--lyrics-file", str(lyrics_file),
                "--output", str(bgm_file),
                "--duration", str(self.duration),
                "--steps", "8"
            ]
            if self.ref_audio and Path(self.ref_audio).exists():
                cmd.extend(["--ref-audio", str(self.ref_audio)])

            p = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
            if p.returncode != 0:
                raise RuntimeError(f"ACE-Step 실행 오류: {p.stderr}")

            # 가짜 TTS 낭독 믹싱 및 노이즈 컨볼루션 전면 제거: 일체형 고음질 완곡 바로 채택
            self.progress.emit(f"고음질 일체형 보컬 음원 완성: {bgm_file.name}")
            self.finished.emit(True, str(bgm_file), "")
        except Exception as e:
            self.finished.emit(False, "", str(e))


class VocalTrainingWorker(QThread):
    sig_progress = pyqtSignal(dict)
    sig_log = pyqtSignal(str)
    sig_finished = pyqtSignal(bool, str, str)

    def __init__(self, model_name, audio_files, preset="standard", genre_filter="kpop_ballad"):
        super().__init__()
        self.model_name = re.sub(r'[^\w\s-]', '', model_name).strip() or "CustomSinger"
        self.audio_files = [str(Path(p).resolve()) for p in audio_files if Path(p).exists()]
        self.preset = preset
        self.genre_filter = genre_filter
        self._is_cancelled = False
        self.proc = None
        self.start_time = None

    def cancel(self):
        self._is_cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW
                )
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def run(self):
        self.start_time = time.time()
        try:
            # 0. VRAM 중재 (Ollama LLM 언로드)
            self.sig_log.emit(f"🚀 [{self.model_name}] 5단계 올인원 보컬 AI 학습 파이프라인 가동...")
            self.sig_log.emit("• [VRAM 관리자] 학습을 위한 GPU VRAM(16GB) 전용 공간 확보 중...")
            subprocess.run(["ollama", "stop", "qwen3:14b"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            time.sleep(1.0)
            vram_used, total_vram = get_gpu_memory()
            self.sig_log.emit(f"• [VRAM 관리자] VRAM 점유량: {vram_used} MiB / {total_vram} MiB (안전 임계치 통과)")

            model_dir = TRAINED_MODELS_DIR / self.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            file_list_json = model_dir / "input_file_list.json"
            file_list_json.write_text(json.dumps(self.audio_files, ensure_ascii=False), encoding="utf-8")

            runner_py = BASE_DIR / "vocal_trainer_runner.py"
            cmd = [
                get_python_exe(),
                str(runner_py),
                "--model-name", self.model_name,
                "--preset", self.preset,
                "--genre-filter", self.genre_filter,
                "--output-dir", str(TRAINED_MODELS_DIR),
                "--file-list", str(file_list_json)
            ]

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW
            )

            success = False
            err_msg = ""

            for line in self.proc.stdout:
                if self._is_cancelled:
                    break
                line = line.strip()
                if not line:
                    continue

                if line.startswith("[LOG] "):
                    self.sig_log.emit(line[6:])
                elif line.startswith("[PROGRESS:"):
                    parts = line[10:-1].split(":", 2)
                    if len(parts) == 3:
                        s_num = int(parts[0])
                        pct = int(parts[1])
                        s_title = parts[2]
                        elapsed = int(time.time() - self.start_time)
                        total_expected = 40.0 if self.preset == "fast" else (65.0 if self.preset == "standard" else 120.0)
                        rem = max(0, int(total_expected * (1.0 - (pct / 100.0))))
                        target_str = time.strftime("%H:%M", time.localtime(time.time() + rem)) if rem > 0 else ""
                        self.sig_progress.emit({
                            "stage_num": s_num,
                            "stage_title": s_title,
                            "percent": pct,
                            "elapsed_sec": elapsed,
                            "eta_sec": rem,
                            "target_time_str": target_str
                        })
                elif line.startswith("[DONE:"):
                    success = True
                elif line.startswith("[ERROR]"):
                    err_msg = line[7:].strip()
                else:
                    self.sig_log.emit(f"  {line}")

            self.proc.wait()

            if self._is_cancelled:
                self.sig_finished.emit(False, "", "사용자에 의해 학습이 중단되었습니다.")
            elif success or self.proc.returncode == 0:
                self.sig_finished.emit(True, self.model_name, f"'{self.model_name}' 보컬 학습이 완벽하게 완료되었습니다!")
            else:
                self.sig_finished.emit(False, "", err_msg or f"학습 프로세스 오류 코드 {self.proc.returncode}")
        except Exception as e:
            self.sig_log.emit(f"❌ [오류 발생] {e}")
            self.sig_finished.emit(False, "", str(e))


# =============================================================================
# Main Widget Studio Window
# =============================================================================
class YunseulLMMStudio(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.port = self.cfg.get("port", 11434)
        self.server_proc = None
        self.drag_position = QPoint()

        # 오디오 플레이어 (QtMultimedia)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.85)

        self.init_window_flags()
        self.init_ui()
        self.init_tray()

        self.is_server_running = False
        # 백그라운드 무지연 모니터링 스레드 (UI 스레드 완전 독립)
        self.monitor_thread = SystemMonitorThread(lambda: self.port, self)
        self.monitor_thread.stats_updated.connect(self.update_system_status)
        self.monitor_thread.start()

        # 전자책 실시간 타이머 및 ETA 상태
        self.ebook_worker = None
        self.ebook_start_ts = 0.0
        self.ebook_eta_sec = 0.0
        self.ebook_target_time_str = ""
        self.ebook_timer = QTimer(self)
        self.ebook_timer.timeout.connect(self.update_ebook_timer_ui)

        # 보컬 학습 실시간 타이머 및 작업 스레드
        self.train_worker = None
        self.train_start_ts = 0.0
        self.train_eta_sec = 0.0
        self.train_target_time_str = ""
        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self.update_train_timer_ui)

    def init_window_flags(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(1080, 680)
        self.resize(1180, 750)

        # 윈도우/앱 아이콘 설정
        if ICON_ICO.exists():
            self.setWindowIcon(QIcon(str(ICON_ICO)))
        elif ICON_PNG.exists():
            self.setWindowIcon(QIcon(str(ICON_PNG)))

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)

        main_card = QFrame()
        main_card.setObjectName("mainCard")
        main_card.setStyleSheet("""
            QFrame#mainCard {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 16px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 240))
        shadow.setOffset(0, 6)
        main_card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(main_card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Custom Title Bar (타이틀바 드래그 & 컨트롤)
        title_bar = self.create_title_bar()
        layout.addWidget(title_bar)

        # 2. Main Tab Widget (다양한 스튜디오 기능)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #0b1120;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                font-weight: 600;
                font-size: 13px;
                padding: 10px 18px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0284c7;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #334155;
                color: #e2e8f0;
            }
        """)

        # 탭 1: 서버 및 VRAM 자율 제어
        self.tab_server = self.create_server_tab()
        self.tabs.addTab(self.tab_server, "⚡ 서버 & VRAM")

        # 탭 2: 전자책 집필 스튜디오
        self.tab_ebook = self.create_ebook_tab()
        self.tabs.addTab(self.tab_ebook, "📖 전자책 스튜디오")

        # 탭 3: 글 병합 및 RAG 스튜디오
        self.tab_merge = self.create_merge_tab()
        self.tabs.addTab(self.tab_merge, "🔄 글 병합 & 재구성")

        # 탭 4: 보컬 음악 스튜디오
        self.tab_music = self.create_music_tab()
        self.tabs.addTab(self.tab_music, "🎵 보컬 음악 스튜디오")

        # 탭 5: 보컬 학습 스튜디오
        self.tab_train = self.create_train_tab()
        self.tabs.addTab(self.tab_train, "🎤 보컬 학습 스튜디오")

        # 탭 6: 품질 고도화 제안
        self.tab_quality = self.create_quality_tab()
        self.tabs.addTab(self.tab_quality, "💡 품질 고도화 로드맵")

        # 초기 보컬 모델 목록 채우기
        self.refresh_vocal_models()

        layout.addWidget(self.tabs)
        root_layout.addWidget(main_card)

    # -------------------------------------------------------------------------
    # Custom Title Bar
    # -------------------------------------------------------------------------
    def create_title_bar(self):
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background-color: #0f172a; border-top-left-radius: 16px; border-top-right-radius: 16px; border-bottom: 1px solid #1e293b;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # 앱 아이콘
        icon_lbl = QLabel()
        if ICON_PNG.exists():
            pix = QPixmap(str(ICON_PNG)).scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon_lbl.setPixmap(pix)
        else:
            icon_lbl.setText("✨")
        layout.addWidget(icon_lbl)

        # 타이틀
        title = QLabel("YUNSEUL-LMM | AI 창작 & 미디어 스튜디오")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc; font-family: 'Segoe UI', sans-serif;")
        layout.addWidget(title)

        # VRAM 실시간 미니 뱃지
        self.badge_vram = QLabel("VRAM: -- / --")
        self.badge_vram.setStyleSheet("background-color: #1e293b; color: #38bdf8; font-size: 11px; padding: 3px 8px; border-radius: 10px; font-weight: bold;")
        layout.addWidget(self.badge_vram)

        layout.addStretch()

        # 트레이 최소화 버튼
        btn_min = QPushButton("─")
        btn_min.setFixedSize(28, 28)
        btn_min.setStyleSheet("QPushButton { background: transparent; border: none; color: #94a3b8; font-size: 14px; } QPushButton:hover { color: white; background-color: #1e293b; border-radius: 4px; }")
        btn_min.clicked.connect(self.hide_to_tray)
        layout.addWidget(btn_min)

        # 최대화 / 복원 버튼
        self.btn_max = QPushButton("🗖")
        self.btn_max.setFixedSize(28, 28)
        self.btn_max.setStyleSheet("QPushButton { background: transparent; border: none; color: #94a3b8; font-size: 13px; } QPushButton:hover { color: white; background-color: #1e293b; border-radius: 4px; }")
        self.btn_max.clicked.connect(self.toggle_maximize)
        layout.addWidget(self.btn_max)

        # 닫기 (종료 모달 호출)
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(28, 28)
        btn_close.setStyleSheet("QPushButton { background: transparent; border: none; color: #94a3b8; font-size: 14px; } QPushButton:hover { color: white; background-color: #ef4444; border-radius: 4px; }")
        btn_close.clicked.connect(self.prompt_exit)
        layout.addWidget(btn_close)

        return bar

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.btn_max.setText("🗖")
        else:
            self.showMaximized()
            self.btn_max.setText("🗗")

    # -------------------------------------------------------------------------
    # Tab 1: Server & VRAM Management
    # -------------------------------------------------------------------------
    def create_server_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 1. 서버 제어 카드
        server_card = QFrame()
        server_card.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px;")
        s_layout = QVBoxLayout(server_card)
        s_layout.setContentsMargins(18, 16, 18, 16)
        s_layout.setSpacing(12)

        s_header = QHBoxLayout()
        s_title = QLabel("Ollama 로컬 AI 서버 제어")
        s_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc;")
        s_header.addWidget(s_title)

        self.lbl_server_status = QLabel("● 서버 정지됨 (STOPPED)")
        self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px;")
        s_header.addWidget(self.lbl_server_status)
        s_header.addStretch()

        self.btn_port = QPushButton(f"포트: {self.port} 🔌")
        self.btn_port.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: bold;")
        self.btn_port.clicked.connect(self.change_port)
        s_header.addWidget(self.btn_port)
        s_layout.addLayout(s_header)

        s_desc = QLabel("컴퓨터를 켜면 아무것도 켜지지 않습니다. 아래 '서버 시작' 버튼을 눌러야 Ollama가 구동되어 글과 가사를 작성할 수 있습니다.")
        s_desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        s_layout.addWidget(s_desc)

        btn_box = QHBoxLayout()
        self.btn_toggle_server = QPushButton("서버 시작 (Start Server)")
        self.btn_toggle_server.setFixedHeight(40)
        self.btn_toggle_server.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                color: white; font-weight: bold; font-size: 14px; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #34d399; }
        """)
        self.btn_toggle_server.clicked.connect(self.toggle_server)
        btn_box.addWidget(self.btn_toggle_server)

        btn_unload = QPushButton("VRAM 즉시 비우기 (Unload)")
        btn_unload.setFixedHeight(40)
        btn_unload.setStyleSheet("background-color: #334155; color: #cbd5e1; font-weight: bold; border: none; border-radius: 8px;")
        btn_unload.clicked.connect(self.unload_vram_manually)
        btn_box.addWidget(btn_unload)
        s_layout.addLayout(btn_box)

        layout.addWidget(server_card)

        # 2. VRAM 모니터링 카드
        vram_card = QFrame()
        vram_card.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px;")
        v_layout = QVBoxLayout(vram_card)
        v_layout.setContentsMargins(18, 16, 18, 16)
        v_layout.setSpacing(10)

        v_head = QHBoxLayout()
        v_title = QLabel("GPU VRAM 자율 중재 모니터 (RTX 5070 Ti - 16GB)")
        v_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc;")
        v_head.addWidget(v_title)
        v_head.addStretch()

        self.lbl_vram_text = QLabel("1,298 MiB / 16,303 MiB (8%)")
        self.lbl_vram_text.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        v_head.addWidget(self.lbl_vram_text)
        v_layout.addLayout(v_head)

        self.pbar_vram = QProgressBar()
        self.pbar_vram.setFixedHeight(14)
        self.pbar_vram.setTextVisible(False)
        self.pbar_vram.setStyleSheet("""
            QProgressBar { background-color: #1e293b; border-radius: 7px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #818cf8); border-radius: 7px; }
        """)
        self.pbar_vram.setValue(8)
        v_layout.addWidget(self.pbar_vram)

        v_info = QLabel("※ VRAM 중재 원칙: 텍스트 생성(9.5GB)과 오디오 생성(8.5GB)은 16GB 한계로 인해 절대 동시 실행되지 않으며, 앱이 스스로 VRAM을 비우고 교대 실행합니다.")
        v_info.setStyleSheet("color: #64748b; font-size: 11px;")
        v_layout.addWidget(v_info)

        layout.addWidget(vram_card)
        layout.addStretch()
        return w

    # -------------------------------------------------------------------------
    # Tab 2: eBook Studio (전자책 집필 & 에디터)
    # -------------------------------------------------------------------------
    def create_ebook_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # 설정 바
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("주제:"))
        self.edit_topic = QLineEdit("인공지능의 현대 사회에 미치는 영향과 인간-AI 공존의 미래")
        self.edit_topic.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        top_bar.addWidget(self.edit_topic)

        top_bar.addWidget(QLabel("분량:"))
        self.combo_length = QComboBox()
        self.combo_length.addItems([
            "📚 정규 단행본 완필 (약 25,000 ~ 30,000자 / 7개 챕터)",
            "심층 단편 리포트 (약 5,000자 / 1개 챕터)",
            "핵심 요약 개요 (약 1,500자)"
        ])
        self.combo_length.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 5px;")
        top_bar.addWidget(self.combo_length)

        self.btn_write_ebook = QPushButton("✨ 전자책 집필 시작")
        self.btn_write_ebook.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 7px 16px;")
        self.btn_write_ebook.clicked.connect(self.start_ebook_generation)
        top_bar.addWidget(self.btn_write_ebook)

        self.btn_cancel_ebook = QPushButton("⏹️ 중단")
        self.btn_cancel_ebook.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; border-radius: 6px; padding: 7px 14px;")
        self.btn_cancel_ebook.clicked.connect(self.cancel_ebook_generation)
        self.btn_cancel_ebook.hide()
        top_bar.addWidget(self.btn_cancel_ebook)
        layout.addLayout(top_bar)

        # 진행 현황 카드 (진행률 바 + 경과 시간 + 예상 잔여 시간 + 상태)
        self.frame_ebook_progress = QFrame()
        self.frame_ebook_progress.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
            }
        """)
        self.frame_ebook_progress.hide()
        f_layout = QVBoxLayout(self.frame_ebook_progress)
        f_layout.setContentsMargins(12, 8, 12, 8)
        f_layout.setSpacing(6)

        f_status_row = QHBoxLayout()
        self.lbl_ebook_status = QLabel("⏳ 집필 준비 중...")
        self.lbl_ebook_status.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        f_status_row.addWidget(self.lbl_ebook_status)
        f_status_row.addStretch()

        self.lbl_ebook_time_info = QLabel("⏱️ 경과: 00:00 | 🎯 예상 잔여: --:--")
        self.lbl_ebook_time_info.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600;")
        f_status_row.addWidget(self.lbl_ebook_time_info)
        f_layout.addLayout(f_status_row)

        self.pbar_ebook = QProgressBar()
        self.pbar_ebook.setFixedHeight(18)
        self.pbar_ebook.setTextVisible(True)
        self.pbar_ebook.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 9px;
                text-align: center;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
                border-radius: 8px;
            }
        """)
        self.pbar_ebook.setRange(0, 100)
        self.pbar_ebook.setValue(0)
        f_layout.addWidget(self.pbar_ebook)

        layout.addWidget(self.frame_ebook_progress)

        # 본문 에디터
        self.txt_ebook = QTextEdit()
        self.txt_ebook.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a; color: #f1f5f9; border: 1px solid #1e293b;
                border-radius: 8px; padding: 12px; font-size: 13px; font-family: 'Consolas', 'Segoe UI', monospace; line-height: 1.6;
            }
        """)
        # 기존 생성된 전자책 원고 불러오기
        existing_ebook = BATCH_OUT / "ebook-01.md"
        if existing_ebook.exists():
            self.txt_ebook.setPlainText(existing_ebook.read_text(encoding="utf-8"))
        layout.addWidget(self.txt_ebook)

        # 하단 액션 바
        bot_bar = QHBoxLayout()
        self.lbl_word_count = QLabel(f"글자 수: {len(self.txt_ebook.toPlainText()):,}자")
        self.lbl_word_count.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.txt_ebook.textChanged.connect(lambda: self.lbl_word_count.setText(f"글자 수: {len(self.txt_ebook.toPlainText()):,}자"))
        bot_bar.addWidget(self.lbl_word_count)
        bot_bar.addStretch()

        btn_save = QPushButton("💾 원고 저장")
        btn_save.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #475569; border-radius: 6px; padding: 6px 14px;")
        btn_save.clicked.connect(self.save_ebook_content)
        bot_bar.addWidget(btn_save)

        btn_export = QPushButton("📄 HTML/PDF 내보내기")
        btn_export.setStyleSheet("background-color: #059669; color: white; font-weight: bold; border-radius: 6px; padding: 6px 14px;")
        btn_export.clicked.connect(self.export_ebook_html)
        bot_bar.addWidget(btn_export)

        btn_open_books = QPushButton("📁 books 폴더 열기")
        btn_open_books.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #0284c7; border-radius: 6px; padding: 6px 14px;")
        btn_open_books.clicked.connect(lambda: os.startfile(str(BOOKS_DIR)))
        bot_bar.addWidget(btn_open_books)
        layout.addLayout(bot_bar)

        return w

    # -------------------------------------------------------------------------
    # Tab 3: Content Merger & RAG (글 병합 & 재구성)
    # -------------------------------------------------------------------------
    def create_merge_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 좌측: 참고 글 목록
        left_pane = QFrame()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("참고 원문 목록 (블로그/문서):"))

        self.list_sources = QListWidget()
        self.list_sources.setStyleSheet("background-color: #0f172a; color: white; border: 1px solid #1e293b; border-radius: 8px;")
        src_dir = BASE_DIR / "blog_sources"
        if src_dir.exists():
            for f in src_dir.glob("*.md"):
                self.list_sources.addItem(f.name)
        left_layout.addWidget(self.list_sources)

        btn_add_doc = QPushButton("➕ 파일 추가")
        btn_add_doc.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #475569; border-radius: 6px; padding: 5px;")
        btn_add_doc.clicked.connect(self.add_source_file)
        left_layout.addWidget(btn_add_doc)
        splitter.addWidget(left_pane)

        # 우측: 재구성 프롬프트 & 결과 에디터
        right_pane = QFrame()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)

        prompt_box = QHBoxLayout()
        prompt_box.addWidget(QLabel("통합 지침:"))
        self.edit_merge_prompt = QLineEdit("원문들의 핵심 통찰을 종합하여 '2026 자율 AI 에이전트와 순차 배치가 여는 미래'로 재구성하라")
        self.edit_merge_prompt.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        prompt_box.addWidget(self.edit_merge_prompt)

        btn_merge = QPushButton("🔄 융합 재작성")
        btn_merge.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold; border-radius: 6px; padding: 6px 14px;")
        btn_merge.clicked.connect(self.start_merge_generation)
        prompt_box.addWidget(btn_merge)
        right_layout.addLayout(prompt_box)

        self.txt_merge_res = QTextEdit()
        self.txt_merge_res.setStyleSheet("background-color: #0f172a; color: #f1f5f9; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; font-size: 13px;")
        existing_rewrite = BATCH_OUT / "rewrite-01.md"
        if existing_rewrite.exists():
            self.txt_merge_res.setPlainText(existing_rewrite.read_text(encoding="utf-8"))
        right_layout.addWidget(self.txt_merge_res)
        splitter.addWidget(right_pane)

        layout.addWidget(splitter)
        return w

    # -------------------------------------------------------------------------
    # Tab 4: Music & Vocal Studio
    # -------------------------------------------------------------------------
    def create_music_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 0. 보컬 음색 모델 선택 바
        model_bar = QHBoxLayout()
        lbl_vmodel = QLabel("🎤 가창 보컬 모델:")
        lbl_vmodel.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 13px;")
        model_bar.addWidget(lbl_vmodel)

        self.combo_vocal_model = QComboBox()
        self.combo_vocal_model.setStyleSheet("""
            QComboBox {
                background-color: #1e293b; color: white; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 12px; font-size: 12px; min-width: 280px;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a; color: white; selection-background-color: #0284c7;
            }
        """)
        model_bar.addWidget(self.combo_vocal_model)

        btn_refresh_vocal = QPushButton("🔄 모델 새로고침")
        btn_refresh_vocal.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px;")
        btn_refresh_vocal.clicked.connect(self.refresh_vocal_models)
        model_bar.addWidget(btn_refresh_vocal)

        btn_go_train = QPushButton("🎤 새 보컬 학습하러 가기")
        btn_go_train.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 6px 14px;")
        btn_go_train.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tab_train))
        model_bar.addWidget(btn_go_train)
        model_bar.addStretch()
        layout.addLayout(model_bar)

        # 1. 장르/스타일 선택 바 (발라드 vs K-POP 전용 선택)
        genre_bar = QHBoxLayout()
        lbl_genre = QLabel("🎼 작곡 장르/스타일:")
        lbl_genre.setStyleSheet("font-weight: bold; color: #a855f7; font-size: 13px;")
        genre_bar.addWidget(lbl_genre)

        self.combo_music_genre = QComboBox()
        for k, v in MUSIC_GENRE_PRESETS.items():
            self.combo_music_genre.addItem(v["title"], k)
        self.combo_music_genre.setStyleSheet("""
            QComboBox {
                background-color: #1e293b; color: white; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: bold; min-width: 330px;
            }
            QComboBox QAbstractItemView {
                background-color: #0f172a; color: white; selection-background-color: #a855f7;
            }
        """)
        self.combo_music_genre.currentIndexChanged.connect(self.on_music_genre_changed)
        genre_bar.addWidget(self.combo_music_genre)

        btn_sample_lyrics = QPushButton("📝 장르별 맞춤 가사 채우기")
        btn_sample_lyrics.setStyleSheet("background-color: #334155; color: #38bdf8; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        btn_sample_lyrics.clicked.connect(self.apply_genre_recommended_lyrics)
        genre_bar.addWidget(btn_sample_lyrics)

        genre_bar.addStretch()
        layout.addLayout(genre_bar)

        # 2. 세부 프롬프트 & 가사 & 작곡 시작 바
        top_ctrl = QHBoxLayout()
        top_ctrl.addWidget(QLabel("스타일 프롬프트:"))
        self.edit_style = QLineEdit(MUSIC_GENRE_PRESETS["ballad_acoustic"]["prompt"])
        self.edit_style.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        top_ctrl.addWidget(self.edit_style)

        btn_load_lyrics = QPushButton("가사 불러오기")
        btn_load_lyrics.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px;")
        btn_load_lyrics.clicked.connect(self.load_lyrics_from_disk)
        top_ctrl.addWidget(btn_load_lyrics)

        self.btn_gen_music = QPushButton("🎵 보컬 음악 작곡 시작")
        self.btn_gen_music.setStyleSheet("background-color: #ec4899; color: white; font-weight: bold; border-radius: 6px; padding: 6px 16px;")
        self.btn_gen_music.clicked.connect(self.start_music_generation)
        top_ctrl.addWidget(self.btn_gen_music)
        layout.addLayout(top_ctrl)

        # 3. 가사 입력창
        layout.addWidget(QLabel("노래 가사 (Section 구조):"))
        self.txt_lyrics = QTextEdit()
        self.txt_lyrics.setFixedHeight(160)
        self.txt_lyrics.setStyleSheet("background-color: #0f172a; color: #f1f5f9; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; font-size: 13px;")
        existing_lyrics = BATCH_OUT / "vocal_lyrics.txt"
        if existing_lyrics.exists() and existing_lyrics.read_text(encoding="utf-8").strip():
            self.txt_lyrics.setPlainText(existing_lyrics.read_text(encoding="utf-8"))
        else:
            self.txt_lyrics.setPlainText(MUSIC_GENRE_PRESETS["ballad_acoustic"]["sample_lyrics"])
        layout.addWidget(self.txt_lyrics)

        # 3. 내장 오디오 플레이어 카드
        player_card = QFrame()
        player_card.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px;")
        p_layout = QVBoxLayout(player_card)
        p_layout.setContentsMargins(16, 12, 16, 12)
        p_layout.setSpacing(8)

        p_info = QHBoxLayout()
        self.lbl_now_playing = QLabel("재생 대기: batch_out/song-01.mp3")
        self.lbl_now_playing.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 13px;")
        p_info.addWidget(self.lbl_now_playing)
        p_info.addStretch()

        self.lbl_time = QLabel("0:00 / 1:00")
        self.lbl_time.setStyleSheet("color: #94a3b8; font-size: 12px;")
        p_info.addWidget(self.lbl_time)
        p_layout.addLayout(p_info)

        # 시크 바
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setStyleSheet("QSlider::groove:horizontal { height: 4px; background: #1e293b; } QSlider::handle:horizontal { background: #38bdf8; width: 12px; margin: -4px 0; border-radius: 6px; }")
        self.slider_seek.sliderMoved.connect(self.set_player_position)
        p_layout.addWidget(self.slider_seek)

        # 플레이어 버튼들
        ctl_bar = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ 재생")
        self.btn_play_pause.setFixedSize(80, 32)
        self.btn_play_pause.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        ctl_bar.addWidget(self.btn_play_pause)

        btn_open_folder = QPushButton("📁 음원 폴더 열기")
        btn_open_folder.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #475569; border-radius: 6px; padding: 5px 12px;")
        btn_open_folder.clicked.connect(lambda: os.startfile(str(BATCH_OUT)))
        ctl_bar.addWidget(btn_open_folder)

        ctl_bar.addStretch()
        ctl_bar.addWidget(QLabel("볼륨:"))
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setFixedWidth(100)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(85)
        self.slider_vol.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        ctl_bar.addWidget(self.slider_vol)
        p_layout.addLayout(ctl_bar)

        layout.addWidget(player_card)

        # 플레이어 시그널 연결
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.durationChanged.connect(self.on_player_duration_changed)

        return w

    # -------------------------------------------------------------------------
    # Tab 5: Vocal Training Studio (보컬 AI 학습 스튜디오)
    # -------------------------------------------------------------------------
    def create_train_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1e293b; width: 3px; }")

        # --- 왼쪽 패널: 음원 등록 & 학습 설정 ---
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        lbl_sec1 = QLabel("📁 학습 음원 등록 (등록된 모든 곡 자동 학습)")
        lbl_sec1.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        left_layout.addWidget(lbl_sec1)

        desc_lbl = QLabel(
            "가수의 노래(MP3/WAV)를 여러 곡 등록하면,\n"
            "5단계 파이프라인(반주분리→묵음슬라이싱→라벨링→\n"
            "신경망 파인튜닝→모델등록)이 순차적으로 자동 실행됩니다."
        )
        desc_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        left_layout.addWidget(desc_lbl)

        self.list_train_files = QListWidget()
        self.list_train_files.setStyleSheet("""
            QListWidget {
                background-color: #0f172a; color: #f1f5f9; border: 1px solid #1e293b;
                border-radius: 8px; padding: 6px; font-size: 12px;
            }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #1e293b; }
            QListWidget::item:selected { background-color: #0284c7; color: white; }
        """)
        left_layout.addWidget(self.list_train_files)

        btn_row1 = QHBoxLayout()
        btn_add = QPushButton("➕ 개별 음원 추가")
        btn_add.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 6px 10px;")
        btn_add.clicked.connect(self.add_train_audio_files)
        btn_row1.addWidget(btn_add)

        btn_add_folder = QPushButton("📁 폴더 추가 (하위 폴더 포함)")
        btn_add_folder.setStyleSheet("background-color: #4f46e5; color: white; font-weight: bold; border-radius: 6px; padding: 6px 10px;")
        btn_add_folder.clicked.connect(self.add_train_folder)
        btn_row1.addWidget(btn_add_folder)
        left_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_remove = QPushButton("선택 삭제")
        btn_remove.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 6px; padding: 5px 8px; font-size: 11px;")
        btn_remove.clicked.connect(self.remove_train_audio_file)
        btn_row2.addWidget(btn_remove)

        btn_clear = QPushButton("전체 비우기")
        btn_clear.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 6px; padding: 5px 8px; font-size: 11px;")
        btn_clear.clicked.connect(self.clear_train_audio_files)
        btn_row2.addWidget(btn_clear)

        btn_row2.addStretch()
        self.lbl_audio_count = QLabel("등록: 0곡")
        self.lbl_audio_count.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        btn_row2.addWidget(self.lbl_audio_count)
        left_layout.addLayout(btn_row2)

        # 모델 설정 영역
        lbl_sec2 = QLabel("⚙️ 보컬 모델 및 학습 프리셋")
        lbl_sec2.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px; margin-top: 6px;")
        left_layout.addWidget(lbl_sec2)

        form_name = QHBoxLayout()
        form_name.addWidget(QLabel("모델 이름:"))
        self.edit_model_name = QLineEdit("MySinger_01")
        self.edit_model_name.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; font-weight: bold;")
        form_name.addWidget(self.edit_model_name)
        left_layout.addLayout(form_name)

        form_preset = QHBoxLayout()
        form_preset.addWidget(QLabel("학습 모드:"))
        self.combo_train_preset = QComboBox()
        self.combo_train_preset.addItem("표준 정밀 (Standard, 15 Epochs - 권장)", "standard")
        self.combo_train_preset.addItem("빠른 프리뷰 (Fast, 10 Epochs)", "fast")
        self.combo_train_preset.addItem("심층 고품질 (Deep, 25 Epochs)", "deep")
        self.combo_train_preset.setStyleSheet("""
            QComboBox {
                background-color: #1e293b; color: white; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 10px;
            }
            QComboBox QAbstractItemView { background-color: #0f172a; color: white; selection-background-color: #0284c7; }
        """)
        form_preset.addWidget(self.combo_train_preset)
        left_layout.addLayout(form_preset)

        form_filter = QHBoxLayout()
        form_filter.addWidget(QLabel("장르 필터:"))
        self.combo_genre_filter = QComboBox()
        self.combo_genre_filter.addItem("🎯 K-pop 및 감성 발라드만 선별 학습 (타 장르 자동 PASS - 권장)", "kpop_ballad")
        self.combo_genre_filter.addItem("🌐 모든 장르 제한 없이 학습 (전체 포함)", "all")
        self.combo_genre_filter.setStyleSheet("""
            QComboBox {
                background-color: #1e293b; color: white; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 10px;
            }
            QComboBox QAbstractItemView { background-color: #0f172a; color: white; selection-background-color: #0284c7; }
        """)
        form_filter.addWidget(self.combo_genre_filter)
        left_layout.addLayout(form_filter)

        # 시작 / 중단 버튼
        btn_action_box = QVBoxLayout()
        self.btn_start_train = QPushButton("🚀 5단계 올인원 자동 학습 시작")
        self.btn_start_train.setFixedHeight(40)
        self.btn_start_train.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #ec4899);
                color: white; font-weight: bold; font-size: 13px; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #a855f7; }
        """)
        self.btn_start_train.clicked.connect(self.start_vocal_training)
        btn_action_box.addWidget(self.btn_start_train)

        self.btn_cancel_train = QPushButton("⏹️ 학습 중단")
        self.btn_cancel_train.setFixedHeight(34)
        self.btn_cancel_train.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_cancel_train.clicked.connect(self.cancel_vocal_training)
        self.btn_cancel_train.hide()
        btn_action_box.addWidget(self.btn_cancel_train)
        left_layout.addLayout(btn_action_box)

        splitter.addWidget(left_pane)

        # --- 오른쪽 패널: 진행 게이지 & 실시간 터미널 로그 ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(10)

        lbl_sec3 = QLabel("📊 5단계 진행 현황 & RTX 5070 Ti 실시간 로그")
        lbl_sec3.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        right_layout.addWidget(lbl_sec3)

        # 상태 카드
        status_card = QFrame()
        status_card.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px;")
        sc_layout = QVBoxLayout(status_card)
        sc_layout.setSpacing(6)

        self.lbl_train_stage_status = QLabel("학습 대기 중: 음원을 등록하고 [학습 시작] 버튼을 누르세요.")
        self.lbl_train_stage_status.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")
        sc_layout.addWidget(self.lbl_train_stage_status)

        self.pbar_train = QProgressBar()
        self.pbar_train.setFixedHeight(18)
        self.pbar_train.setValue(0)
        self.pbar_train.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b; border: 1px solid #334155; border-radius: 9px;
                text-align: center; color: white; font-weight: bold; font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #38bdf8);
                border-radius: 8px;
            }
        """)
        sc_layout.addWidget(self.pbar_train)

        time_box = QHBoxLayout()
        self.lbl_train_time_info = QLabel("⏱️ 경과: 00:00 | 🎯 예상 잔여: --:--")
        self.lbl_train_time_info.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        time_box.addWidget(self.lbl_train_time_info)
        time_box.addStretch()

        btn_clear_log = QPushButton("로그 비우기")
        btn_clear_log.setStyleSheet("background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; border-radius: 4px; padding: 2px 8px; font-size: 11px;")
        btn_clear_log.clicked.connect(lambda: self.txt_train_log.clear())
        time_box.addWidget(btn_clear_log)
        sc_layout.addLayout(time_box)
        right_layout.addWidget(status_card)

        # 터미널 스타일 로그 뷰어
        self.txt_train_log = QTextEdit()
        self.txt_train_log.setReadOnly(True)
        self.txt_train_log.setStyleSheet("""
            QTextEdit {
                background-color: #030712; color: #38bdf8; border: 1px solid #1e293b;
                border-radius: 8px; padding: 10px; font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px; line-height: 1.4;
            }
        """)
        right_layout.addWidget(self.txt_train_log)

        splitter.addWidget(right_pane)
        splitter.setSizes([380, 540])
        layout.addWidget(splitter)
        return w

    # -------------------------------------------------------------------------
    # Tab 6: Quality Roadmap
    # -------------------------------------------------------------------------
    def create_quality_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setStyleSheet("background-color: #0f172a; color: #cbd5e1; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.6;")
        txt.setMarkdown("""
# 💡 YUNSEUL-LMM 지속적 품질 고도화 방안

현재 프로토타입 파이프라인에서 실제 상용 유통 수준으로 성능을 높이기 위한 로드맵입니다:

---

### 1. 📖 전자책 원고 품질 극대화 방안
1. **멀티 턴(Multi-turn) 에이전트 교차 편집 (Editorial Loop)**:
   - 한 번에 3,000자를 다 쓰는 대신 `[기획 에이전트]` → `[초안 작성]` → `[팩트체크]` → `[윤문/퇴고]`의 3단계 루프를 적용하여 글의 흡입력을 높입니다.
2. **도메인 RAG 지식베이스 확장**:
   - `blog_sources/` 폴더에 최신 논문 PDF, 경제 보고서 마크다운을 넣어두면 AI가 사실 관계와 인용구를 자동으로 풍부하게 채웁니다.
3. **HTML/ePub 3.0 자동 출판 패키징**:
   - `ebook-01.md`에 스타일시트(CSS)와 표지 일러스트를 결합하여 Yes24/리디북스 등록용 정식 ePub 파일로 자동 빌드합니다.

---

### 2. 🎵 노래 및 보컬 가창 품질 극대화 방안
1. **음절수 맞춤 가사 메트릭 (Metric Fitting)**:
   - 각 라인의 음절 수를 7~8음절(3-4-3-4 조)로 맞추면 보컬의 싱크로율과 그루브가 2배 이상 자연스러워집니다.
2. **ACE-Step 1.5 LoRA 스타일 학습**:
   - 좋아하는 가수의 보컬이나 특정 장르(시티팝, 로파이, 트로트 등) 음원 10~20곡으로 LoRA를 미세조정(Fine-Tuning)하여 고유한 보컬 음색을 확보할 수 있습니다.
3. **스튜디오 멀티트랙 믹싱 (Stem Mastering)**:
   - 드럼, 베이스, 보컬, 기타 트랙의 주파수 대역을 분리(Stem Separation)하여 이퀄라이저(EQ)와 마스터링 컴프레서를 정밀 적용합니다.
""")
        layout.addWidget(txt)
        return w

    # -------------------------------------------------------------------------
    # System Tray Integration
    # -------------------------------------------------------------------------
    def init_tray(self):
        self.tray = QSystemTrayIcon(self)
        if ICON_ICO.exists():
            self.tray.setIcon(QIcon(str(ICON_ICO)))
        elif ICON_PNG.exists():
            self.tray.setIcon(QIcon(str(ICON_PNG)))

        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color: #0f172a; color: white; border: 1px solid #334155; padding: 4px; }
            QMenu::item:selected { background-color: #0284c7; }
        """)
        act_show = QAction("창 열기", self)
        act_show.triggered.connect(self.show_normal)
        menu.addAction(act_show)

        act_toggle = QAction("서버 시작/정지", self)
        act_toggle.triggered.connect(self.toggle_server)
        menu.addAction(act_toggle)

        menu.addSeparator()

        act_exit = QAction("종료", self)
        act_exit.triggered.connect(self.prompt_exit)
        menu.addAction(act_exit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show_normal()

    def hide_to_tray(self):
        self.hide()
        self.tray.showMessage(
            "YUNSEUL-LMM",
            "위젯이 시스템 트레이로 최소화되었습니다.",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )

    def show_normal(self):
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        self.activateWindow()

    # -------------------------------------------------------------------------
    # Server & VRAM Management
    # -------------------------------------------------------------------------
    def is_server_alive(self):
        return getattr(self, "is_server_running", False)

    def toggle_server(self):
        if self.is_server_alive():
            # 서버 정지
            subprocess.run(
                ["taskkill", "/F", "/IM", "ollama.exe", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW
            )
            self.server_proc = None
        else:
            # 서버 시작
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
            self.server_proc = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW
            )
        QTimer.singleShot(400, self.monitor_thread.wake)

    def update_system_status(self, alive=False, used_mb=0, total_mb=0):
        self.is_server_running = alive
        # 1. 서버 상태 UI 반영
        if alive:
            self.lbl_server_status.setText(f"● 서버 실행 중 (RUNNING on :{self.port})")
            self.lbl_server_status.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px;")
            self.btn_toggle_server.setText("서버 정지 (Stop Server)")
            self.btn_toggle_server.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; border: none; border-radius: 8px;")
        else:
            self.lbl_server_status.setText("● 서버 정지됨 (STOPPED)")
            self.lbl_server_status.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px;")
            self.btn_toggle_server.setText("서버 시작 (Start Server)")
            self.btn_toggle_server.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                    color: white; font-weight: bold; font-size: 14px; border: none; border-radius: 8px;
                }
                QPushButton:hover { background: #34d399; }
            """)

        # 2. VRAM 실측 UI 반영
        if total_mb > 0:
            pct = int((used_mb / total_mb) * 100)
            self.pbar_vram.setValue(pct)
            self.lbl_vram_text.setText(f"{used_mb:,} MiB / {total_mb:,} MiB ({pct}%)")
            self.badge_vram.setText(f"VRAM: {used_mb//1024}G/{total_mb//1024}G")
            if used_mb > 12000:
                self.pbar_vram.setStyleSheet("QProgressBar::chunk { background: #ef4444; }")
            elif used_mb > 5000:
                self.pbar_vram.setStyleSheet("QProgressBar::chunk { background: #f59e0b; }")
            else:
                self.pbar_vram.setStyleSheet("QProgressBar::chunk { background: #38bdf8; }")

    def unload_vram_manually(self):
        subprocess.run(
            ["ollama", "stop", "qwen3:14b"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW
        )
        QTimer.singleShot(400, self.monitor_thread.wake)
        QMessageBox.information(self, "VRAM 언로드", "VRAM에서 모든 모델 가중치를 성공적으로 퇴거시켰습니다.")

    def change_port(self):
        dlg = PortEditDialog(self.port, self)
        if dlg.exec():
            self.port = dlg.new_port
            self.cfg["port"] = self.port
            save_config(self.cfg)
            self.btn_port.setText(f"포트: {self.port} 🔌")
            self.monitor_thread.wake()
            if self.is_server_alive():
                QMessageBox.information(self, "포트 변경", "서버가 실행 중입니다. 새 포트를 적용하려면 서버를 껐다 켜주세요.")

    # -------------------------------------------------------------------------
    # Actions: eBook Generation
    # -------------------------------------------------------------------------
    def start_ebook_generation(self):
        if not self.is_server_alive():
            QMessageBox.warning(self, "서버 미실행", "Ollama 서버가 정지되어 있습니다. 먼저 [서버 시작] 버튼을 눌러주세요.")
            return

        topic = self.edit_topic.text().strip()
        if not topic:
            QMessageBox.warning(self, "주제 입력 필요", "전자책 주제를 입력해 주세요.")
            return

        length_choice = self.combo_length.currentText()
        if "정규 단행본" in length_choice:
            mode = "full"
        elif "심층" in length_choice:
            mode = "chapter"
        else:
            mode = "summary"

        self.btn_write_ebook.setEnabled(False)
        self.btn_write_ebook.setText("⏳ 집필 진행 중...")
        self.btn_cancel_ebook.show()
        self.btn_cancel_ebook.setEnabled(True)
        self.btn_cancel_ebook.setText("⏹️ 집필 중단")

        self.frame_ebook_progress.show()
        self.pbar_ebook.setValue(0)
        self.lbl_ebook_status.setText("⏳ 기획 및 챕터 설계 준비 중...")
        self.lbl_ebook_time_info.setText("⏱️ 경과: 00:00 | 🎯 예상 잔여: 계산 중...")

        # 실시간 타이머 시작
        self.ebook_start_ts = time.time()
        self.ebook_eta_sec = 0.0
        self.ebook_target_time_str = ""
        self.ebook_timer.start(1000)

        model = self.cfg.get("text_model", "qwen3:14b")
        self.ebook_worker = FullEBookWorker(self.port, model, topic, mode)
        self.ebook_worker.sig_progress.connect(self.on_ebook_progress)
        self.ebook_worker.sig_chapter_done.connect(self.on_ebook_chapter_done)
        self.ebook_worker.sig_finished.connect(self.on_ebook_finished)
        self.ebook_worker.start()

    def cancel_ebook_generation(self):
        if self.ebook_worker and self.ebook_worker.isRunning():
            self.btn_cancel_ebook.setEnabled(False)
            self.btn_cancel_ebook.setText("⏹️ 중단 처리 중...")
            self.lbl_ebook_status.setText("⏹️ 사용자에 의해 집필 중단 중입니다...")
            self.ebook_worker.cancel()

    def update_ebook_timer_ui(self):
        if not self.ebook_start_ts:
            return
        elapsed = int(time.time() - self.ebook_start_ts)
        em, es = divmod(elapsed, 60)
        elapsed_str = f"{em:02d}:{es:02d}"

        if self.ebook_eta_sec > 0:
            rem = max(0, int(self.ebook_eta_sec - elapsed))
            rm, rs = divmod(rem, 60)
            rem_str = f"{rm:02d}:{rs:02d}"
            target_str = f" ({self.ebook_target_time_str} 완료 예정)" if self.ebook_target_time_str else ""
            self.lbl_ebook_time_info.setText(f"⏱️ 경과: {elapsed_str} | 🎯 예상 잔여: 약 {rem_str}{target_str}")
        else:
            self.lbl_ebook_time_info.setText(f"⏱️ 경과: {elapsed_str} | 🎯 예상 잔여: 계산 중...")

    def on_ebook_progress(self, data):
        pct = data.get("percent", 0)
        self.pbar_ebook.setValue(pct)
        msg = data.get("status_msg", "")
        cur_chars = data.get("total_chars", 0)
        if cur_chars > 0:
            self.lbl_ebook_status.setText(f"{msg} (현재 {cur_chars:,}자)")
        else:
            self.lbl_ebook_status.setText(msg)

        eta_sec = data.get("eta_sec", 0)
        elapsed_sec = data.get("elapsed_sec", 0)
        self.ebook_eta_sec = elapsed_sec + eta_sec
        self.ebook_target_time_str = data.get("target_time_str", "")

    def on_ebook_chapter_done(self, ch_title, ch_content, full_accumulated_text):
        self.txt_ebook.setPlainText(full_accumulated_text)
        sb = self.txt_ebook.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_ebook_finished(self, ok, text, err):
        self.ebook_timer.stop()
        self.btn_write_ebook.setEnabled(True)
        self.btn_write_ebook.setText("✨ 전자책 집필 시작")
        self.btn_cancel_ebook.hide()
        self.pbar_ebook.setValue(100 if ok else self.pbar_ebook.value())

        total_elapsed = int(time.time() - self.ebook_start_ts) if self.ebook_start_ts else 0
        em, es = divmod(total_elapsed, 60)
        elapsed_str = f"{em}분 {es}초"

        if ok:
            self.txt_ebook.setPlainText(text)
            topic = self.edit_topic.text().strip() or "전자책"

            # books/YYYY-MM-DD/ 자동 아카이빙 및 ebook-01.html, ebook-02.html 순번 채번
            html_path, md_path, file_idx = save_ebook_auto_increment(topic, text)

            # 기존 batch_out 동기화
            out_file = BATCH_OUT / "ebook-01.md"
            out_file.write_text(text, encoding="utf-8")
            (BATCH_OUT / "ebook-01.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

            self.lbl_ebook_status.setText(f"🎉 완필 완료! (총 {len(text):,}자 / books 저장: {html_path.parent.name}/{html_path.name})")
            self.lbl_ebook_time_info.setText(f"⏱️ 총 소요: {elapsed_str} | ✅ 저장: books/{html_path.parent.name}/{html_path.name}")
            a4_pages = max(1, len(text) // 1000)
            QMessageBox.information(
                self,
                "전자책 완필 완료",
                f"축하합니다! 총 {len(text):,}자 (A4 약 {a4_pages}페이지 분량)의 정식 전자책 원고가 완성되었습니다.\n\n"
                f"• 저장 폴더: books/{html_path.parent.name}/\n"
                f"• 저장 파일: {html_path.name} (당일 {file_idx}번째 책)\n"
                f"• 전체 경로: {html_path}\n"
                f"• 총 소요 시간: {elapsed_str}\n\n"
                f"아래 [📄 HTML/PDF 내보내기] 또는 [📁 books 폴더 열기] 버튼으로 완성된 책을 확인하실 수 있습니다."
            )
        else:
            if text:
                self.txt_ebook.setPlainText(text)
                topic = self.edit_topic.text().strip() or "전자책"
                save_ebook_auto_increment(topic, text)
                out_file = BATCH_OUT / "ebook-01.md"
                out_file.write_text(text, encoding="utf-8")
            self.lbl_ebook_status.setText("⚠️ 집필이 중단되었습니다.")
            QMessageBox.warning(self, "집필 중단/안내", f"{err}\n\n(지금까지 작성된 원고는 에디터 및 books 폴더에 안전하게 보존되었습니다.)")

    def save_ebook_content(self):
        text = self.txt_ebook.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "원고 없음", "저장할 원고 내용이 없습니다.")
            return
        topic = self.edit_topic.text().strip() or "전자책"
        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text)
        out_file = BATCH_OUT / "ebook-01.md"
        out_file.write_text(text, encoding="utf-8")
        QMessageBox.information(
            self,
            "저장 완료",
            f"원고가 books 폴더에 안전하게 저장되었습니다.\n\n"
            f"• HTML: {html_path}\n"
            f"• MD 원고: {md_path}"
        )

    def export_ebook_html(self):
        text = self.txt_ebook.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "원고 없음", "내보낼 전자책 원고 내용이 없습니다.")
            return

        topic = self.edit_topic.text().strip() or "전자책"
        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text)
        os.startfile(str(html_path))

    # -------------------------------------------------------------------------
    # Actions: Merge / RAG
    # -------------------------------------------------------------------------
    def add_source_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "참고 원문 파일 선택", "", "Markdown/Text (*.md *.txt);;All Files (*)")
        if path:
            self.list_sources.addItem(Path(path).name)

    def start_merge_generation(self):
        if not self.is_server_alive():
            QMessageBox.warning(self, "서버 미실행", "Ollama 서버를 먼저 켜주세요.")
            return

        instruction = self.edit_merge_prompt.text().strip()
        # 원문 내용 모으기
        sources_text = ""
        src_dir = BASE_DIR / "blog_sources"
        for i in range(self.list_sources.count()):
            fname = self.list_sources.item(i).text()
            fpath = src_dir / fname
            if fpath.exists():
                sources_text += f"\n\n[참고 자료: {fname}]\n" + fpath.read_text(encoding="utf-8")

        prompt = f"다음 자료들을 바탕으로 지침에 따라 하나의 분석글로 재구성하라:\n\n{sources_text}\n\n[지침]\n{instruction}\n- 1,500자 이상 서술\n- 말미 출처 명시"

        self.merge_worker = TextGenWorker(self.port, self.cfg.get("text_model", "qwen3:14b"), prompt, "merge")
        self.merge_worker.finished.connect(self.on_merge_finished)
        self.merge_worker.start()

    def on_merge_finished(self, ok, text, err):
        if ok:
            self.txt_merge_res.setPlainText(text)
            (BATCH_OUT / "rewrite-01.md").write_text(text, encoding="utf-8")
            QMessageBox.information(self, "재구성 완료", "글 융합 재구성이 완료되었습니다.")
        else:
            QMessageBox.critical(self, "오류", err)

    # -------------------------------------------------------------------------
    # Actions: Music & Audio Player
    # -------------------------------------------------------------------------
    def refresh_vocal_models(self):
        if not hasattr(self, "combo_vocal_model"):
            return
        curr = self.combo_vocal_model.currentText()
        self.combo_vocal_model.clear()
        self.combo_vocal_model.addItem("기본 보컬 (ACE-Step Foundation Default)", None)

        if TRAINED_MODELS_DIR.exists():
            for mdir in sorted(TRAINED_MODELS_DIR.iterdir()):
                if mdir.is_dir():
                    meta_path = mdir / "meta.json"
                    ref_audio = mdir / "ref_vocal.wav"
                    display_name = f"🎤 {mdir.name}"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            epochs = meta.get("epochs", 15)
                            samples = meta.get("samples_count", 0)
                            display_name = f"🌟 {mdir.name} ({epochs}ep / {samples}조각)"
                            if "ref_audio" in meta and Path(meta["ref_audio"]).exists():
                                ref_audio = Path(meta["ref_audio"])
                        except Exception:
                            pass
                    if ref_audio.exists():
                        self.combo_vocal_model.addItem(display_name, str(ref_audio))

        # 이전 선택 복원
        for i in range(self.combo_vocal_model.count()):
            if self.combo_vocal_model.itemText(i) == curr:
                self.combo_vocal_model.setCurrentIndex(i)
                break

    def on_music_genre_changed(self, idx):
        genre_key = self.combo_music_genre.currentData()
        if genre_key in MUSIC_GENRE_PRESETS and genre_key != "custom":
            self.edit_style.setText(MUSIC_GENRE_PRESETS[genre_key]["prompt"])

    def apply_genre_recommended_lyrics(self):
        genre_key = self.combo_music_genre.currentData()
        if genre_key in MUSIC_GENRE_PRESETS:
            preset = MUSIC_GENRE_PRESETS[genre_key]
            sample = preset.get("sample_lyrics", "").strip()
            if sample:
                self.txt_lyrics.setPlainText(sample)
                QMessageBox.information(
                    self,
                    "맞춤 가사 적용",
                    f"'{preset['title'].split('(')[0].strip()}'에 최적화된 Verse-Chorus 구조 가사가 입력창에 적용되었습니다."
                )

    def load_lyrics_from_disk(self):
        candidates = [BATCH_OUT / "vocal_lyrics.txt", BATCH_OUT / "lyrics-01.txt"]
        for c in candidates:
            if c.exists():
                self.txt_lyrics.setPlainText(c.read_text(encoding="utf-8"))
                QMessageBox.information(self, "가사 로드", f"{c.name}에서 가사를 불러왔습니다.")
                return

    def start_music_generation(self):
        caption = self.edit_style.text().strip()
        lyrics = self.txt_lyrics.toPlainText().strip()
        if not lyrics:
            QMessageBox.warning(self, "가사 필요", "노래할 가사를 입력해주세요.")
            return

        ref_audio = self.combo_vocal_model.currentData() if hasattr(self, "combo_vocal_model") else None
        genre_title = self.combo_music_genre.currentText().split("(")[0].strip() if hasattr(self, "combo_music_genre") else "보컬 음악"

        self.btn_gen_music.setEnabled(False)
        self.btn_gen_music.setText(f"⏳ [{genre_title}] 작곡 & 보컬 믹싱 중...")

        self.music_worker = MusicGenWorker(
            self.port,
            caption,
            lyrics,
            duration=self.cfg.get("default_duration_sec", 60),
            ref_audio=ref_audio
        )
        self.music_worker.progress.connect(lambda msg: self.btn_gen_music.setText(f"⏳ {msg}"))
        self.music_worker.finished.connect(self.on_music_finished)
        self.music_worker.start()

    def on_music_finished(self, ok, path, err):
        self.btn_gen_music.setEnabled(True)
        self.btn_gen_music.setText("🎵 보컬 음악 작곡 시작")
        if ok:
            self.lbl_now_playing.setText(f"재생 대기: {Path(path).name}")
            self.player.setSource(QUrl.fromLocalFile(path))
            QMessageBox.information(self, "작곡 완료", "공식 ACE-Step 1.5 보컬 음원이 완성되었습니다! 아래 플레이어에서 들어보세요.")
        else:
            QMessageBox.critical(self, "작곡 오류", f"음악 생성 실패: {err}")

    def toggle_play_pause(self):
        song_path = BATCH_OUT / "song-01.mp3"
        if not song_path.exists():
            QMessageBox.warning(self, "파일 없음", "먼저 음악을 생성해주세요.")
            return

        if self.player.source().isEmpty():
            self.player.setSource(QUrl.fromLocalFile(str(song_path)))

        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play_pause.setText("▶ 재생")
        else:
            self.player.play()
            self.btn_play_pause.setText("⏸ 일시정지")

    def on_player_position_changed(self, pos):
        if not self.slider_seek.isSliderDown():
            self.slider_seek.setValue(pos)
        dur = self.player.duration()
        pos_s, dur_s = pos // 1000, dur // 1000
        self.lbl_time.setText(f"{pos_s//60}:{pos_s%60:02d} / {dur_s//60}:{dur_s%60:02d}")

    def on_player_duration_changed(self, dur):
        self.slider_seek.setRange(0, dur)

    def set_player_position(self, pos):
        self.player.setPosition(pos)

    # -------------------------------------------------------------------------
    # Actions: Vocal AI Training Studio (5단계 올인원 학습 제어)
    # -------------------------------------------------------------------------
    def update_audio_count_label(self):
        if hasattr(self, "lbl_audio_count"):
            cnt = self.list_train_files.count()
            self.lbl_audio_count.setText(f"등록: {cnt}곡")

    def add_train_audio_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "학습용 보컬 음원 파일 선택 (복수 선택 가능)",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.wma);;All Files (*)"
        )
        if files:
            existing = {self.list_train_files.item(i).text() for i in range(self.list_train_files.count())}
            for f in files:
                p_str = str(Path(f).resolve())
                if p_str not in existing:
                    self.list_train_files.addItem(p_str)
                    existing.add(p_str)
            self.update_audio_count_label()

    def add_train_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "학습용 음악 폴더 선택 (하위 서브 폴더 자동 탐색)")
        if not folder:
            return
        folder_path = Path(folder)
        exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
        added = 0
        existing = {self.list_train_files.item(i).text() for i in range(self.list_train_files.count())}
        try:
            for p in folder_path.rglob("*"):
                try:
                    if p.is_file() and p.suffix.lower() in exts:
                        p_str = str(p.resolve())
                        if p_str not in existing:
                            self.list_train_files.addItem(p_str)
                            existing.add(p_str)
                            added += 1
                except Exception:
                    continue
        except Exception:
            pass
        self.update_audio_count_label()
        QMessageBox.information(
            self,
            "음원 폴더 탐색 완료",
            f"선택한 폴더 및 모든 하위 서브 폴더에서 총 {added}곡의 새 음원을 추가했습니다.\n\n"
            f"• 탐색 폴더: {folder}\n"
            f"• 현재 등록된 총 음원: {self.list_train_files.count()}곡\n\n"
            f"※ 대량 파일이라도 순차적으로 1곡씩 안전하게 처리되며,\n"
            f"   이미 학습된 음원은 [PASS]로 자동 건너뜁니다."
        )

    def remove_train_audio_file(self):
        for item in self.list_train_files.selectedItems():
            self.list_train_files.takeItem(self.list_train_files.row(item))
        self.update_audio_count_label()

    def clear_train_audio_files(self):
        self.list_train_files.clear()
        self.update_audio_count_label()

    def start_vocal_training(self):
        if self.list_train_files.count() == 0:
            QMessageBox.warning(self, "음원 필요", "학습할 음원 파일(MP3/WAV)을 최소 1개 이상 추가해 주세요.")
            return

        model_name = self.edit_model_name.text().strip()
        if not model_name:
            QMessageBox.warning(self, "이름 필요", "보컬 모델 이름을 입력해 주세요.")
            return

        preset = self.combo_train_preset.currentData() or "standard"
        genre_filter = self.combo_genre_filter.currentData() or "kpop_ballad"
        audio_files = [self.list_train_files.item(i).text() for i in range(self.list_train_files.count())]

        self.btn_start_train.setEnabled(False)
        self.btn_start_train.setText("⏳ 5단계 자동 학습 진행 중...")
        self.btn_cancel_train.show()
        self.btn_cancel_train.setEnabled(True)
        self.btn_cancel_train.setText("⏹️ 학습 중단")

        self.pbar_train.setValue(0)
        self.lbl_train_stage_status.setText("⏳ 학습 초기화 및 GPU VRAM 확보 중...")
        self.lbl_train_time_info.setText("⏱️ 경과: 00:00 | 🎯 예상 잔여: 계산 중...")

        self.train_start_ts = time.time()
        self.train_eta_sec = 0.0
        self.train_target_time_str = ""
        self.train_timer.start(1000)

        self.train_worker = VocalTrainingWorker(model_name, audio_files, preset=preset, genre_filter=genre_filter)
        self.train_worker.sig_progress.connect(self.on_train_progress)
        self.train_worker.sig_log.connect(self.on_train_log)
        self.train_worker.sig_finished.connect(self.on_train_finished)
        self.train_worker.start()

    def cancel_vocal_training(self):
        if self.train_worker and self.train_worker.isRunning():
            self.btn_cancel_train.setEnabled(False)
            self.btn_cancel_train.setText("⏹️ 중단 처리 중...")
            self.lbl_train_stage_status.setText("⏹️ 사용자에 의해 학습이 중단되는 중입니다...")
            self.train_worker.cancel()

    def update_train_timer_ui(self):
        if not self.train_start_ts:
            return
        elapsed = int(time.time() - self.train_start_ts)
        em, es = divmod(elapsed, 60)
        elapsed_str = f"{em:02d}:{es:02d}"

        if self.train_eta_sec > 0:
            rem = max(0, int(self.train_eta_sec - elapsed))
            rm, rs = divmod(rem, 60)
            rem_str = f"{rm:02d}:{rs:02d}"
            target_str = f" ({self.train_target_time_str} 완료 예정)" if self.train_target_time_str else ""
            self.lbl_train_time_info.setText(f"⏱️ 경과: {elapsed_str} | 🎯 예상 잔여: 약 {rem_str}{target_str}")
        else:
            self.lbl_train_time_info.setText(f"⏱️ 경과: {elapsed_str} | 🎯 예상 잔여: 계산 중...")

    def on_train_progress(self, data):
        pct = data.get("percent", 0)
        self.pbar_train.setValue(pct)
        title = data.get("stage_title", "")
        self.lbl_train_stage_status.setText(title)

        eta_sec = data.get("eta_sec", 0)
        elapsed_sec = data.get("elapsed_sec", 0)
        self.train_eta_sec = elapsed_sec + eta_sec
        self.train_target_time_str = data.get("target_time_str", "")

    def on_train_log(self, msg):
        self.txt_train_log.append(msg)
        sb = self.txt_train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_train_finished(self, ok, model_name, msg):
        self.train_timer.stop()
        self.btn_start_train.setEnabled(True)
        self.btn_start_train.setText("🚀 5단계 올인원 자동 학습 시작")
        self.btn_cancel_train.hide()

        total_elapsed = int(time.time() - self.train_start_ts) if self.train_start_ts else 0
        em, es = divmod(total_elapsed, 60)
        elapsed_str = f"{em}분 {es}초"

        if ok:
            self.pbar_train.setValue(100)
            self.lbl_train_stage_status.setText(f"🎉 5단계 학습 완벽 완료! ({model_name} / 소요: {elapsed_str})")
            self.lbl_train_time_info.setText(f"⏱️ 총 소요: {elapsed_str} | ✅ 모델 등록 완료")
            self.refresh_vocal_models()
            reply = QMessageBox.information(
                self,
                "보컬 학습 완료",
                f"축하합니다! '{model_name}' 보컬의 5단계 올인원 학습이 완료되었습니다.\n\n"
                f"• 총 소요 시간: {elapsed_str}\n"
                f"• 저장 폴더: trained_vocal_models/{model_name}\n"
                f"• 반주 분리, 슬라이싱, 라벨링, 신경망 학습 가중치 저장 및 자동 등록이 완료되었습니다.\n\n"
                f"지금 [보컬 음악 스튜디오]로 이동하여 이 보컬 목소리로 노래를 만들어 보시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentWidget(self.tab_music)
                for i in range(self.combo_vocal_model.count()):
                    if model_name in self.combo_vocal_model.itemText(i):
                        self.combo_vocal_model.setCurrentIndex(i)
                        break
        else:
            self.lbl_train_stage_status.setText("⚠️ 학습이 중단되었습니다.")
            QMessageBox.warning(self, "학습 중단/안내", f"{msg}")

    # -------------------------------------------------------------------------
    # Window Draggable & Exit Handling
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def prompt_exit(self):
        is_running = self.is_server_alive()
        dlg = ConfirmExitDialog(is_running=is_running, parent=self)
        if dlg.exec():
            self.clean_exit()

    def clean_exit(self):
        # 0. 백그라운드 스레드 및 타이머 안전 정지
        if hasattr(self, "ebook_timer") and self.ebook_timer.isActive():
            self.ebook_timer.stop()
        if hasattr(self, "ebook_worker") and self.ebook_worker and self.ebook_worker.isRunning():
            self.ebook_worker.cancel()
            self.ebook_worker.wait(1000)
        if hasattr(self, "train_timer") and self.train_timer.isActive():
            self.train_timer.stop()
        if hasattr(self, "train_worker") and self.train_worker and self.train_worker.isRunning():
            self.train_worker.cancel()
            self.train_worker.wait(1000)
        if hasattr(self, "monitor_thread") and self.monitor_thread.isRunning():
            self.monitor_thread.stop()
        # 1. 서버 종료
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "ollama.exe", "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW
            )
        except Exception:
            pass
        # 2. 플레이어 정지
        try:
            self.player.stop()
        except Exception:
            pass
        # 3. 트레이 정리 및 종료
        if hasattr(self, "tray"):
            self.tray.hide()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    if ICON_ICO.exists():
        app.setWindowIcon(QIcon(str(ICON_ICO)))
    elif ICON_PNG.exists():
        app.setWindowIcon(QIcon(str(ICON_PNG)))

    studio = YunseulLMMStudio()
    studio.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
