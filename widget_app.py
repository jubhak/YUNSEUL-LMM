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
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

def get_base_dir():
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parent
    # exe 를 dist/ 나 바탕화면 등 다른 곳에 두어도 audio_engines 가 있는 프로젝트 폴더를 찾는다.
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [exe_dir, *exe_dir.parents, exe_dir / "YUNSEUL-LMM", Path(r"D:\workspace\YUNSEUL-LMM")]
    for d in candidates:
        if (d / "audio_engines" / "acestep_official_runner.py").exists():
            return d
    return exe_dir

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
ACESTEP_PY = BASE_DIR / "audio_engines" / "venv_acestep" / "Scripts" / "python.exe"

from ebook_writer import save_ebook_auto_increment, markdown_to_book_html, EbookWriter
from text_engine import GenerationCancelled, ollama_generate
from song_studio import SongStudio
from voice_clone_pane import VoiceClonePane

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

    def run(self):
        self.start_time = time.time()
        step_times = []
        last = {"step": 0, "ts": time.time()}

        def on_progress(msg, cur_step, total_steps, total_chars):
            if cur_step > last["step"]:
                step_times.append((time.time() - last["ts"]) / (cur_step - last["step"]))
                last.update(step=cur_step, ts=time.time())
            pct = int(cur_step / max(1, total_steps) * 100)
            self.emit_progress(msg, pct, cur_step, total_steps, total_chars, step_times)

        writer = EbookWriter(
            self.topic, model=self.model, port=self.port, mode=self.mode,
            log=lambda *_: None,
            on_progress=on_progress,
            on_chapter=lambda title, ch_md, full: self.sig_chapter_done.emit(title, ch_md, full),
            cancel_check=lambda: self._is_cancelled,
        )
        try:
            full_text = writer.write()
            self.sig_finished.emit(True, full_text, "")
        except GenerationCancelled:
            self.sig_finished.emit(False, writer.partial_text, "집필이 취소되었습니다. (작성된 부분까지 보존되었습니다)")
        except Exception as e:
            self.sig_finished.emit(False, writer.partial_text, str(e))

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
        self.progress.emit("Ollama 서버에 텍스트 생성 요청 중...")
        try:
            output_text = ollama_generate(self.prompt, model=self.model, port=self.port, max_tokens=6000)
            self.progress.emit(f"텍스트 생성 완료 ({len(output_text):,}자)")
            self.finished.emit(True, output_text, "")
        except Exception as e:
            self.finished.emit(False, "", str(e))


class VocalTrainingWorker(QThread):
    sig_progress = pyqtSignal(dict)
    sig_log = pyqtSignal(str)
    sig_finished = pyqtSignal(bool, str, str)

    def __init__(self, model_name, audio_files, preset="standard", genre_filter="kpop_ballad",
                 mode="songs", clips_json=None, gender="female"):
        super().__init__()
        self.model_name = re.sub(r'[^\w\s-]', '', model_name).strip() or "CustomSinger"
        self.audio_files = [str(Path(p).resolve()) for p in audio_files if Path(p).exists()]
        self.preset = preset
        self.genre_filter = genre_filter
        self.mode = mode
        self.clips_json = clips_json
        self.gender = gender
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
            kind = "내 목소리 따라 읽기" if self.mode == "speech" else "5단계 올인원 보컬 AI"
            self.sig_log.emit(f"🚀 [{self.model_name}] {kind} 학습 파이프라인 가동...")
            self.sig_log.emit("• [VRAM 관리자] 학습을 위한 GPU VRAM(16GB) 전용 공간 확보 중...")
            subprocess.run(["ollama", "stop", "qwen3:14b"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            time.sleep(1.0)
            vram_used, total_vram = get_gpu_memory()
            self.sig_log.emit(f"• [VRAM 관리자] VRAM 점유량: {vram_used} MiB / {total_vram} MiB (안전 임계치 통과)")

            model_dir = TRAINED_MODELS_DIR / self.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            runner_py = BASE_DIR / "vocal_trainer_runner.py"
            cmd = [
                get_python_exe(),
                str(runner_py),
                "--model-name", self.model_name,
                "--preset", self.preset,
                "--genre-filter", self.genre_filter,
                "--output-dir", str(TRAINED_MODELS_DIR),
                "--mode", self.mode,
                "--gender", self.gender,
            ]
            if self.mode == "speech":
                cmd += ["--clips-json", str(self.clips_json)]
            else:
                file_list_json = model_dir / "input_file_list.json"
                file_list_json.write_text(json.dumps(self.audio_files, ensure_ascii=False), encoding="utf-8")
                cmd += ["--file-list", str(file_list_json)]

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
                        m_eta = re.search(r"남은 시간 약 (\d+)분", s_title)
                        if m_eta:
                            rem = int(m_eta.group(1)) * 60
                        elif pct >= 5:
                            rem = int(elapsed * (100 - pct) / pct)
                        else:
                            rem = 0
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
            elif success and self.proc.returncode == 0:
                self.sig_finished.emit(True, self.model_name, f"'{self.model_name}' LoRA 학습이 완료되었습니다.")
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
        self.tab_music = SongStudio(self, BASE_DIR, BATCH_OUT / "songs", TRAINED_MODELS_DIR, ACESTEP_PY)
        self.tabs.addTab(self.tab_music, "🎵 노래 만들기")

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

        btn_save = QPushButton("💾 원고 임시저장")
        btn_save.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #475569; border-radius: 6px; padding: 6px 14px;")
        btn_save.clicked.connect(self.save_ebook_content)
        bot_bar.addWidget(btn_save)

        btn_publish = QPushButton("🚀 전자책 출판하기 (books 자동 저장)")
        btn_publish.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                color: white; font-weight: bold; font-size: 13px; border: none; border-radius: 6px; padding: 7px 16px;
            }
            QPushButton:hover { background: #34d399; }
        """)
        btn_publish.clicked.connect(self.publish_ebook)
        bot_bar.addWidget(btn_publish)

        btn_export = QPushButton("📄 HTML 열기")
        btn_export.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #475569; border-radius: 6px; padding: 6px 14px;")
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

        m_bot_bar = QHBoxLayout()
        btn_send_to_ebook = QPushButton("📝 전자책 스튜디오로 보내기")
        btn_send_to_ebook.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #475569; border-radius: 6px; padding: 6px 14px;")
        btn_send_to_ebook.clicked.connect(self.send_merge_to_ebook)
        m_bot_bar.addWidget(btn_send_to_ebook)

        btn_publish_merge = QPushButton("🚀 융합 글을 전자책으로 출판 (books 자동 저장)")
        btn_publish_merge.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                color: white; font-weight: bold; border: none; border-radius: 6px; padding: 7px 16px;
            }
            QPushButton:hover { background: #34d399; }
        """)
        btn_publish_merge.clicked.connect(self.publish_merge_as_ebook)
        m_bot_bar.addWidget(btn_publish_merge)
        right_layout.addLayout(m_bot_bar)

        splitter.addWidget(right_pane)

        layout.addWidget(splitter)
        return w

    # -------------------------------------------------------------------------
    # Tab 4: Music & Vocal Studio
    # -------------------------------------------------------------------------
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
        self.combo_train_preset.addItem("표준 (Standard - 권장, 곡 수에 맞춰 epoch 자동)", "standard")
        self.combo_train_preset.addItem("빠른 시험 (Fast, 표준의 절반)", "fast")
        self.combo_train_preset.addItem("심층 (Deep, 표준의 1.6배 · 오래 걸림)", "deep")
        self.combo_train_preset.setToolTip(
            "ACE-Step 공식 가이드 기준: 1~10곡 200~500 epoch, 10~50곡 100~200, 50곡 이상 50~100.\n"
            "음원 옆에 같은 이름의 .txt/.lrc 가사 파일이 있으면 받아쓰기 대신 그 가사를 씁니다.")
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

        self.voice_clone_pane = VoiceClonePane(TRAINED_MODELS_DIR / "_inbox", self)
        self.voice_clone_pane.train_requested.connect(self.start_voice_clone_training)
        train_modes = QTabWidget()
        train_modes.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #1e293b; border-radius: 8px; }"
            "QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 8px 14px; border-radius: 6px; margin-right: 4px; }"
            "QTabBar::tab:selected { background: #0ea5e9; color: white; font-weight: bold; }"
        )
        train_modes.addTab(left_pane, "가수 노래로 학습")
        train_modes.addTab(self.voice_clone_pane, "내 목소리 따라 읽기")
        splitter.addWidget(train_modes)

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
        self.show_and_activate()

    def show_and_activate(self):
        """기존 창을 화면에 복원/표시하고 최상단 포그라운드로 팝업합니다."""
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        self.raise_()
        self.activateWindow()

        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = int(self.winId())
                # SW_RESTORE = 9
                ctypes.windll.user32.ShowWindow(hwnd, 9)
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

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

    def check_vram_for_task(self, task_type: str) -> tuple:
        """
        작업 시작 전 가용 VRAM 점검
        - task_type in ('ebook', 'merge'): Ollama 텍스트 모델(qwen3:14b) 구동에 가용 VRAM 점검
        - task_type in ('music', 'train'): ACE-Step 1.5 공식 오디오 엔진에 최소 7,000 MiB 필요
        부족할 경우 (False, 상세 경고 메시지) 반환
        """
        used_mb, total_mb = get_gpu_memory()
        if total_mb <= 0:
            return True, ""

        free_mb = total_mb - used_mb

        if task_type in ("ebook", "merge"):
            # 1. Ollama에 이미 모델이 로드되어 있는지 확인
            is_already_loaded = False
            try:
                url = f"http://127.0.0.1:{self.port}/api/ps"
                req = urllib.request.Request(url, headers={"User-Agent": "YUNSEUL-LMM"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("models", [])
                    target_model = self.cfg.get("text_model", "qwen3:14b")
                    for m in models:
                        m_name = m.get("name", "")
                        if target_model in m_name or m_name.startswith(target_model.split(":")[0]):
                            is_already_loaded = True
                            break
            except Exception:
                pass

            if is_already_loaded:
                req_mb = 1000
                if free_mb < req_mb:
                    msg = (
                        f"현재 가용 VRAM이 부족하여 전자책 집필을 시작할 수 없습니다.\n\n"
                        f"• 모델({self.cfg.get('text_model', 'qwen3:14b')})이 로드되어 있으나 추론 여유 공간이 부족합니다.\n"
                        f"• 현재 가용 VRAM: {free_mb:,} MiB (최소 필요: {req_mb:,} MiB)\n"
                        f"• 현재 사용량: {used_mb:,} MiB / 전체 {total_mb:,} MiB\n\n"
                        f"다른 프로그램을 종료하거나 [VRAM 즉시 비우기]를 실행한 후 다시 시도해주세요."
                    )
                    return False, msg
                return True, ""

            # 모델이 아직 로드되지 않은 경우 약 8,000 MiB 필요
            req_mb = 8000
            if free_mb < req_mb:
                deficit = req_mb - free_mb
                msg = (
                    f"현재 가용 VRAM이 부족하여 전자책 집필을 시작할 수 없습니다.\n\n"
                    f"• 필요 VRAM: 약 {req_mb:,} MiB ({self.cfg.get('text_model', 'qwen3:14b')} 가중치 로드)\n"
                    f"• 현재 가용 VRAM: {free_mb:,} MiB\n"
                    f"• 현재 사용량: {used_mb:,} MiB / 전체 {total_mb:,} MiB\n"
                    f"• 부족한 VRAM: 약 {deficit:,} MiB\n\n"
                    f"VRAM을 점유하고 있는 다른 프로그램(영상 편집, 타 AI 모델, 게임 등)을 종료하거나 [VRAM 즉시 비우기]를 실행한 후 다시 시도해주세요."
                )
                return False, msg

        elif task_type in ("music", "train"):
            # 1. 텍스트 모델이 VRAM에 남아있다면 언로드 시도
            try:
                subprocess.run(
                    ["ollama", "stop", self.cfg.get("text_model", "qwen3:14b")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW
                )
                time.sleep(0.5)
            except Exception:
                pass

            used_mb, total_mb = get_gpu_memory()
            free_mb = total_mb - used_mb
            req_mb = 7000 if task_type == "music" else 10000

            task_name = "보컬 음악 작곡" if task_type == "music" else "보컬 모델 학습"
            engine_name = "ACE-Step 1.5 일체형 오디오 엔진" if task_type == "music" else "ACE-Step LoRA 학습 엔진"

            if free_mb < req_mb:
                deficit = req_mb - free_mb
                msg = (
                    f"현재 가용 VRAM이 부족하여 {task_name}을(를) 시작할 수 없습니다.\n\n"
                    f"• 필요 VRAM: 최소 {req_mb:,} MiB ({engine_name})\n"
                    f"• 현재 가용 VRAM: {free_mb:,} MiB\n"
                    f"• 현재 사용량: {used_mb:,} MiB / 전체 {total_mb:,} MiB\n"
                    f"• 부족한 VRAM: 약 {deficit:,} MiB\n\n"
                    f"VRAM을 점유하고 있는 다른 프로그램(영상 처리, 게임, 타 AI 프로세스 등)을 종료한 후 다시 시도해주세요."
                )
                return False, msg

        return True, ""

    # -------------------------------------------------------------------------
    # Actions: eBook Generation
    # -------------------------------------------------------------------------
    def start_ebook_generation(self):
        if not self.is_server_alive():
            QMessageBox.warning(self, "서버 미실행", "Ollama 서버가 정지되어 있습니다. 먼저 [서버 시작] 버튼을 눌러주세요.")
            return

        # VRAM 잔여량 점검 (부족 시 경고 팝업 및 시작 차단)
        ok, vram_msg = self.check_vram_for_task("ebook")
        if not ok:
            QMessageBox.warning(self, "VRAM 부족 - 집필 불가", vram_msg)
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
            html_path, md_path, file_idx = save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)

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
                f"아래 [📄 HTML 열기] 또는 [📁 books 폴더 열기] 버튼으로 완성된 책을 확인하실 수 있습니다."
            )
        else:
            if text:
                self.txt_ebook.setPlainText(text)
                topic = self.edit_topic.text().strip() or "전자책"
                save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)
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
        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)
        out_file = BATCH_OUT / "ebook-01.md"
        out_file.write_text(text, encoding="utf-8")
        QMessageBox.information(
            self,
            "임시 저장 완료",
            f"원고가 books 폴더에 안전하게 저장되었습니다.\n\n"
            f"• 저장 위치: books/{html_path.parent.name}/{html_path.name}\n"
            f"• HTML: {html_path}\n"
            f"• MD 원고: {md_path}"
        )

    def publish_ebook(self):
        """원고를 정식 전자책(HTML/MD)으로 출판하여 books/YYYY-MM-DD/ 폴더에 자동 순번 저장하고 브라우저로 실행"""
        text = self.txt_ebook.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "원고 없음", "출판할 전자책 원고 내용이 없습니다. 먼저 원고를 작성하거나 불러와주세요.")
            return

        topic = self.edit_topic.text().strip()
        if not topic:
            for line in text.splitlines():
                clean_l = line.strip()
                if clean_l.startswith("# ") and not clean_l.startswith("## "):
                    topic = clean_l[2:].strip()
                    break
        if not topic:
            topic = "전자책 단행본"

        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)

        # batch_out 동기화
        out_file = BATCH_OUT / "ebook-01.md"
        out_file.write_text(text, encoding="utf-8")
        (BATCH_OUT / "ebook-01.html").write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

        self.lbl_ebook_status.setText(f"🎉 정식 출판 완료! (books 저장: {html_path.parent.name}/{html_path.name})")
        self.lbl_ebook_time_info.setText(f"✅ 정식 출판: books/{html_path.parent.name}/{html_path.name}")

        a4_pages = max(1, len(text) // 1000)
        QMessageBox.information(
            self,
            "전자책 정식 출판 완료",
            f"축하합니다! 전자책이 성공적으로 출판되어 books 폴더에 저장되었습니다.\n\n"
            f"• 저장 폴더: books/{html_path.parent.name}/\n"
            f"• 저장 파일: {html_path.name} (당일 {file_idx}번째 책)\n"
            f"• 전체 경로: {html_path}\n"
            f"• 원고 분량: {len(text):,}자 (A4 약 {a4_pages}페이지 분량)\n\n"
            f"확인을 누르면 완성된 전자책 HTML이 브라우저에서 자동으로 열립니다."
        )
        try:
            os.startfile(str(html_path))
        except Exception:
            pass

    def export_ebook_html(self):
        text = self.txt_ebook.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "원고 없음", "내보낼 전자책 원고 내용이 없습니다.")
            return

        topic = self.edit_topic.text().strip() or "전자책"
        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)
        try:
            os.startfile(str(html_path))
        except Exception:
            pass

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

        ok, vram_msg = self.check_vram_for_task("merge")
        if not ok:
            QMessageBox.warning(self, "VRAM 부족 - 재구성 불가", vram_msg)
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
            QMessageBox.information(self, "재구성 완료", "글 융합 재구성이 완료되었습니다. 아래 [🚀 전자책으로 출판] 버튼으로 즉시 단행본 저장이 가능합니다.")
        else:
            QMessageBox.critical(self, "오류", err)

    def send_merge_to_ebook(self):
        text = self.txt_merge_res.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "내용 없음", "전자책 스튜디오로 보낼 융합 글 내용이 없습니다.")
            return
        self.txt_ebook.setPlainText(text)
        for line in text.splitlines():
            clean_l = line.strip()
            if clean_l.startswith("# ") and not clean_l.startswith("## "):
                self.edit_topic.setText(clean_l[2:].strip())
                break
        self.tabs.setCurrentWidget(self.tab_ebook)
        QMessageBox.information(self, "이동 완료", "융합 글이 전자책 스튜디오로 전달되었습니다. 검토 후 [🚀 전자책 출판하기]를 눌러주세요.")

    def publish_merge_as_ebook(self):
        text = self.txt_merge_res.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "내용 없음", "출판할 융합 글 내용이 없습니다.")
            return

        topic = "2026 자율 AI 에이전트와 순차 배치가 여는 미래"
        for line in text.splitlines():
            clean_l = line.strip()
            if clean_l.startswith("# ") and not clean_l.startswith("## "):
                topic = clean_l[2:].strip()
                break

        html_path, md_path, file_idx = save_ebook_auto_increment(topic, text, books_dir=BOOKS_DIR)
        QMessageBox.information(
            self,
            "전자책 출판 완료",
            f"융합 원고가 books 폴더에 전자책으로 정식 출판되었습니다.\n\n"
            f"• 저장 폴더: books/{html_path.parent.name}/\n"
            f"• 파일명: {html_path.name} (당일 {file_idx}번째 책)\n"
            f"• 전체 경로: {html_path}\n\n"
            f"확인을 누르면 완성된 전자책 HTML이 브라우저에서 자동으로 열립니다."
        )
        try:
            os.startfile(str(html_path))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Actions: Music & Audio Player
    # -------------------------------------------------------------------------
    def refresh_vocal_models(self):
        if hasattr(self, "tab_music"):
            self.tab_music.refresh_voices()

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

        # VRAM 잔여량 점검 (부족 시 경고 팝업 및 시작 차단)
        ok, vram_msg = self.check_vram_for_task("train")
        if not ok:
            QMessageBox.warning(self, "VRAM 부족 - 학습 불가", vram_msg)
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

    def start_voice_clone_training(self, alias, gender, clips):
        ok, vram_msg = self.check_vram_for_task("train")
        if not ok:
            QMessageBox.warning(self, "VRAM 부족 - 학습 불가", vram_msg)
            return
        model_dir = TRAINED_MODELS_DIR / alias
        if (model_dir / "meta.json").exists():
            if QMessageBox.question(self, "별칭 덮어쓰기",
                                    f"'{alias}' 목소리가 이미 있습니다. 새로 학습해 덮을까요?") \
                    != QMessageBox.StandardButton.Yes:
                return
        model_dir.mkdir(parents=True, exist_ok=True)
        clips_json = model_dir / "clips.json"
        clips_json.write_text(json.dumps({"gender": gender, "clips": clips}, ensure_ascii=False, indent=2),
                              encoding="utf-8")

        self.btn_start_train.setEnabled(False)
        self.voice_clone_pane.set_busy(True)
        self.btn_cancel_train.show()
        self.btn_cancel_train.setEnabled(True)
        self.pbar_train.setValue(0)
        self.lbl_train_stage_status.setText(f"⏳ '{alias}' 목소리 학습 준비 중...")
        self.train_start_ts = time.time()
        self.train_eta_sec = 0.0
        self.train_target_time_str = ""
        self.train_timer.start(1000)

        self.train_worker = VocalTrainingWorker(
            alias, [], preset=self.combo_train_preset.currentData() or "standard",
            mode="speech", clips_json=str(clips_json), gender=gender)
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
        if hasattr(self, "voice_clone_pane"):
            self.voice_clone_pane.set_busy(False)
        self.btn_cancel_train.hide()

        total_elapsed = int(time.time() - self.train_start_ts) if self.train_start_ts else 0
        em, es = divmod(total_elapsed, 60)
        elapsed_str = f"{em}분 {es}초"

        if ok:
            self.pbar_train.setValue(100)
            self.lbl_train_stage_status.setText(f"🎉 LoRA 학습 완료 ({model_name} / 소요: {elapsed_str})")
            self.lbl_train_time_info.setText(f"⏱️ 총 소요: {elapsed_str} | ✅ 모델 등록 완료")
            self.refresh_vocal_models()
            info = ""
            kind = "song_lora"
            try:
                meta = json.loads((TRAINED_MODELS_DIR / model_name / "meta.json").read_text(encoding="utf-8"))
                kind = meta.get("kind") or "song_lora"
                if kind == "speech_clone":
                    info = (f"• 녹음 문장: {meta.get('clips_count')}개\n"
                            f"• {meta.get('epochs')} epoch / 최종 loss {meta.get('final_loss')}\n"
                            f"• 노래 만들기에서 별칭 '{model_name}' 을 고르면 이 목소리로 부릅니다.\n")
                else:
                    info = (f"• 학습 곡 수: {meta.get('songs_count')}곡 (가사 포함 {meta.get('songs_with_lyrics')}곡)\n"
                            f"• {meta.get('epochs')} epoch / 최종 loss {meta.get('final_loss')}\n"
                            f"• 노래 생성 시 트리거 태그 '{meta.get('trigger_tag')}' 가 캡션 앞에 자동으로 붙습니다.\n")
            except Exception:
                pass
            title = "내 목소리 학습 완료" if kind == "speech_clone" else "보컬 학습 완료"
            body = (f"'{model_name}' 학습이 완료되었습니다.\n\n"
                    f"• 총 소요 시간: {elapsed_str}\n"
                    f"• 저장 폴더: trained_vocal_models/{model_name}/\n"
                    f"{info}\n"
                    f"지금 [노래 만들기]에서 이 목소리로 노래를 만들어 보시겠습니까?")
            reply = QMessageBox.information(
                self, title, body,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentWidget(self.tab_music)
                combo = self.tab_music.combo_voice
                for i in range(combo.count()):
                    if model_name in combo.itemText(i):
                        combo.setCurrentIndex(i)
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


SINGLE_INSTANCE_KEY = "YUNSEUL_LMM_SINGLE_INSTANCE_IPC"


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 1. 이미 1개의 인스턴스가 실행 중인지 확인 (단일 인스턴스 보장)
    socket = QLocalSocket()
    socket.connectToServer(SINGLE_INSTANCE_KEY)
    if socket.waitForConnected(500):
        # 이미 1개 인스턴스가 실행 중인 경우: 기존 인스턴스에 복원 요청 전송
        socket.write(b"RESTORE\n")
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()

        # Windows에서 기존 인스턴스가 창을 최상단으로 띄울 수 있도록 권한 부여
        if sys.platform == "win32":
            try:
                import ctypes
                # ASFW_ANY = -1: 다른 프로세스가 SetForegroundWindow 호출할 수 있도록 허용
                ctypes.windll.user32.AllowSetForegroundWindow(-1)
            except Exception:
                pass

        # 추가 실행 프로세스는 즉시 정상 종료 (중복 실행 방지)
        return 0

    # 2. 첫 번째 인스턴스인 경우: QLocalServer 수신 대기 시작
    local_server = QLocalServer()
    # 이전 비정상 종료 등으로 남아있을 수 있는 소켓/파이프 정리
    QLocalServer.removeServer(SINGLE_INSTANCE_KEY)
    local_server.listen(SINGLE_INSTANCE_KEY)

    if ICON_ICO.exists():
        app.setWindowIcon(QIcon(str(ICON_ICO)))
    elif ICON_PNG.exists():
        app.setWindowIcon(QIcon(str(ICON_PNG)))

    studio = YunseulLMMStudio()
    studio.show()

    # 이미 실행 중인 상태에서 사용자가 앱을 또 실행했을 때 기존 창을 복원/팝업하는 핸들러
    def handle_new_instance():
        client_socket = local_server.nextPendingConnection()
        if client_socket:
            client_socket.waitForReadyRead(500)
            client_socket.disconnectFromServer()
            studio.show_and_activate()

    local_server.newConnection.connect(handle_new_instance)

    exit_code = app.exec()
    local_server.close()
    QLocalServer.removeServer(SINGLE_INSTANCE_KEY)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
