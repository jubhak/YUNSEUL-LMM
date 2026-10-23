# -*- coding: utf-8 -*-
"""
voice_record_studio.py - '🎙️ 내 목소리 녹음' 탭

문장 10~30개를 하나씩 따라 읽어 녹음하면, 그 파일들로 내 목소리를 학습한다.
 - 왼쪽: 대본 목록(문장/음역/감정/팁) + 진행률
 - 가운데: 녹음(마이크) · 다시 듣기 · 파일에서 가져오기
 - 오른쪽: 학습 설정(이름/모드) + 로그 + 결과(음높이·포먼트·캡션 힌트)

실제 학습은 voice_enroll.py 를 subprocess 로 실행한다(위젯 멈춤 없음).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea, QTextEdit,
    QVBoxLayout, QWidget,
)

import voice_scripts

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

STYLE = """
QWidget#recRoot { background-color: #0b1120; }
QFrame[class="card"] { background-color: #111a2e; border: 1px solid #1e293b; border-radius: 12px; }
QLabel { color: #cbd5e1; font-size: 12px; }
QLabel[class="h1"] { color: #f8fafc; font-size: 15px; font-weight: bold; }
QLabel[class="h2"] { color: #94a3b8; font-size: 11px; font-weight: bold; }
QLabel[class="script"] { color: #f8fafc; font-size: 18px; font-weight: bold; }
QLabel[class="hint"] { color: #64748b; font-size: 11px; }
QListWidget { background-color: #0f172a; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 8px; font-size: 12px; }
QListWidget::item { padding: 7px 6px; border-bottom: 1px solid #1e293b; }
QListWidget::item:selected { background-color: #155e75; color: white; }
QPushButton[class="ghost"] { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; }
QPushButton[class="ghost"]:hover { background-color: #334155; }
QPushButton#btnRec { background-color: #be123c; color: white; font-weight: bold; font-size: 14px; border-radius: 8px; padding: 10px 18px; }
QPushButton#btnRec:checked { background-color: #f43f5e; }
QPushButton#btnTrain { background-color: #7c3aed; color: white; font-weight: bold; font-size: 14px; border-radius: 8px; padding: 10px 18px; }
QPushButton#btnTrain:disabled { background-color: #334155; color: #94a3b8; }
QLineEdit, QComboBox { background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 8px; font-size: 12px; }
QTextEdit { background-color: #0f172a; color: #cbd5e1; border: 1px solid #1e293b; border-radius: 8px; font-size: 11px; }
QProgressBar { background-color: #1e293b; border: none; border-radius: 4px; height: 8px; color: transparent; }
QProgressBar::chunk { background-color: #ec4899; border-radius: 4px; }
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


class MicRecorder(QWidget):
    """QAudioSource 로 마이크 녹음 (메인 스레드 + QTimer 폴링, 16bit PCM)."""

    level = pyqtSignal(float)
    finished = pyqtSignal(str)          # 저장된 wav 경로
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.io = None
        self.buf = bytearray()
        self.fmt = None
        self.sr = 48000
        self.ch = 1
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._poll)
        self.target = None

    def available(self) -> bool:
        try:
            return QMediaDevices.defaultAudioInput() is not None
        except Exception:
            return False

    def start(self, target_path) -> bool:
        try:
            dev = QMediaDevices.defaultAudioInput()
            if dev is None:
                self.failed.emit("마이크를 찾지 못했습니다. 파일로 가져오기를 사용해 주세요.")
                return False
            fmt = QAudioFormat()
            fmt.setSampleRate(48000)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if not dev.isFormatSupported(fmt):
                fmt = dev.preferredFormat()
            self.fmt = fmt
            self.sr = fmt.sampleRate() or 48000
            self.ch = fmt.channelCount() or 1
            self.source = QAudioSource(dev, fmt)
            self.io = self.source.start()
            if self.io is None:
                self.failed.emit("마이크를 열 수 없습니다(다른 앱이 사용 중일 수 있음).")
                return False
            self.buf = bytearray()
            self.target = Path(target_path)
            self.timer.start()
            return True
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"녹음 시작 실패: {e}")
            return False

    def _poll(self):
        if self.io is None:
            return
        data = self.io.readAll()
        if data is None:
            return
        b = bytes(data)
        if b:
            self.buf.extend(b)
            try:
                import array
                arr = array.array("h")
                arr.frombytes(b[: len(b) - (len(b) % 2)])
                if len(arr):
                    peak = max(abs(min(arr)), abs(max(arr))) / 32768.0
                    self.level.emit(min(1.0, peak))
            except Exception:
                pass

    def stop(self) -> str | None:
        self.timer.stop()
        try:
            if self.source:
                self.source.stop()
        except Exception:
            pass
        self.source = None
        self.io = None
        if not self.buf or self.target is None:
            return None
        path = self.target
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(self.ch)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(bytes(self.buf))
        self.finished.emit(str(path))
        return str(path)


class EnrollWorker(QThread):
    """voice_enroll.py 실행 중계."""
    log_line = pyqtSignal(str)
    prog = pyqtSignal(int, str)
    done = pyqtSignal(bool, str, str)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.proc = None

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=CREATE_NO_WINDOW)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def run(self):
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                         errors="replace", env=env, cwd=self.cwd,
                                         creationflags=CREATE_NO_WINDOW)
            ok, err = False, ""
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("[PROGRESS:"):
                    parts = line[10:-1].split(":", 2)
                    if len(parts) == 3:
                        try:
                            self.prog.emit(int(float(parts[1])), parts[2])
                        except Exception:
                            pass
                    continue
                if line.startswith("[LOG] "):
                    self.log_line.emit(line[6:])
                    continue
                if line.startswith("[DONE:"):
                    ok = True
                    continue
                if line.startswith("[ERROR]"):
                    err = line[7:].strip()
                self.log_line.emit("  " + line)
            self.proc.wait()
            if ok and self.proc.returncode == 0:
                self.done.emit(True, "", "")
            else:
                self.done.emit(False, "", err or f"학습 프로세스 종료 코드 {self.proc.returncode}")
        except Exception as e:  # noqa: BLE001
            self.done.emit(False, "", f"{type(e).__name__}: {e}")


class VoiceRecordStudio(QWidget):
    def __init__(self, host, base_dir: Path, trained_dir: Path, acestep_py: Path):
        super().__init__()
        self.host = host
        self.base_dir = Path(base_dir)
        self.trained_dir = Path(trained_dir)
        self.acestep_py = Path(acestep_py)
        self.limit = 30
        self.recording_id = None
        self.worker = None
        self.takes = {}             # sentence id -> wav path
        self.setObjectName("recRoot")
        self.setStyleSheet(STYLE)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)
        root.addWidget(self._build_script_panel())
        root.addWidget(self._build_record_panel(), 1)
        root.addWidget(self._build_train_panel())
        self._reload_script()
        self._refresh_takes()

    # ------------------------------------------------------------------ 대본
    def _build_script_panel(self):
        card = _card()
        card.setFixedWidth(330)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        lay.addWidget(_label("따라 읽을 문장", "h1"))
        row = QHBoxLayout()
        row.addWidget(_label("문장 수", "h2"))
        self.combo_limit = QComboBox()
        for n, label in [(10, "10문장 (약 2분 · 최소)"), (20, "20문장 (약 5분)"),
                         (30, "30문장 (약 8분 · 추천)")]:
            self.combo_limit.addItem(label, n)
        self.combo_limit.setCurrentIndex(2)
        self.combo_limit.currentIndexChanged.connect(self._on_limit_changed)
        row.addWidget(self.combo_limit, 1)
        lay.addLayout(row)

        self.list_script = QListWidget()
        self.list_script.currentItemChanged.connect(self._on_script_selected)
        lay.addWidget(self.list_script, 1)

        self.lbl_progress = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_progress)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setTextVisible(False)
        lay.addWidget(self.pbar)

        btn = QPushButton("📄 대본 txt 로 저장")
        btn.setProperty("class", "ghost")
        btn.clicked.connect(self._save_script_txt)
        lay.addWidget(btn)
        lay.addWidget(_label("문장 하나를 고르고 → 가운데 [● 녹음] → [다음 문장] 순서로 진행하세요.", "hint", wrap=True))
        return card

    def _on_limit_changed(self):
        self.limit = self.combo_limit.currentData() or 30
        self._reload_script()

    def _reload_script(self):
        self.list_script.blockSignals(True)
        self.list_script.clear()
        for s in voice_scripts.script_sentences(self.limit):
            mark = "✅" if s["id"] in self.takes else "▫️"
            item = QListWidgetItem(f"{mark} {s['id']:02d}. {s['text'][:26]}…" if len(s["text"]) > 26
                                   else f"{mark} {s['id']:02d}. {s['text']}")
            item.setData(Qt.ItemDataRole.UserRole, s["id"])
            self.list_script.addItem(item)
        self.list_script.blockSignals(False)
        if self.list_script.count():
            self.list_script.setCurrentRow(0)
            self._on_script_selected(self.list_script.item(0), None)
        self._update_progress()

    def _current_sentence(self):
        item = self.list_script.currentItem()
        if not item:
            return None
        sid = item.data(Qt.ItemDataRole.UserRole)
        for s in voice_scripts.SENTENCES:
            if s["id"] == sid:
                return s
        return None

    def _on_script_selected(self, item, _prev):
        s = self._current_sentence()
        if not s:
            return
        self.lbl_sentence.setText(s["text"])
        self.lbl_meta.setText(f"{s['pitch']} · {s['tone']} · {s['focus']}")
        self.lbl_tip.setText("💡 " + s["tip"])
        self.lbl_pitch_guide.setText("음역 안내 — " + voice_scripts.PITCH_GUIDE.get(s["pitch"], ""))
        take = self.takes.get(s["id"])
        self.btn_play_take.setEnabled(bool(take))
        self.lbl_take.setText(f"녹음됨: {Path(take).name}" if take else "아직 녹음하지 않았습니다.")

    def _update_progress(self):
        need = 10
        done = len(self.takes)
        self.pbar.setValue(int(min(100, done / max(1, self.limit) * 100)))
        msg = f"녹음 {done} / {self.limit} 문장"
        if done < need:
            msg += f"  · 최소 {need}문장까지 {need - done}개 남음"
        elif done < 20:
            msg += "  · 학습 가능! 20~30문장이면 더 자연스러워집니다"
        else:
            msg += "  · 학습 준비 완료"
        self.lbl_progress.setText(msg)
        self.btn_train.setEnabled(done >= need and self.worker is None)

    def _save_script_txt(self):
        p = self.base_dir / f"voice_script_{self.limit}.txt"
        p.write_text(voice_scripts.read_text(self.limit), encoding="utf-8")
        QMessageBox.information(self, "대본 저장", f"저장했습니다:\n{p}")

    # ------------------------------------------------------------------ 녹음
    def _build_record_panel(self):
        card = _card()
        card.setFixedWidth(430)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        lay.addWidget(_label("① 문장을 소리 내어 읽기", "h1"))
        self.lbl_sentence = _label("문장을 선택하세요", "script", wrap=True)
        self.lbl_sentence.setMinimumHeight(64)
        lay.addWidget(self.lbl_sentence)
        self.lbl_meta = _label("", "hint")
        lay.addWidget(self.lbl_meta)
        self.lbl_tip = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_tip)
        self.lbl_pitch_guide = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_pitch_guide)

        lay.addWidget(_label("② 녹음", "h1"))
        row = QHBoxLayout()
        self.btn_rec = QPushButton("● 녹음")
        self.btn_rec.setObjectName("btnRec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._toggle_record)
        row.addWidget(self.btn_rec, 1)
        self.btn_play_take = QPushButton("▶ 내 녹음 듣기")
        self.btn_play_take.setProperty("class", "ghost")
        self.btn_play_take.clicked.connect(self._play_take)
        row.addWidget(self.btn_play_take)
        lay.addLayout(row)

        self.rec_level = QProgressBar()
        self.rec_level.setRange(0, 100)
        self.rec_level.setTextVisible(False)
        self.rec_level.setFixedHeight(6)
        lay.addWidget(self.rec_level)
        self.lbl_take = _label("아직 녹음하지 않았습니다.", "hint", wrap=True)
        lay.addWidget(self.lbl_take)

        nav = QHBoxLayout()
        b_prev = QPushButton("← 이전 문장")
        b_prev.setProperty("class", "ghost")
        b_prev.clicked.connect(lambda: self._step(-1))
        nav.addWidget(b_prev)
        b_next = QPushButton("다음 문장 →")
        b_next.setProperty("class", "ghost")
        b_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(b_next)
        lay.addLayout(nav)

        lay.addWidget(_label("③ 이미 녹음한 파일이 있다면", "h1"))
        imp = QHBoxLayout()
        b_file = QPushButton("📂 파일에서 가져오기")
        b_file.setProperty("class", "ghost")
        b_file.clicked.connect(self._import_files)
        imp.addWidget(b_file)
        b_del = QPushButton("🗑 이 문장 녹음 삭제")
        b_del.setProperty("class", "ghost")
        b_del.clicked.connect(self._delete_take)
        imp.addWidget(b_del)
        lay.addLayout(imp)
        lay.addWidget(_label("가져온 파일은 문장 순서대로 자동 배정됩니다. "
                             "이미 있는 녹음도 그대로 쌓이니 나눠서 해도 됩니다.", "hint", wrap=True))
        lay.addStretch()
        return card

    def _name(self):
        name = self.edit_name.text().strip() if hasattr(self, "edit_name") else ""
        return name or "내목소리"

    def _take_path(self, sid):
        d = self.trained_dir / self._name() / "speech"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"s{sid:02d}.wav"

    def _toggle_record(self, checked):
        if checked:
            s = self._current_sentence()
            if not s:
                self.btn_rec.setChecked(False)
                return
            path = self._take_path(s["id"])
            self.recorder = getattr(self, "recorder", None) or MicRecorder(self)
            if not hasattr(self, "recorder") or self.recorder is None:
                self.recorder = MicRecorder(self)
                self.recorder.level.connect(lambda v: self.rec_level.setValue(int(v * 100)))
                self.recorder.failed.connect(lambda m: QMessageBox.warning(self, "녹음 오류", m))
                self.recorder.finished.connect(self._on_recorded)
            elif not self.recorder.timer.isActive():
                self.recorder.level.connect(lambda v: self.rec_level.setValue(int(v * 100)))
                self.recorder.failed.connect(lambda m: QMessageBox.warning(self, "녹음 오류", m))
                self.recorder.finished.connect(self._on_recorded)
            if self.recorder.start(path):
                self.recording_id = s["id"]
                self.btn_rec.setText("■ 중지")
                self.lbl_take.setText("● 녹음 중... 문장을 읽고 끝나면 [■ 중지]를 누르세요.")
            else:
                self.btn_rec.setChecked(False)
        else:
            self.btn_rec.setText("● 녹음")
            if getattr(self, "recorder", None):
                self.recorder.stop()
            self.rec_level.setValue(0)

    def _on_recorded(self, path):
        if self.recording_id is not None:
            self.takes[self.recording_id] = path
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("● 녹음")
        self.lbl_take.setText(f"✅ 저장됨: {Path(path).name}")
        self._refresh_takes()

    def _play_take(self):
        s = self._current_sentence()
        if not s:
            return
        take = self.takes.get(s["id"])
        if not take or not Path(take).exists():
            return
        try:
            self.host.player.stop()
            self.host.player.setSource(QUrl.fromLocalFile(str(take)))
            self.host.player.play()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "재생 실패", str(e))

    def _step(self, delta):
        row = self.list_script.currentRow() + delta
        if 0 <= row < self.list_script.count():
            self.list_script.setCurrentRow(row)

    def _delete_take(self):
        s = self._current_sentence()
        if not s:
            return
        take = self.takes.pop(s["id"], None)
        if take:
            try:
                Path(take).unlink(missing_ok=True)
            except Exception:
                pass
        self.lbl_take.setText("이 문장 녹음을 삭제했습니다.")
        self._refresh_takes()

    def _import_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "녹음 파일 가져오기", "",
                                                "오디오 (*.wav *.mp3 *.m4a *.ogg *.flac)")
        if not files:
            return
        assigned = 0
        pending = [s for s in voice_scripts.script_sentences(self.limit) if s["id"] not in self.takes]
        for f, s in zip(sorted(files), pending):
            dst = self._take_path(s["id"])
            try:
                if Path(f).suffix.lower() == ".wav":
                    import shutil
                    shutil.copy2(f, dst)
                else:
                    import subprocess as sp
                    ff = shutil.which("ffmpeg")
                    if not ff:
                        raise RuntimeError("ffmpeg 없음")
                    sp.run([ff, "-y", "-v", "error", "-i", f, "-ar", "48000", "-ac", "1", str(dst)],
                           check=True)
                self.takes[s["id"]] = str(dst)
                assigned += 1
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "가져오기 실패", f"{Path(f).name}: {e}")
        self._refresh_takes()
        QMessageBox.information(self, "가져오기 완료", f"{assigned}개 파일을 문장에 배정했습니다.")

    def _refresh_takes(self):
        # 폴더에 있는 기존 녹음도 불러온다
        d = self.trained_dir / self._name() / "speech"
        if d.exists():
            for p in sorted(d.glob("s*.wav")):
                try:
                    sid = int(p.stem[1:])
                except Exception:
                    continue
                self.takes.setdefault(sid, str(p))
        self._reload_script()
        if self.list_script.count():
            self._on_script_selected(self.list_script.currentItem(), None)

    # ------------------------------------------------------------------ 학습
    def _build_train_panel(self):
        card = _card()
        card.setFixedWidth(360)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        lay.addWidget(_label("내 목소리 학습", "h1"))

        lay.addWidget(_label("이름", "h2"))
        self.edit_name = QLineEdit("내목소리")
        self.edit_name.setPlaceholderText("예: 내목소리")
        self.edit_name.editingFinished.connect(self._refresh_takes)
        lay.addWidget(self.edit_name)

        lay.addWidget(_label("학습 강도", "h2"))
        self.combo_preset = QComboBox()
        for val, label in [("fast", "빠르게 (품질 보통)"), ("standard", "표준 (추천)"),
                           ("deep", "정밀 (오래 걸림)")]:
            self.combo_preset.addItem(label, val)
        self.combo_preset.setCurrentIndex(1)
        lay.addWidget(self.combo_preset)

        self.btn_train = QPushButton("🚀 내 목소리로 학습하기")
        self.btn_train.setObjectName("btnTrain")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self._start_train)
        lay.addWidget(self.btn_train)
        self.btn_cancel = QPushButton("중지")
        self.btn_cancel.setProperty("class", "ghost")
        self.btn_cancel.hide()
        self.btn_cancel.clicked.connect(self._cancel_train)
        lay.addWidget(self.btn_cancel)

        self.pbar_train = QProgressBar()
        self.pbar_train.setRange(0, 100)
        self.pbar_train.setTextVisible(False)
        lay.addWidget(self.pbar_train)
        self.lbl_stage = _label("10문장부터 학습할 수 있습니다.", "hint", wrap=True)
        lay.addWidget(self.lbl_stage)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        lay.addWidget(self.txt_log, 1)

        self.lbl_result = _label("", "hint", wrap=True)
        lay.addWidget(self.lbl_result)
        return card

    def _start_train(self):
        clips = [self.takes[k] for k in sorted(self.takes) if Path(self.takes[k]).exists()]
        if len(clips) < 10:
            QMessageBox.warning(self, "녹음 부족", "최소 10문장을 녹음해야 학습할 수 있습니다.")
            return
        name = self._name()
        if QMessageBox.question(self, "내 목소리 학습",
                                f"'{name}' 이름으로 {len(clips)}문장을 학습합니다.\n"
                                f"(GPU 사용 · 수십 분 걸릴 수 있습니다)\n계속할까요?") \
                != QMessageBox.StandardButton.Yes:
            return
        model_dir = self.trained_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        script_map = {str(self.takes[s["id"]]): s["text"]
                      for s in voice_scripts.script_sentences(self.limit) if s["id"] in self.takes}
        sm_path = model_dir / "script_map.json"
        sm_path.write_text(json.dumps([{"file": k, "text": v} for k, v in script_map.items()],
                                      ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = [sys.executable, "-u", str(self.base_dir / "voice_enroll.py"), "enroll",
               "--name", name, "--out-root", str(self.trained_dir),
               "--script-map", str(sm_path),
               "--preset", self.combo_preset.currentData() or "standard",
               "--clips", *clips]
        if self.acestep_py.exists():
            cmd += ["--acestep-python", str(sys.executable)]
        self.txt_log.clear()
        self.btn_train.setEnabled(False)
        self.btn_train.setText("⏳ 학습 중...")
        self.btn_cancel.show()
        self.pbar_train.setValue(0)
        self.worker = EnrollWorker(cmd, str(self.base_dir))
        self.worker.log_line.connect(lambda m: (self.txt_log.append(m),
                                                self.txt_log.verticalScrollBar().setValue(
                                                    self.txt_log.verticalScrollBar().maximum())))
        self.worker.prog.connect(self._on_prog)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_prog(self, pct, title):
        self.pbar_train.setValue(int(pct))
        self.lbl_stage.setText(title)

    def _cancel_train(self):
        if self.worker:
            self.lbl_stage.setText("중지하는 중...")
            self.worker.cancel()

    def _on_done(self, ok, _model, err):
        self.worker = None
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🚀 내 목소리로 학습하기")
        self.btn_cancel.hide()
        if not ok:
            self.lbl_stage.setText("⚠️ 학습 실패")
            QMessageBox.critical(self, "학습 실패", err or "알 수 없는 오류")
            return
        self.pbar_train.setValue(100)
        self.lbl_stage.setText("🎉 학습 완료 — 노래 만들기에서 내 목소리를 선택하세요")
        prof_path = self.trained_dir / self._name() / "voice_profile.json"
        if prof_path.exists():
            try:
                p = json.loads(prof_path.read_text(encoding="utf-8"))
                self.lbl_result.setText(
                    f"음높이 {p.get('f0_median_hz')}Hz ({p.get('gender_guess')}) · "
                    f"HNR {p.get('hnr_db')}dB · 포먼트 {p.get('formants_hz')}\n"
                    f"캡션 힌트: {p.get('caption_hint')}\n"
                    + " ".join(p.get("quality_notes", [])[:2]))
            except Exception:
                pass
        try:
            self.host.tab_music.refresh_voices()
            self.host.refresh_vocal_models()
        except Exception:
            pass
        QMessageBox.information(self, "학습 완료",
                                f"'{self._name()}' 목소리 학습이 끝났습니다.\n"
                                f"[🎵 노래 만들기] 탭의 목소리 목록에서 선택할 수 있습니다.")
