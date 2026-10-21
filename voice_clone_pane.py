# -*- coding: utf-8 -*-
"""따라 읽기로 내 목소리를 별칭 저장하는 학습 화면."""
import wave
from pathlib import Path

from PyQt6.QtCore import Qt, QByteArray, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices, QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from voice_script import VOICE_SENTENCES

MIN_CLIPS = 10


class MicRecorder:
    def __init__(self):
        self.source = None
        self.device_io = None
        self.buf = QByteArray()
        self.fmt = None

    def start(self):
        fmt = QAudioFormat()
        fmt.setSampleRate(44100)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            raise RuntimeError("마이크를 찾지 못했습니다. 녹음 장치를 연결하거나 파일을 올려 주세요.")
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        self.fmt = fmt
        self.buf = QByteArray()
        self.source = QAudioSource(device, fmt)
        self.device_io = self.source.start()
        if self.device_io is None:
            raise RuntimeError("마이크를 열 수 없습니다.")
        self.device_io.readyRead.connect(self._on_ready)
        from PyQt6.QtCore import QTimer
        self._poll = QTimer()
        self._poll.timeout.connect(self._on_ready)
        self._poll.start(50)

    def _on_ready(self):
        if self.device_io:
            self.buf.append(self.device_io.readAll())

    def stop(self, path: Path):
        if getattr(self, "_poll", None):
            self._poll.stop()
        self._on_ready()
        if self.source:
            self.source.stop()
        path.parent.mkdir(parents=True, exist_ok=True)
        rate = self.fmt.sampleRate() if self.fmt else 44100
        ch = self.fmt.channelCount() if self.fmt else 1
        width = 2
        if self.fmt and self.fmt.sampleFormat() == QAudioFormat.SampleFormat.Int16:
            width = 2
        elif self.fmt and self.fmt.sampleFormat() == QAudioFormat.SampleFormat.UInt8:
            width = 1
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(bytes(self.buf))
        self.source = None
        self.device_io = None
        return path


class VoiceClonePane(QWidget):
    train_requested = pyqtSignal(str, str, list)

    def __init__(self, inbox_dir: Path, parent=None):
        super().__init__(parent)
        self.inbox = Path(inbox_dir)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.clips = {}  # idx -> path
        self.recorder = None
        self.recording = False
        self.preview = QMediaPlayer()
        self.preview_out = QAudioOutput()
        self.preview.setAudioOutput(self.preview_out)
        self._build()
        self._select_row(0)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 10, 0)
        lay.setSpacing(8)

        title = QLabel("🎙 문장을 따라 읽어 내 목소리 만들기")
        title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        lay.addWidget(title)
        hint = QLabel(
            "아래 문장을 10~27개 또렷이 읽으면 별칭으로 저장됩니다.\n"
            "노래 만들기에서 여성 / 남성 / 이 별칭 중 하나를 고를 수 있습니다."
        )
        hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("별칭:"))
        self.edit_alias = QLineEdit()
        self.edit_alias.setPlaceholderText("예: 준호, 지은")
        self.edit_alias.setStyleSheet(
            "background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px; font-weight: bold;")
        row.addWidget(self.edit_alias, 1)
        row.addWidget(QLabel("성별:"))
        self.combo_gender = QComboBox()
        self.combo_gender.addItem("여성", "female")
        self.combo_gender.addItem("남성", "male")
        self.combo_gender.setStyleSheet(
            "background-color: #1e293b; color: white; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        row.addWidget(self.combo_gender)
        lay.addLayout(row)

        self.lbl_sentence = QLabel("")
        self.lbl_sentence.setWordWrap(True)
        self.lbl_sentence.setStyleSheet(
            "color: #f8fafc; font-size: 16px; font-weight: bold; background-color: #0f172a; "
            "border: 1px solid #334155; border-radius: 8px; padding: 14px;")
        self.lbl_sentence.setMinimumHeight(72)
        lay.addWidget(self.lbl_sentence)

        rec_row = QHBoxLayout()
        self.btn_rec = QPushButton("● 녹음 시작")
        self.btn_rec.setFixedHeight(36)
        self.btn_rec.setStyleSheet(
            "background-color: #dc2626; color: white; font-weight: bold; border-radius: 8px;")
        self.btn_rec.clicked.connect(self.toggle_record)
        rec_row.addWidget(self.btn_rec)
        btn_play = QPushButton("▶ 들어보기")
        btn_play.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        btn_play.clicked.connect(self.play_clip)
        rec_row.addWidget(btn_play)
        btn_upload = QPushButton("파일 올리기")
        btn_upload.setStyleSheet(
            "background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;")
        btn_upload.clicked.connect(self.upload_clip)
        rec_row.addWidget(btn_upload)
        lay.addLayout(rec_row)
        self.lbl_rec = QLabel("문장을 고른 뒤 녹음하거나, 이미 읽은 파일을 올리세요.")
        self.lbl_rec.setStyleSheet("color: #64748b; font-size: 11px;")
        lay.addWidget(self.lbl_rec)

        self.list_sent = QListWidget()
        self.list_sent.setStyleSheet("""
            QListWidget { background-color: #0f172a; color: #f1f5f9; border: 1px solid #1e293b;
                border-radius: 8px; padding: 4px; font-size: 12px; }
            QListWidget::item { padding: 5px; border-bottom: 1px solid #1e293b; }
            QListWidget::item:selected { background-color: #0284c7; color: white; }
        """)
        for i, s in enumerate(VOICE_SENTENCES):
            item = QListWidgetItem(f"☐  {i + 1:02d}. {s}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_sent.addItem(item)
        self.list_sent.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list_sent, 1)

        self.lbl_count = QLabel(f"녹음: 0 / 최소 {MIN_CLIPS}문장 (많을수록 음색이 안정됩니다)")
        self.lbl_count.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")
        lay.addWidget(self.lbl_count)

        self.btn_train = QPushButton("🚀 이 목소리로 학습하고 별칭 저장")
        self.btn_train.setFixedHeight(40)
        self.btn_train.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #8b5cf6);
                color: white; font-weight: bold; font-size: 13px; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #38bdf8; }
        """)
        self.btn_train.clicked.connect(self._emit_train)
        lay.addWidget(self.btn_train)

        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._tick_rec)
        self._rec_sec = 0

    def _select_row(self, i):
        self.list_sent.setCurrentRow(i)
        self._on_row(i)

    def _on_row(self, i):
        if i < 0 or i >= len(VOICE_SENTENCES):
            return
        self.lbl_sentence.setText(f"{i + 1} / {len(VOICE_SENTENCES)}\n{VOICE_SENTENCES[i]}")
        if i in self.clips:
            self.lbl_rec.setText("이 문장은 녹음되어 있습니다. 다시 녹음하면 덮어씁니다.")
        else:
            self.lbl_rec.setText("또렷이, 평소 말하는 속도로 한 번만 읽어 주세요.")

    def _clip_path(self, idx):
        return self.inbox / f"{idx + 1:02d}.wav"

    def toggle_record(self):
        if self.recording:
            self._stop_record()
            return
        idx = self.list_sent.currentRow()
        if idx < 0:
            return
        try:
            self.recorder = MicRecorder()
            self.recorder.start()
        except Exception as e:
            QMessageBox.warning(self, "마이크", str(e))
            return
        self.recording = True
        self._rec_sec = 0
        self.rec_timer.start(200)
        self.btn_rec.setText("■ 녹음 중지")
        self.lbl_rec.setText("녹음 중... 문장을 읽고 끝나면 중지를 누르세요.")

    def _tick_rec(self):
        self._rec_sec += 0.2
        self.lbl_rec.setText(f"녹음 중... {self._rec_sec:.1f}초")
        if self._rec_sec > 20:
            self._stop_record()

    def _stop_record(self):
        self.rec_timer.stop()
        self.recording = False
        self.btn_rec.setText("● 녹음 시작")
        idx = self.list_sent.currentRow()
        if idx < 0 or not self.recorder:
            return
        path = self._clip_path(idx)
        try:
            self.recorder.stop(path)
        except Exception as e:
            QMessageBox.warning(self, "녹음 저장 실패", str(e))
            return
        if path.stat().st_size < 2000:
            QMessageBox.warning(self, "녹음이 너무 짧습니다", "문장을 끝까지 읽어 주세요.")
            path.unlink(missing_ok=True)
            return
        self.clips[idx] = path
        self._refresh_item(idx)
        if idx + 1 < self.list_sent.count():
            self._select_row(idx + 1)

    def play_clip(self):
        idx = self.list_sent.currentRow()
        path = self.clips.get(idx)
        if not path or not Path(path).exists():
            QMessageBox.information(self, "들어보기", "이 문장은 아직 녹음되지 않았습니다.")
            return
        self.preview.setSource(QUrl.fromLocalFile(str(path)))
        self.preview.play()

    def upload_clip(self):
        idx = self.list_sent.currentRow()
        if idx < 0:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "이 문장을 읽은 음성 파일", "", "오디오 (*.wav *.mp3 *.m4a *.flac);;All Files (*)")
        if not files:
            return
        import shutil
        dest = self._clip_path(idx)
        shutil.copy2(files[0], dest)
        self.clips[idx] = dest
        self._refresh_item(idx)
        extras = files[1:]
        n = idx + 1
        for extra in extras:
            while n < len(VOICE_SENTENCES) and n in self.clips:
                n += 1
            if n >= len(VOICE_SENTENCES):
                break
            d = self._clip_path(n)
            shutil.copy2(extra, d)
            self.clips[n] = d
            self._refresh_item(n)
            n += 1

    def _refresh_item(self, idx):
        item = self.list_sent.item(idx)
        if not item:
            return
        mark = "✅" if idx in self.clips else "☐"
        item.setText(f"{mark}  {idx + 1:02d}. {VOICE_SENTENCES[idx]}")
        self.lbl_count.setText(
            f"녹음: {len(self.clips)} / 최소 {MIN_CLIPS}문장 (많을수록 음색이 안정됩니다)")

    def _emit_train(self):
        alias = self.edit_alias.text().strip()
        if not alias:
            QMessageBox.warning(self, "별칭 필요", "목소리를 부를 별칭을 적어 주세요. 예: 준호")
            self.edit_alias.setFocus()
            return
        if len(self.clips) < MIN_CLIPS:
            QMessageBox.warning(
                self, "녹음이 부족합니다",
                f"최소 {MIN_CLIPS}문장을 녹음하거나 올려 주세요. 지금은 {len(self.clips)}개입니다.")
            return
        clips = [{"path": str(self.clips[i]), "text": VOICE_SENTENCES[i]}
                 for i in sorted(self.clips)]
        self.train_requested.emit(alias, self.combo_gender.currentData(), clips)

    def set_busy(self, busy: bool):
        self.btn_train.setEnabled(not busy)
        self.btn_rec.setEnabled(not busy)
        self.btn_train.setText("⏳ 목소리 학습 중..." if busy else "🚀 이 목소리로 학습하고 별칭 저장")
