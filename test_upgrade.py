"""Offline regression suite. No real API calls, model downloads or music generation.
Run: QT_QPA_PLATFORM=offscreen python -m unittest -v test_upgrade
"""
import io
import json
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
import numpy as np

import gemini_lyrics
import lyrics_writer
import workspace_state
import training_genres
import vocal_pipeline
import svc
import vocal_post
import vocal_coach
from audio_dsp_safe import spectral_gain, smooth_edge


class GeminiTests(unittest.TestCase):
    def response(self, text="[Verse 1]\n창가에 앉아", reason="STOP"):
        return io.BytesIO(json.dumps({"candidates": [{"finishReason": reason, "content": {
            "parts": [{"text": "private thought", "thought": True}, {"text": text}]}}]}).encode())

    def test_key_header_and_payload(self):
        with patch("urllib.request.urlopen", return_value=self.response()) as req:
            self.assertIn("창가", gemini_lyrics.generate("주제", "작사가", api_key="secret-key"))
        request = req.call_args.args[0]
        self.assertNotIn("secret-key", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "secret-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "작사가")

    def test_missing_key(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "API 키"):
            gemini_lyrics.generate("p", "s")

    def test_environment_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}), patch("urllib.request.urlopen", return_value=self.response()) as req:
            gemini_lyrics.generate("p", "s")
            self.assertEqual(req.call_args.args[0].get_header("X-goog-api-key"), "env-key")

    def test_error_redacts_key(self):
        err = urllib.error.HTTPError("secret-key-in-url", 403, "secret-key", {}, None)
        with patch("urllib.request.urlopen", side_effect=err), self.assertRaises(RuntimeError) as raised:
            gemini_lyrics.generate("p", "s", api_key="secret-key")
        self.assertNotIn("secret-key", str(raised.exception))

    def test_retry_transient(self):
        err = urllib.error.HTTPError("url", 429, "limited", {}, None)
        with patch("urllib.request.urlopen", side_effect=[err, self.response()]) as req, patch("time.sleep"):
            gemini_lyrics.generate("p", "s", api_key="key")
            self.assertEqual(req.call_count, 2)

    def test_empty_or_truncated_rejected(self):
        for text, reason in [("", "STOP"), ("partial", "MAX_TOKENS"), ("blocked", "SAFETY")]:
            with self.subTest(reason=reason), patch("urllib.request.urlopen", return_value=self.response(text, reason)), self.assertRaises(RuntimeError):
                gemini_lyrics.generate("p", "s", api_key="key")

    def test_model_path_validation(self):
        with self.assertRaises(ValueError):
            gemini_lyrics.generate("p", "s", api_key="key", model="bad/model?key=x")

    def test_gemini_provider_never_calls_local(self):
        raw = "[Verse 1]\n창가에 남은 온기\n조용히 놓인 찻잔\n그날의 작은 약속\n이제는 웃어 본다\n[Chorus]\n너를 보내는 저녁\n노을이 물든 거리\n내일을 향해 걷자\n다시 내 이름 불러"
        with patch.object(lyrics_writer, "gemini_generate", return_value=raw) as remote, patch.object(lyrics_writer, "ollama_generate") as local:
            result = lyrics_writer.generate_lyrics("이별 뒤의 일상", "trot", duration=60, candidates=1, log=lambda *_: None)
            self.assertIn("[Chorus]", result)
            self.assertGreaterEqual(remote.call_count, 2)
            local.assert_not_called()

    def test_api_failure_no_local_fallback(self):
        with patch.object(lyrics_writer, "gemini_generate", side_effect=RuntimeError("failed")), patch.object(lyrics_writer, "ollama_generate") as local, self.assertRaises(RuntimeError):
            lyrics_writer.generate_lyrics("주제", candidates=1)
        local.assert_not_called()


class DraftAndGenreTests(unittest.TestCase):
    def test_atomic_roundtrip_and_empty_tombstone(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "draft.json"
            state = {"topic": "새 주제", "body": "고친 본문", "pages": 120, "genre": "essay"}
            workspace_state.save_draft(p, state)
            self.assertEqual(workspace_state.load_draft(p)["body"], "고친 본문")
            workspace_state.save_draft(p, {"topic": "", "body": ""})
            self.assertEqual(workspace_state.load_draft(p)["body"], "")
            self.assertFalse(list(Path(d).glob("*.tmp")))

    def test_corrupt_draft_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "draft.json"
            for raw in ("{bad", "[]", "null", '{"version":99}'):
                p.write_text(raw)
                self.assertEqual(workspace_state.load_draft(p), {})

    def test_all_genres_and_unknown_caption(self):
        self.assertEqual(training_genres.detect_genre("내노래_트로트.wav"), "trot")
        self.assertEqual(training_genres.detect_genre("quiet_jazz.wav"), "jazz")
        self.assertEqual(training_genres.detect_genre("unlabelled-123.wav"), "unknown")
        self.assertNotIn("pop", training_genres.caption_for("unknown").lower())
        import vocal_trainer_runner
        for key in training_genres.GENRES:
            self.assertTrue(vocal_trainer_runner.build_caption(key, 100, False))
        self.assertIn("trot", vocal_trainer_runner.build_caption("trot", 100, False))

    def test_trot_available_in_song_menu(self):
        self.assertIn("trot", [key for _, keys in lyrics_writer.GENRE_MENU for key in keys])
        self.assertIn("trot", lyrics_writer.build_caption("trot", "female"))


class AudioSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.d = Path(self.temp.name)

    def tone(self, n=96000, freq=330.0):
        t = np.arange(n) / 48000
        y = (.15 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        return np.column_stack([y, y])

    def test_unity_eq_preserves_both_edges_and_short_clips(self):
        for n in (1, 127, 4096, 48000, 48137):
            y = np.random.default_rng(7).normal(0, .1, (n, 2)).astype(np.float32)
            actual = spectral_gain(y, np.ones(2049))
            np.testing.assert_allclose(actual, y, atol=1e-6)

    def test_eq_implementations_preserve_tail(self):
        y = self.tone(48137)
        np.testing.assert_allclose(svc._apply_eq_curve(y, 48000, [80, 24000], [0, 0]), y, atol=1e-6)
        np.testing.assert_allclose(vocal_coach._apply_eq(y, 48000, np.zeros(2049)), y, atol=1e-6)

    def test_edge_smoothing_no_boundary_dip(self):
        for n in (1, 20, 500):
            np.testing.assert_allclose(smooth_edge(np.ones(n), 64), np.ones(n))

    def test_legato_no_nan_at_exact_hop_duration(self):
        for n in (48000, 48137, 120):
            y = self.tone(n)
            out = vocal_post.legato(y, 48000, .4)
            self.assertEqual(out.shape, y.shape)
            self.assertTrue(np.isfinite(out).all())
            self.assertGreater(float(np.sqrt(np.mean(out[-min(n, 1000):] ** 2))), .03)

    def test_speech_profile_cannot_lower_singing_pitch(self):
        y = self.tone()
        with patch.object(svc, "decode", return_value=y), patch.object(svc, "_f0_median", return_value=330), patch.object(svc, "_pitch_shift") as shifter, patch.object(svc, "write_wav"):
            report = svc.convert_dsp("v.wav", {"f0_median_hz": 100, "hnr_db": 12}, "o.wav")
        shifter.assert_not_called()
        self.assertEqual(report["pitch_shift_semitones"], 0)

    def test_no_stems_never_treats_full_mix_as_voice(self):
        with patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(vocal_pipeline.prepare_stems("mix.mp3", self.d), (None, None))

    def test_requires_both_explicit_stems(self):
        with self.assertRaises(ValueError):
            vocal_pipeline.prepare_stems("mix.mp3", self.d, vocal="v.wav")

    def test_remix_unchanged_stems_reconstruct_original(self):
        v, b = self.tone(48137, 330), self.tone(48137, 110) * .5
        for name, y in [("v.wav", v), ("b.wav", b)]:
            svc.write_wav(y, self.d / name)
        result = vocal_pipeline.remix(self.d / "v.wav", self.d / "v.wav", self.d / "b.wav", self.d / "mix.wav")
        out = svc.decode(self.d / "mix.wav")
        np.testing.assert_allclose(out, v+b, atol=1e-6)
        self.assertTrue(result["backing_preserved"])

    def test_remix_bad_duration_is_rejected(self):
        svc.write_wav(self.tone(), self.d / "ref.wav")
        svc.write_wav(self.tone(48000), self.d / "bad.wav")
        with self.assertRaisesRegex(RuntimeError, "길이"):
            vocal_pipeline.remix(self.d / "bad.wav", self.d / "ref.wav", self.d / "ref.wav", self.d / "out.wav")

    def test_dropout_screen_and_scope(self):
        y = self.tone(240000)
        svc.write_wav(y, self.d / "steady.wav")
        self.assertEqual(vocal_pipeline.continuity_metrics(self.d / "steady.wav")["penalty"], 0)
        y[96000:100800] = 0
        svc.write_wav(y, self.d / "dip.wav")
        report = vocal_pipeline.continuity_metrics(self.d / "dip.wav")
        self.assertGreater(report["dropout_count"], 0)
        self.assertEqual(report["scope"], "mix_energy_only")

    def test_rvc_failure_does_not_switch_to_dsp(self):
        with patch.object(svc, "load_profile", return_value={"name": "me"}), patch.object(svc, "rvc_available", return_value=True), patch.object(svc, "convert_rvc", side_effect=RuntimeError("bad")), patch.object(svc, "convert_dsp") as dsp, self.assertRaisesRegex(RuntimeError, "대체하지"):
            svc.convert_vocal("in.wav", "profile.json", "out.wav")
        dsp.assert_not_called()

    def test_coach_preserves_melody_by_default(self):
        source, out = self.d / "v.wav", self.d / "out.wav"
        svc.write_wav(self.tone(96000), source)
        with patch.object(vocal_coach, "pitch_style", side_effect=AssertionError("must not shift")):
            report = vocal_coach.polish_star(source, out, intensity=.5)
        self.assertFalse(report["stages"][0]["applied"])
        self.assertTrue(np.isfinite(svc.decode(out)).all())

    def test_finalizer_without_stems_preserves_mp3_audio_bytes(self):
        import subprocess
        source, out = self.d / "source.mp3", self.d / "out.mp3"
        wav = self.d / "test.wav"
        svc.write_wav(self.tone(), wav)
        subprocess.run([svc._ffmpeg(), "-y", "-v", "error", "-i", str(wav), str(source)], check=True)
        import song_finalize
        args = ["song_finalize", "--input", str(source), "--output", str(out), "--cover", "none", "--no-lyrics-tag"]
        with patch("sys.argv", args), patch.object(vocal_pipeline, "prepare_stems", return_value=(None, None)), patch.object(vocal_coach, "polish_star") as coach, patch.object(vocal_post, "polish_song") as polish:
            self.assertEqual(song_finalize.main(), 0)
        coach.assert_not_called()
        polish.assert_not_called()
        self.assertEqual(source.read_bytes(), out.read_bytes())


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import widget_app
        from PyQt6.QtWidgets import QWidget
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(widget_app, "BATCH_OUT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.widget_app = widget_app
        self.studio = widget_app.YunseulLMMStudio.__new__(widget_app.YunseulLMMStudio)
        QWidget.__init__(self.studio)
        self.studio.ebook_worker = None
        self.tab = self.studio.create_ebook_tab()
        self.merge = self.studio.create_merge_tab()
        self.addCleanup(self.studio.deleteLater)
        self.addCleanup(self.tab.deleteLater)
        self.addCleanup(self.merge.deleteLater)

    def test_initial_editors_blank_not_sample_files(self):
        self.assertEqual(self.studio.edit_topic.text(), "")
        self.assertEqual(self.studio.txt_ebook.toPlainText(), "")
        self.assertEqual(self.studio.list_sources.count(), 0)
        self.assertEqual(self.studio.edit_merge_prompt.text(), "")
        self.assertEqual(self.studio.txt_merge_res.toPlainText(), "")

    def test_restore_latest_manual_edits_and_settings(self):
        s = self.studio
        s.edit_topic.setText("내 원고")
        s.txt_ebook.setPlainText("직전에 직접 고친 본문")
        s.edit_ebook_persona.setText("담백한 1인칭")
        s.spin_ebook_pages.setValue(120)
        s._save_ebook_workspace()
        s.txt_ebook.clear()
        s.edit_topic.clear()
        s._restore_ebook_workspace()
        self.assertEqual(s.txt_ebook.toPlainText(), "직전에 직접 고친 본문")
        self.assertEqual(s.edit_topic.text(), "내 원고")
        self.assertEqual(s.spin_ebook_pages.value(), 120)
        self.assertEqual(s.edit_ebook_persona.text(), "담백한 1인칭")

    def test_new_draft_backs_up_then_stays_empty_on_restore(self):
        from PyQt6.QtWidgets import QMessageBox
        s = self.studio
        s.txt_ebook.setPlainText("소중한 이전 원고")
        s.edit_topic.setText("이전 주제")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            s.new_ebook_workspace()
        self.assertEqual(s.txt_ebook.toPlainText(), "")
        self.assertEqual(s.edit_topic.text(), "")
        backups = list(self.root.glob("ebook-backup-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(workspace_state.load_draft(backups[0])["body"], "소중한 이전 원고")
        s._restore_ebook_workspace()
        self.assertEqual(s.txt_ebook.toPlainText(), "")

    def test_new_draft_save_failure_does_not_clear(self):
        from PyQt6.QtWidgets import QMessageBox
        s = self.studio
        s.txt_ebook.setPlainText("보호할 원고")
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), patch.object(QMessageBox, "warning"), patch.object(s, "_save_ebook_workspace", return_value=False):
            s.new_ebook_workspace()
        self.assertEqual(s.txt_ebook.toPlainText(), "보호할 원고")

    def test_merge_paths_keep_external_files_and_clear(self):
        from PyQt6.QtWidgets import QFileDialog
        from PyQt6.QtCore import Qt
        source = self.root / "outside.txt"
        source.write_text("이번 작업의 원문", encoding="utf-8")
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(source), "")):
            self.studio.add_source_file()
        self.assertEqual(self.studio.list_sources.item(0).data(Qt.ItemDataRole.UserRole), str(source.resolve()))
        self.studio.txt_merge_res.setPlainText("결과")
        self.studio.clear_merge_workspace()
        self.assertEqual(self.studio.list_sources.count(), 0)
        self.assertEqual(self.studio.txt_merge_res.toPlainText(), "")
        self.assertTrue(source.exists())

    def test_merge_uses_actual_source_not_shipped_folder(self):
        from PyQt6.QtWidgets import QFileDialog
        from unittest.mock import MagicMock
        s = self.studio
        source = self.root / "outside.txt"
        source.write_text("외부 원문으로만 작성", encoding="utf-8")
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(source), "")):
            s.add_source_file()
        s.edit_merge_prompt.setText("정리")
        s.port, s.cfg = 11434, {}
        with patch.object(s, "is_server_alive", return_value=True), patch.object(s, "check_vram_for_task", return_value=(True, "")), patch.object(self.widget_app, "TextGenWorker") as worker:
            s.start_merge_generation()
            self.assertIn("외부 원문으로만 작성", worker.call_args.args[2])

    def test_song_studio_gemini_controls_and_warning_report(self):
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtWidgets import QLineEdit
        from song_studio import SongStudio
        host = self.studio
        host.player, host.audio_output = QMediaPlayer(), QAudioOutput()
        host.cfg, host.port = {}, 11434
        song = SongStudio(host, self.root, self.root / "songs", self.root / "models", self.root / "python")
        self.addCleanup(song.deleteLater)
        self.assertEqual(song.combo_lyrics_provider.currentData(), "gemini")
        self.assertEqual(song.edit_gemini_key.echoMode(), QLineEdit.EchoMode.Password)
        self.assertIn("trot", [b.property("genre") for b in song.genre_group.buttons()])
        target = self.root / "songs" / "song-test.mp3"
        target.write_bytes(b"not an audio fixture")
        target.with_suffix(".finalize.json").write_text(json.dumps({"warnings": ["Demucs 없음: 보컬 처리 생략"]}))
        song.edit_lyrics.setPlainText("[Verse]\n연습 가사")
        with patch.object(song, "refresh_library"):
            song._on_finalize_done(True, str(target), "")
        self.assertIn("일부 단계 생략", song.lbl_progress.text())
        self.assertIn("Demucs", song.lbl_progress.toolTip())
        from PyQt6.QtCore import QUrl
        song.player.stop()
        song.player.setSource(QUrl())

    def test_new_draft_during_generation_is_blocked(self):
        from PyQt6.QtWidgets import QMessageBox
        from unittest.mock import MagicMock
        s = self.studio
        s.txt_ebook.setPlainText("진행 중 원고")
        s.ebook_worker = MagicMock()
        s.ebook_worker.isRunning.return_value = True
        with patch.object(QMessageBox, "information"):
            s.new_ebook_workspace()
        self.assertEqual(s.txt_ebook.toPlainText(), "진행 중 원고")


if __name__ == "__main__":
    unittest.main(verbosity=2)
