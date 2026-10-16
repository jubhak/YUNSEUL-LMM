@echo off
if exist "%~dp0audio_engines\venv_acestep\Scripts\python.exe" (
    "%~dp0audio_engines\venv_acestep\Scripts\python.exe" "%~dp0audio_engines\acestep_official_runner.py" %*
) else (
    python "%~dp0audio_engines\acestep_official_runner.py" %*
)
