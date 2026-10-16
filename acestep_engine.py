#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acestep_engine.py - 호환용 얇은 래퍼.

사인파 합성(generate_ballad_music) 코드는 완전히 삭제했다.
이 파일은 인자를 그대로 audio_engines/acestep_official_runner.py 에 넘기고,
러너가 없으면 대체 음원을 만들지 않고 오류로 종료한다.
"""
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "audio_engines" / "acestep_official_runner.py"


def main() -> int:
    if not RUNNER.exists():
        print(f"[acestep] 오류: 공식 러너가 없습니다: {RUNNER}")
        return 1
    return subprocess.run([sys.executable, str(RUNNER)] + sys.argv[1:]).returncode


if __name__ == "__main__":
    sys.exit(main())
