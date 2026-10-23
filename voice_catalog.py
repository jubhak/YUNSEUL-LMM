#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_catalog.py - 보컬 모델 분류기 (Qt 의존 없음, 테스트 가능)

문제: `trained_vocal_models/` 안에 (a) 내가 문장을 읽어 만든 '내 목소리' 모델과
      (b) 1000곡 이상으로 학습한 '노래/스타일 코퍼스' 모델이 같이 쌓인다.
      기존 코드는 폴더를 전부 훑어 둘 다 같은 콤보박스에 넣었다.
      → 1000곡 모델이 '남자/여자/별칭' 목록에 섞여 들어가 선택이 어려워졌다.

해결: 이름 하드코딩이 아니라 **메타데이터로 역할을 판정**한다.
  1) meta.json 에 role/model_role/kind 가 있으면 그 값을 따른다.
  2) 없으면 학습 규모(songs_count, songs_with_lyrics, train_minutes)로 추정한다.
  3) 판정 결과를 (voices, song_models, legacy) 로 나눠 돌려준다.

이 모듈은 song_studio.py 와 테스트(test_catalog.py)가 함께 사용한다.
"""
from __future__ import annotations

import json
from pathlib import Path

# 이 곡 수 이상이면 '노래 코퍼스(스타일) 모델'로 본다.
# 내 목소리 녹음은 문장 10~30개(=songs_count 1)라서 여기에 걸리지 않는다.
SONG_MODEL_MIN_SONGS = 100
# 학습 시간(분) 기준 보조 판정: 6시간 이상이면 코퍼스 모델.
SONG_MODEL_MIN_MINUTES = 360

ROLE_VOICE = "voice"
ROLE_SONG = "song"

_BUILTIN_GENDERS = [("남성 보컬 (기본)", "male"), ("여성 보컬 (기본)", "female")]


def _as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def classify_meta(meta: dict, folder_name: str = "") -> dict:
    """meta.json 하나를 보고 역할/표시 정보를 결정한다."""
    meta = dict(meta or {})
    role_raw = str(meta.get("role") or meta.get("model_role") or meta.get("kind") or "").lower()
    songs = _as_int(meta.get("songs_count"))
    minutes = _as_int(meta.get("train_minutes"))
    enrolled = bool(meta.get("enrolled"))

    if role_raw in ("song", "song_model", "style", "corpus", "covers", "artist"):
        role, why = ROLE_SONG, f"role={role_raw}"
    elif role_raw in ("voice", "speaker", "my_voice", "alias"):
        role, why = ROLE_VOICE, f"role={role_raw}"
    elif enrolled:
        role, why = ROLE_VOICE, "enrolled=true (문장 녹음으로 등록)"
    elif songs >= SONG_MODEL_MIN_SONGS or minutes >= SONG_MODEL_MIN_MINUTES:
        role = ROLE_SONG
        why = f"songs_count={songs}, train_minutes={minutes} → 대규모 코퍼스"
    else:
        role, why = ROLE_VOICE, f"songs_count={songs} → 개인 목소리"

    name = str(meta.get("name") or folder_name or "이름없음")
    profile = meta.get("voice_profile") or {}
    gender = meta.get("gender") or profile.get("gender_guess") or ""
    return {
        "name": name,
        "role": role,
        "reason": why,
        "songs_count": songs,
        "train_minutes": minutes,
        "gender": gender,
        "meta": meta,
    }


def scan_models(trained_dir) -> dict:
    """폴더 전체를 스캔해 voices / song_models / legacy / invalid 로 분류."""
    out = {"voices": [], "song_models": [], "legacy": [], "invalid": []}
    root = Path(trained_dir)
    if not root.exists():
        return out
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = mdir / "meta.json"
        if not meta_path.exists():
            out["invalid"].append({"name": mdir.name, "reason": "meta.json 없음"})
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            out["invalid"].append({"name": mdir.name, "reason": f"meta.json 읽기 실패: {e}"})
            continue
        info = classify_meta(meta, mdir.name)
        adapter = meta.get("adapter_path")
        has_adapter = bool(adapter) and (Path(adapter) / "adapter_config.json").exists()
        info["adapter_ok"] = has_adapter
        if not has_adapter:
            # LoRA 파일이 없다 → 생성에 쓸 수 없음(이전 방식).
            # 단, 곡 코퍼스 모델은 목록에서 조용히 사라지지 않도록 song_models 에 남긴다.
            out["legacy"].append(info)
            continue
        if info["role"] == ROLE_SONG:
            out["song_models"].append(info)
        else:
            out["voices"].append(info)
    return out


def voice_combo_entries(scan: dict) -> list:
    """보컬 선택 콤보에 넣을 항목. [{'label','data'}] 형태.

    순서: 기본(학습 없음) → 남성 기본 → 여성 기본 → 내 목소리 별칭들
    1000곡 모델은 여기에 **절대 들어가지 않는다.**
    """
    entries = [{"label": "기본 목소리 (학습 모델 없이)", "data": None}]
    for label, gender in _BUILTIN_GENDERS:
        entries.append({"label": label, "data": {"builtin": True, "gender": gender}})
    for v in scan.get("voices", []):
        tag = "🌟"
        if v.get("gender") == "male":
            tag = "👨"
        elif v.get("gender") == "female":
            tag = "👩"
        sub = f"{v['songs_count']}" if v.get("songs_count") else "문장"
        entries.append({
            "label": f"{tag} {v['name']}  ({sub})",
            "data": v["meta"] | {"name": v["name"]},
        })
    return entries


def song_model_entries(scan: dict) -> list:
    """노래 학습 모델(스타일/코퍼스) 선택 콤보 항목."""
    entries = [{"label": "곡 학습 모델 없음 (스타일 미적용)", "data": None}]
    for s in scan.get("song_models", []):
        songs = s.get("songs_count") or 0
        label = f"🎼 {s['name']}" + (f"  ({songs}곡 학습)" if songs else "")
        entries.append({"label": label, "data": s["meta"] | {"name": s["name"]}})
    # 어댑터가 깨진 코퍼스 모델도 안내만 한다(선택 불가).
    for s in scan.get("legacy", []):
        if s.get("role") == ROLE_SONG:
            entries.append({"label": f"⚠️ {s['name']} (LoRA 파일 없음·재학습 필요)",
                            "data": None, "disabled": True})
    return entries
