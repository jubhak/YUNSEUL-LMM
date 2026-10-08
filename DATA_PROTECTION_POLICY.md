# [CRITICAL SYSTEM RULE] 학습 데이터 영구 보존 및 삭제 금지 정책 (Data Protection Policy)

## 1. 기본 원칙 (Absolute Rule)
- **`trained_vocal_models/` 디렉터리 내의 모든 파일(모델 가중치 `.pth`, 참조 음원 `ref_vocal.wav`, 메타데이터 `meta.json`, 슬라이스 데이터셋 `work/slices/*.wav`, 음원 캐시 `audio_cache.json` 등)은 이후 어떠한 작업에서도 절대 삭제되어서는 안 됩니다.**
- 사용자가 "학습 데이터 삭제"를 직접 명시적으로 요구하기 전까지, 모든 리팩토링, 재빌드, 캐시 정리, 프로세스 관리 작업 시 해당 디렉터리의 영구 보존이 최우선됩니다.

## 2. 세부 보호 대상
1. **신경망 모델 가중치**: `trained_vocal_models/*/*.pth`
2. **보컬 음색 참조 음원**: `trained_vocal_models/*/ref_vocal.wav`
3. **가창 세그먼트 데이터셋**: `trained_vocal_models/*/work/slices/*.wav`
4. **중복 방지 및 학습 내역 캐시**: `trained_vocal_models/*/audio_cache.json`
5. **모델 메타데이터**: `trained_vocal_models/*/meta.json`
6. **영구 백업 볼트**: `trained_vocal_models/.permanent_vault/`

## 3. 누적 학습 원칙 (Cumulative Preservation)
- 새로운 곡을 추가하여 학습할 경우, 이전 학습 데이터가 덮어써지거나 지워지지 않고 **누적(Accumulation) 및 확장**되도록 보장합니다.
- 이미 학습된 음원은 `[중복 PASS]`를 통해 보존된 상태로 유지되며, 기존 가창 슬라이스는 영구적으로 모델 데이터셋에 포함됩니다.
