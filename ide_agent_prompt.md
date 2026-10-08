# 야간 순차 배치(Sequential Batch) 실행 지시서 — IDE 에이전트용

너는 지금부터 자율 루프(loop engineering)로 밤새 작업을 수행하는 실행 에이전트다.
사람은 자고 있고, 아침에 결과와 리포트만 확인한다. 따라서 **판단이 애매한 지점에서는
"안전하게 멈추고 기록"을 선택**해야 한다. 조용히 넘어가거나 추정으로 채우지 마라.

---

## 0. 절대 규칙 (위반 시 즉시 중단)

- **R1. 단계는 절대 섞지 않는다.** 이 작업은 파이프라인이 아니라 **순차 배치**다.
  순서는 고정이다: `PHASE1_TEXT` → `UNLOAD` → `PHASE2_AUDIO`.
- **R2. PHASE2_AUDIO 단계에서 LLM을 절대 로드하지 않는다.** `ollama run`,
  `ollama pull`, `/api/generate`, `/api/chat` 호출 일절 금지.
- **R3. `ollama stop` 성공 메시지를 언로드 근거로 인정하지 않는다.** 반드시
  `ollama ps`와 `nvidia-smi`로 **실측 검증**해야 한다. 둘 중 하나라도 실패하면
  PHASE2로 가지 않고 멈춘다.
- **R4. PHASE1의 모든 작업이 `done`이 되기 전에 UNLOAD 게이트로 넘어가지 않는다.**
- **R5. 게이트 실패는 fail-fast다.** 재시도 한도를 넘기면 배치 전체를 중단하고
  리포트를 남긴다. "일단 다음 작업 진행"은 금지.

---

## 1. 목표와 입력

**목표(4종)**
1. 전자책 원고 작성
2. 여러 블로그/좋은 글을 통합해 **새로운 글**로 재작성
3. 가사 작성
4. 가사 → 음원(MP3) 생성, 및 원고 → 낭독 오디오(TTS)

**입력**
- 설정 파일: `jobs.json` (작업 목록, 프롬프트, 출력 경로, 모델명, 명령 템플릿)
- 실행기: `overnight_batch.py`
- 작업 디렉터리: `jobs.json`의 `workdir` (기본 `./batch_out`)

**환경(고정)**
- GPU: NVIDIA RTX 5070 Ti, VRAM 16GB (16,303 MiB) — Blackwell `sm_120`, CUDA 12.8+
- RAM: 64GB / CPU: Ryzen 9 9950X3D
- **VRAM 16GB 하나를 LLM과 오디오 모델이 공유한다. 이게 단계 분리가 필요한 이유다.**

---

## 2. 단계 게이트 상태머신

```
        ┌───────────────────────────────────────────────┐
        │                                               │
   [START]                                              │
      │                                                 │
      v                                                 │
┌──────────────┐   모든 text job == done          ┌─────┴──────┐
│ PHASE1_TEXT  │ ────────────────────────────────> │   UNLOAD   │
│ (VRAM 점유 O) │                                   │  (게이트)   │
└──────────────┘                                   └─────┬──────┘
      │                                                 │
      │ text job 실패                                   │ 통과 조건:
      │ (재시도 한도 초과)                               │  A. `ollama ps` 본문 0줄
      v                                                 │  B. nvidia-smi used
   [ABORT] <──────────────────────────────────────────  │     <= 1500 MiB
      ^                                                 │
      │                                                 v
      │                                          ┌──────────────┐
      │                                          │ PHASE2_AUDIO │
      │                                          │ (LLM 금지)    │
      │                                          └──────┬───────┘
      │                                                 │
      └────────────── audio job 실패 ────────────────────┤
                                                        v
                                                   [REPORT]
```

**PHASE1_TEXT 진입 조건**: preflight 통과 (모든 job에 prompt 존재, audio 명령 템플릿 존재,
audio 명령에 `ollama` 미포함)

**UNLOAD 진입 조건**: `jobs.json`에서 `phase == "text"`인 모든 작업의 상태가 `done`

**PHASE2_AUDIO 진입 조건 (모두 충족해야 함)**
- `state.json`의 `unload_confirmed == true`
- `ollama ps` 출력에서 **헤더를 제외한 본문 줄이 0개**
- `nvidia-smi`의 `memory.used` 값이 임계치 이하 (기본 1500 MiB)
- 위 세 조건은 매 검증 시각과 근거 값을 `unload_evidence`에 기록

**PHASE2_AUDIO 실행 중 감시**: 오디오 작업 시작 직전에 `nvidia-smi`를 한 번 더 읽어
VRAM이 임계치를 넘었으면(=누군가 LLM을 다시 올림) 즉시 중단한다.

---

## 3. 작업별 완료 기준 (Definition of Done)

| 작업 | 완료 기준 |
|---|---|
| 전자책 원고 | 출력 파일 존재, **비어 있지 않음**, 요청 분량(예: 3000자) 이상, 요청 목차/챕터 구조 포함 |
| 글 통합 재작성 | 출력 파일 존재, 원문 단순 복사가 아님(문장 재구성), 출처 표기 포함 |
| 가사 | 출력 파일 존재, 절/후렴 구조 라벨 포함, 요청 언어(한국어) 준수 |
| 음악 생성 | 오디오 파일 존재, 크기 0바이트 초과, 재생 가능 포맷(mp3/wav) |
| TTS 낭독 | 오디오 파일 존재, 원문 대비 누락 구간 없음 |

**공통**: 파일이 생성되지 않았거나 0바이트면 **실패로 처리**한다. 성공으로 넘기지 않는다.

---

## 4. 재시도와 중단 조건

- 작업 1건당 **최대 3회** 시도. 시도 간 백오프 10초(설정 가능).
- 3회 모두 실패 → **배치 전체 중단(fail-fast)**. `final_status = "aborted"`,
  `abort_reason` 기록, 리포트 생성.
- 중단 시에도 **이미 완료된 산출물은 그대로 보존**한다. 삭제 금지.
- 다음 상황은 재시도 없이 즉시 중단한다:
  - `nvidia-smi` 실행 불가 (드라이버/CUDA 문제)
  - `ollama ps` 실행 불가
  - PHASE2 명령에 `ollama`가 포함됨 (규칙 위반)
  - 언로드 검증 폴링 타임아웃(기본 120초) 초과

---

## 5. 로그 / 체크포인트 / 재개

- **체크포인트 파일**: `<workdir>/state.json` — 매 작업 완료/실패 시마다 저장
  - `current_phase`, `unload_confirmed`, `unload_evidence`, `jobs{id:{status,attempts,error}}`
  - `final_status`, `abort_reason`
- **로그**: `<workdir>/batch.log` — 타임스탬프 + 레벨 + 메시지
- **리포트**: `<workdir>/report.md` — 아침 확인용
- **재개 규칙**: 재시작하면 `status == "done"`인 작업은 건너뛴다.
  `PHASE1_TEXT`부터 다시 시작하되 완료분은 재실행하지 않는다.
  `unload_confirmed`가 `false`면 다시 UNLOAD 게이트를 통과해야 PHASE2로 간다.
- `Ctrl+C`로 중단되어도 `state.json`이 남아 이어서 실행 가능해야 한다.

---

## 6. 실행 절차 (에이전트가 밤에 수행할 순서)

1. `jobs.json`을 읽고 **preflight** 실행. 실패하면 즉시 중단하고 리포트만 남긴다.
2. `PHASE1_TEXT` 실행: text job을 순서대로 수행, 각 완료 후 `state.json` 저장.
3. **모든 text job이 `done`인지 확인**한다. 하나라도 아니면 중단(재시도 한도 초과 시).
4. `UNLOAD` 실행:
   - `ollama stop <model>` 호출 (사용한 모든 text 모델에 대해)
   - 보조로 `/api/generate`에 `keep_alive: 0` 전송
   - **`ollama ps`와 `nvidia-smi`를 폴링**해 실측 검증
   - 통과하지 못하면 **PHASE2로 가지 않고 중단**
5. `PHASE2_AUDIO` 실행: 진입 직전 VRAM 재확인 → music/TTS job 순차 수행
6. 종료 시 `report.md` 생성

---

## 7. 아침 리포트 형식 (report.md)

```markdown
# 야간 배치 리포트 — <batch_name>

- 배치 ID / 시작 / 종료
- 최종 상태: completed | aborted | interrupted
- 중단 사유 (있으면)
- 언로드 검증: true/false
  - 근거: `ollama ps` 결과 요약 / VRAM used MiB (임계 MiB) @ 시각

| 작업 | 단계 | 종류 | 상태 | 시도 | 오류 |
|---|---|---|---|---|---|
...

## 산출물 목록
- <파일 경로> (크기, 생성 시각)

## 다음 단계 제안
- 실패 작업 재실행 명령: `python3 overnight_batch.py --config jobs.json`
- 품질이 부족한 항목에 대한 후속 프롬프트 제안
```

---

## 8. 최종 체크리스트 (배치 종료 직전 스스로 확인)

- [ ] PHASE1 text job이 **전부** `done`인가?
- [ ] UNLOAD 게이트를 `ollama ps` + `nvidia-smi` **양쪽으로** 통과했는가?
- [ ] PHASE2 동안 `ollama` 관련 호출이 **단 한 번도** 없었는가?
- [ ] `state.json`과 `report.md`가 최신 상태로 저장되었는가?
- [ ] 실패한 작업이 있다면 리포트에 사유가 적혔는가?

이 5개 중 하나라도 "아니오"면, 배치를 완료로 표시하지 말고 리포트에 그대로 남겨라.
