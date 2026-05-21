# 2_Class_Embeder — 고신뢰성 RAG 데이터 파이프라인

적응형 청킹 및 다중 에이전트(수집 → 검수 → 결재) 검증 기반 RAG 파이프라인입니다.  
**실험군(Agent)** 과 **대조군(Legacy)** 을 동일 Gemini 모델로 비교할 수 있습니다.

- **저장소:** [https://github.com/dydals99/2_Class_Embeder](https://github.com/dydals99/2_Class_Embeder)

---

## 연구 개요

| 구분 | 파이프라인 | 설명 |
|------|------------|------|
| **실험군** | `agent_crawl.py` | Crawl4AI 수집 → **Verifier**(너무 넓으면 반려) → **Collector 피드백**으로 URL 축소 재수집 → Approver → 최소 구간만 청킹 → Chroma `rag_proposed` |
| **대조군** | `legacy_crawl.py` | Crawl4AI 수집 → 검수 없음 → 고정 길이 청킹 → Chroma `rag_legacy` |
| **질의** | `rag.py` | 두 컬렉션에 RAG 질의 (동일 Gemini) |

---

## 요구 사항

- Python **3.10+** (3.12 권장)
- Windows / macOS / Linux
- [Gemini API 키](https://aistudio.google.com/apikey)
- (선택) `CRAWL_MODE=browser` 사용 시 Playwright Chromium

---

## 설치

```powershell
# 저장소 클론
git clone https://github.com/dydals99/2_Class_Embeder.git
cd 2_Class_Embeder

# 가상환경 (폴더 이름은 자유: venv, myvenv 등)
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1   # Windows PowerShell
# source myvenv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

브라우저 크롤이 필요할 때만:

```powershell
python -m playwright install chromium
```

---

## 환경 변수 (.env) 설정

### 1) 파일 만들기

프로젝트 **루트**에서:

```powershell
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

### 2) 반드시 수정할 값

| 변수 | 필수 | 설명 |
|------|------|------|
| `GEMINI_API_KEY` | **예** | Google AI Studio에서 발급한 API 키 |

### 3) 자주 쓰는 선택 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | LLM (실험·대조 동일) |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | 임베딩 |
| `GEMINI_FALLBACK_MODEL` | `gemini-2.0-flash` | 429/503 시 폴백 |
| `CRAWL_MODE` | `http` | `http`(권장) 또는 `browser` |
| `MAX_COLLECT_RETRIES` | `2` | 검수 반려 후 Collector 재시도 횟수 |
| `PROPOSED_MAX_CHUNKS_PER_DOC` | `3` | 실험군 문서당 최대 청크 수 |
| `RAG_TOP_K` | `5` | RAG 검색 청크 개수 |

전체 목록은 [`.env.example`](.env.example) 를 참고하세요.

### 4) 보안 주의

- **`.env`는 Git에 올리지 마세요.** (`.gitignore`에 포함됨)
- **API 키를 README·이슈·커밋에 적지 마세요.**
- `.env.example`에는 `your_gemini_api_key_here` 같은 **플레이스홀더만** 둡니다.

### 5) 로컬 전용 데이터 경로 (자동 생성)

| 경로 | 용도 |
|------|------|
| `data/chroma/` | ChromaDB 저장 |
| `data/exports/` | 청크 CSV (`proposed_chunks.csv`, `legacy_chunks.csv`) |
| `Documents/` | 업로드용 PDF/DOCX/TXT (실험군 `--uploads-only`) |

---

## 실행 방법

가상환경 활성화 후 **프로젝트 루트**에서 실행합니다.

### A/B 비교 권장 순서 (동일 URL·동일 질의)

```powershell
# 1) 대조군 인덱스 구축
python legacy_crawl.py --query "for 문 관련" --urls https://docs.python.org/ko/3/tutorial/index.html --reset

# 2) 실험군 인덱스 구축 (검수 → 피드백 → 좁은 URL 재수집)
python agent_crawl.py --query "for 문 관련" --urls https://docs.python.org/ko/3/tutorial/index.html --reset

# 3) RAG 비교
python rag.py --query "for문에 대한 URL 알려줘" --collection both
```

`--reset` 은 해당 파이프라인 Chroma 컬렉션만 비우고 다시 쌓습니다. **공정 비교 시 둘 다 `--reset` 권장.**

---

### 대조군 — `legacy_crawl.py`

```powershell
python legacy_crawl.py --query "RAG 청킹 전략" --urls https://example.com/docs --reset
```

| 옵션 | 설명 |
|------|------|
| `--query`, `-q` | 연구·수집 기준 질의 (**필수**) |
| `--urls` | 크롤할 시드 URL (여러 개 가능) |
| `--reset` | `rag_legacy` 컬렉션 초기화 후 재적재 |

---

### 실험군 — `agent_crawl.py`

```powershell
# 웹: index 등 넓은 URL → 검수가 너무 넓다고 하면 Collector가 subsection URL 재크롤
python agent_crawl.py --query "for 문 관련" --urls https://docs.python.org/ko/3/tutorial/index.html --reset

# 이미 좁은 URL을 알 때
python agent_crawl.py --query "for문 예제" --urls https://docs.python.org/ko/3/tutorial/controlflow.html#for-statements --reset

# 로컬 문서만 (수집 생략, Documents/ 폴더)
python agent_crawl.py --query "보고서 요약" --uploads-only --reset
```

| 옵션 | 설명 |
|------|------|
| `--query`, `-q` | **필수** |
| `--urls` | 시드 URL (`--uploads-only`가 아니면 사용) |
| `--uploads-only` | `Documents/` 파일만 검수·결재·청킹 |
| `--documents-dir` | 기본 `Documents/` 대신 다른 폴더 |
| `--reset` | `rag_proposed` 컬렉션 초기화 |

**로그에서 확인할 것**

- `[verifier] REJECT (too broad) → feedback for Collector`
- `[collector] Received Verifier feedback — narrowing crawl scope...`
- `[pipeline] Re-crawled narrower doc: ...`
- `[processor] Indexed N chunk(s), ...` (N이 작을수록 좁은 scope)

---

### RAG 질의 — `rag.py`

```powershell
python rag.py --query "for문 예제 설명해줘"
python rag.py --query "..." --collection proposed   # 실험군만
python rag.py --query "..." --collection legacy     # 대조군만
python rag.py --query "..." --collection both       # 둘 다 (기본)
python rag.py --query "..." --top-k 8
```

---

## 프로젝트 구조

```
.
├── config.py              # 공통 설정 (.env 로드)
├── agent_crawl.py         # 실험군 진입점
├── legacy_crawl.py        # 대조군 진입점
├── rag.py                 # RAG 질의
├── requirements.txt
├── .env.example           # 환경 변수 템플릿 (키 없음)
├── Documents/             # 사용자 업로드 문서
├── core/
│   ├── gemini.py          # Gemini LLM / Embedding
│   ├── crawler.py         # Crawl4AI
│   ├── chunking.py        # 고정 / 적응형 청킹
│   ├── vectorstore.py     # ChromaDB
│   ├── scope_llm.py       # LLM scope·구간 추출
│   └── agents/
│       ├── collector.py
│       ├── verifier.py
│       ├── approver.py
│       ├── processor.py
│       └── orchestrator.py
└── data/                  # 로컬 전용 (Git 제외)
    ├── chroma/
    └── exports/
```

---

## GitHub 저장소 연결 (최초 1회)

이미 클론한 경우 remote만 확인:

```powershell
git remote -v
# origin  https://github.com/dydals99/2_Class_Embeder.git
```

로컬 프로젝트를 처음 연결할 때:

```powershell
git init
git remote add origin https://github.com/dydals99/2_Class_Embeder.git
git add .
git commit -m "Initial commit: agent/legacy RAG pipeline"
git branch -M main
git push -u origin main
```

이미 원격에 커밋이 있으면:

```powershell
git pull origin main --rebase
git push origin main
```

---

## 자주 나는 오류

| 증상 | 대응 |
|------|------|
| `Missing required environment variable: GEMINI_API_KEY` | `.env` 생성 및 키 입력 |
| `429` / `quota` | API 한도 초과 → 대기, 유료 플랜, 호출 줄이기 |
| Playwright `Executable doesn't exist` | `CRAWL_MODE=http` 유지 또는 `python -m playwright install chromium` |
| 실험군 `chunks_indexed: 0` | 검수 임계값 조정 또는 더 구체적인 `--urls` / 질의 |
| 대조/실험 RAG 결과가 섞임 | 각각 `--reset` 후 **같은 URL**로 재인덱싱 |

---

## 라이선스 / 과제

MJC 프로젝트 수업용 연구 구현입니다.  
문의·이슈는 GitHub Issues를 이용해 주세요.
