# Vietnamese Policy RAG

[![CI](https://github.com/haminhthong/Enterprise-Rag-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Enterprise-Rag-Assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.11.0-005571.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-rank__bm25-6f42c1.svg)](https://github.com/dorianbrown/rank_bm25)

RAG tra cứu chính sách nội bộ bằng tiếng Việt. Version/effective-date được xử
lý trước retrieval, ACL được lọc trước candidate, Dense và BM25 hợp nhất bằng
RRF, sau đó Cross-Encoder rerank và evidence threshold quyết định có tạo câu
trả lời hay không.

## Bài Toán & Phạm Vi Ứng Dụng (Problem & Scope)

Kho chính sách nội bộ có nhiều version, quyền truy cập khác nhau và câu hỏi
chứa cả ngữ nghĩa lẫn mã/số liệu. RAG đơn giản dễ lấy nhầm policy cũ, đưa tài
liệu ngoài quyền vào candidate hoặc trả lời khi thiếu bằng chứng.

Project tập trung vào corpus policy tiếng Việt và giữ các xử lý có giá trị:

- catalog quản lý `document_id`, `policy_key`, version, effective dates, status,
  department và `allowed_groups`;
- chỉ version `ACTIVE` còn hiệu lực tại ngày build được chunk và index;
- API key được map server-side thành `AccessContext`, client không tự gửi ACL;
- retrieval dùng Dense + BM25 + RRF, Cross-Encoder là bước rerank candidate;
- response có ba mode: `answer`, `sources_only`, `abstain`;
- citation validator kiểm tra reference ID và sentence-level coverage, không
  tuyên bố semantic entailment.

Ngoài phạm vi: OCR cho PDF scan, semantic entailment tự động, vector database,
multi-region HA và feedback loop chưa có pipeline phân tích thật.

## Kiến Trúc Kỹ Thuật Duy Nhất (Canonical Technical Architecture)

Sơ đồ Mermaid dưới đây là luồng duy nhất chi phối code, artifact, cấu hình và
báo cáo.

~~~mermaid
flowchart TD
    subgraph OFFLINE["OFFLINE INDEX PIPELINE"]
        D["Policy documents"] --> C["Catalog validation"]
        M["knowledge_catalog.yaml"] --> C
        C --> Q["Document QA"]
        Q --> K["Structure-aware chunks"]
        K --> I["Dense + BM25 per ACL group"]
        I --> A["artifacts/groups/<group>"]
        A --> V["Count validation"]
    end

    subgraph ONLINE["ONLINE QUERY PIPELINE"]
        U["User + X-API-Key"] --> X["Server-side AccessContext"]
        X --> F["Authorized search space"]
        F --> H["Dense + BM25"]
        H --> R["RRF"]
        R --> E["Cross-Encoder"]
        E --> G{"Evidence threshold"}
        G -->|insufficient| AB["ABSTAIN"]
        G -->|sufficient| L["Grounded LLM"]
        L --> CV["Citation reference validation"]
        CV -->|valid| AN["ANSWER"]
        CV -->|invalid| SO["SOURCES_ONLY"]
        E -. unavailable .-> SO
        L -. unavailable .-> SO
        AB --> O["REST response"]
        AN --> O
        SO --> O
    end

    A -. loaded by .-> F
    B["Dev/Test evaluation queries"] --> T["Dev tuning + Test report"]
    T --> J["reports/*.json"]
~~~

### Luồng offline

`scripts/build_index.py` đọc catalog, lọc tài liệu active theo `date.today()`,
chạy Document QA, chunk theo cấu trúc, encode chunk rồi ghi artifact hiện hành.
Mỗi ACL group có `index.faiss`, `chunks.json` và `bm25_index.json`.
`documents.json` giữ metadata tối thiểu của tài liệu đã index. Không có active
pointer, release directory, atomic promotion hoặc manifest chain.

### Luồng online

`X-API-Key` được `src/security.py` map thành `AccessContext`.
`src/retrieval.py` chỉ mở các group giao với context và lọc lại
`allowed_groups` trước RRF. Dense và BM25 mỗi nhánh lấy tối đa 30 candidate;
RRF dùng `k=60`, sau đó tối đa 20 candidate được Cross-Encoder rerank. Service
chỉ gọi LLM khi reranker neural và evidence threshold cùng đạt.

## Luồng dữ liệu

| Đầu vào | Xử lý | Kết quả |
|---|---|---|
| `data/raw` + catalog | Catalog validation + active-date filter | Active catalog entries |
| Active entries + tài liệu | Parser + Document QA + chunking | Chunk có version, page, section, ACL |
| Chunk + embedding | Build Dense/BM25 theo group | `artifacts/groups/<group>/` |
| API key + câu hỏi | Server-side authentication | `AccessContext` + query |
| Query + context | Dense/BM25 -> RRF -> Cross-Encoder | Reranked evidence |
| Evidence | Threshold -> grounded generation -> citation check | `answer` / `sources_only` / `abstain` |

## Benchmark & Evaluation Status

Evaluation có code cho Recall@K, MRR, nDCG, evidence recall, abstention,
latency, citation reference validity và sentence-level citation coverage. Repo
không điền số giả khi chưa có locked-test run tái lập với model runtime.

> Current release status: evaluation pipeline implemented; headline benchmark pending reproducible locked-test run.

| Pipeline | Recall@K | MRR | nDCG@K | Citation validity | Answer coverage |
|---|---:|---:|---:|---:|---:|
| BM25 | Pending | Pending | Pending | Pending | Pending |
| Dense | Pending | Pending | Pending | Pending | Pending |
| Hybrid RRF | Pending | Pending | Pending | Pending | Pending |
| + Cross Encoder | Pending | Pending | Pending | Pending | Pending |

`scripts/evaluate.py` tune threshold chỉ trên `dev`, sau đó chạy canonical trên
`test` và ghi `reports/dev_metrics.json`, `reports/final_test_metrics.json`.
`scripts/ablation_experiments.py` chỉ so sánh baseline trên dev. Metric nên
theo dõi thêm là expired-policy retrieval và unauthorized retrieval rate; mục
tiêu security là không có ACL leakage.

Threshold `1.72` hiện là default/provisional raw reranker logit. Không gọi nó là
đã calibrated cho đến khi có Dev report tái lập; test không được dùng để chọn
threshold.

## Corpus / Data Snapshot

Đây là quy mô corpus mẫu đã kiểm tra từ catalog, raw data và evaluation files,
không phải headline benchmark:

~~~yaml
Documents: 18 catalog records; 17 ACTIVE tại snapshot 2026-09-08
Pages: null với TXT/MD/DOCX; PDF số có page metadata khi parser đọc được
Chunks: 38 active chunks với cấu hình structure-aware 250/40 đã kiểm tra
Policies: 17 active policy_key
Languages: Vietnamese (vi)
Evaluation queries: 50 human-authored (12 dev, 38 test) + 4 synthetic regression
~~~

~~~text
Corpus
├── Vietnamese internal policy documents
├── versioned policy_key
├── effective dates
└── ACL groups
~~~

`annual_leave_policy_2025.txt` là hard negative đã hết hiệu lực. Version đúng
được chọn bởi catalog và ngày hiệu lực, không chọn theo similarity tên file.

## Ví dụ RAG hoàn chỉnh

~~~text
Question
  "Nhân viên chính thức có bao nhiêu ngày phép năm?"

Retrieved chunks
  - annual_leave_policy_2026.txt, ACTIVE, allowed_groups=[employee, hr]
  - annual_leave_policy_2025.txt, ARCHIVED, bị loại trước retrieval

Reranked evidence
  - C1: annual_leave_policy_2026.txt / mục "Tiêu chuẩn phép năm"

Decision
  - API key hợp lệ -> AccessContext có group employee
  - evidence gate đạt
  - Cross-Encoder và LLM sẵn sàng

Answer
  "Nhân viên chính thức được hưởng 12 ngày phép năm có lương. [C1]"

Citations
  - [C1] trỏ tới chunk active, version hiện hành trong catalog
~~~

Không có chunk được phép hoặc evidence không đạt threshold -> `abstain`. Có
nguồn nhưng reranker/LLM/citation validation không sẵn sàng -> `sources_only`.

## Why This Is Enterprise RAG

- Effective-date filtering: không trộn policy cũ với policy đang hiệu lực.
- ACL before retrieval: tài liệu ngoài quyền không vào candidate hoặc prompt.
- Hybrid retrieval: Dense giữ ngữ nghĩa, BM25 giữ từ khóa/mã/số liệu, RRF hợp
  nhất theo rank thay vì cộng hai thang điểm khác nhau.
- Abstention: evidence không đủ thì từ chối thay vì đoán.
- Reproducible current artifact: build từ catalog/corpus active hiện tại và
  validate count giữa FAISS, BM25 và chunks trước khi phục vụ.

## Contract An Toàn

- ACL được kiểm tra trước retrieval.
- Chỉ document `ACTIVE` và còn hiệu lực được index.
- Evidence không đủ -> `abstain`.
- `reranker_logit` không phải probability.
- Citation validator không đồng nghĩa với semantic entailment.

Chi tiết payload và mã lỗi xem tại [docs/API.md](docs/API.md).

## Cấu Trúc Thư Mục Dự Án (Project Structure)

~~~text
Rag-Knowledge-Assistant/
├── configs/
│   └── knowledge_catalog.yaml       # Identity, version, ngày hiệu lực, ACL
├── data/
│   ├── raw/                         # Corpus mẫu, được CI tạo lại
│   └── evaluation/                  # Dev/Test và synthetic regression
├── artifacts/                       # Sinh khi build, bị ignore khỏi Git
│   ├── config.json
│   ├── documents.json
│   └── groups/<group>/
│       ├── index.faiss
│       ├── chunks.json
│       └── bm25_index.json
├── docs/API.md                      # API chi tiết
├── scripts/
│   ├── download_data.py              # Tạo corpus mẫu
│   ├── validate_catalog.py            # Validate catalog và coverage
│   ├── build_index.py                # Full rebuild artifact
│   ├── evaluate.py                   # Dev tuning + Test report
│   └── ablation_experiments.py       # Baseline trên Dev
├── src/
│   ├── ingestion/
│   │   ├── parsers.py                # TXT/MD/PDF/DOCX
│   │   ├── chunking.py               # Structure-aware + baseline
│   │   └── models.py                 # Chunk và DocumentBlock
│   ├── catalog.py                    # Document catalog
│   ├── security.py                   # API key -> AccessContext
│   ├── index.py                      # Build artifact theo ACL group
│   ├── ranking.py                    # BM25, RRF, Cross-Encoder
│   ├── retrieval.py                  # ACL-filtered retrieval
│   ├── generation.py                 # Grounded prompt/citation checks
│   ├── service.py                    # answer/sources_only/abstain
│   ├── api.py                        # FastAPI: /health, /query
│   └── evaluate.py                   # Metrics và threshold tuning
├── tests/
│   ├── test_catalog_and_access.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_smoke.py
├── .github/workflows/ci.yml          # Lint, test, Docker smoke test
├── Dockerfile
├── Makefile
├── pyproject.toml
└── LICENSE
~~~

## Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài dependencies

Yêu cầu Python `3.10+`:

~~~bash
python -m pip install -r requirements.txt
~~~

### 2. Tạo và validate corpus

~~~bash
python scripts/download_data.py
python scripts/validate_catalog.py
~~~

### 3. Build artifact

~~~bash
python scripts/build_index.py \
  --data-dir data/raw \
  --model-dir artifacts \
  --catalog-path configs/knowledge_catalog.yaml
~~~

Mặc định dùng structure-aware 250/40. `--strategy sliding_window` chỉ dùng làm
baseline evaluation. Build đầy đủ cần FAISS, SentenceTransformers và model
embedding; Cross-Encoder được tải lazy khi query/evaluation.

### 4. Kiểm tra chất lượng code

~~~bash
python -m ruff check --no-cache src scripts tests
python -m ruff format --check src scripts tests
python -m pytest -q
~~~

### 5. Chạy evaluation

~~~bash
python scripts/evaluate.py
python scripts/ablation_experiments.py
~~~

Evaluation yêu cầu artifact đã build và model embedding. Threshold chỉ được
chọn từ Dev; Test dùng để báo cáo cuối. Reports là file sinh tự động và không
commit.

### 6. Chạy API

~~~bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

Development có key server-side mặc định `local-demo-employee`. Production đặt:

~~~text
ENV=production
RAG_API_KEYS_JSON={"key":{"user_id":"u1","groups":["employee","hr"]}}
~~~

API đầy đủ xem tại [docs/API.md](docs/API.md).

## API Tóm Tắt

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/health` | Trạng thái API và artifact hiện hành |
| `POST` | `/query` | Hỏi đáp, bắt buộc `X-API-Key` |
| `POST` | `/debug/retrieve` | Debug development-only, admin key |

## CI và tính tái lập

Workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) chạy Python 3.10
và 3.11 trên push, pull request hoặc manual dispatch. CI cài dependencies, tạo
corpus mẫu, validate catalog, chạy Ruff, toàn bộ unit/smoke tests và build Docker image
để smoke test `/health`. Image không được push/deploy; workflow chỉ kiểm tra
Dockerfile và API khởi động được.

## Giới Hạn Hiện Tại

- PDF scan dạng ảnh chưa có OCR; PDF số được đọc qua `pypdf`.
- TXT/MD/DOCX không có số trang ổn định nên page citation có thể là `null`.
- Ollama là integration tùy chọn; khi không sẵn sàng hệ thống trả `sources_only`.
- Citation runtime chỉ kiểm tra reference ID và sentence coverage, không chứng
  minh semantic entailment.
- Corpus demo nhỏ; headline benchmark vẫn `Pending` cho đến khi có run Dev/Test
  tái lập với model cache.

## Contract hiện tại

~~~text
package: enterprise-rag-assistant 1.0.0
artifact: artifacts/config.json + artifacts/groups/<group>/
service: FastAPI 1.0.0
query modes: answer | sources_only | abstain
~~~
