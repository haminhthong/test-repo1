# Vietnamese Enterprise Policy RAG

[![CI](https://github.com/haminhthong/Enterprise-Rag-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/Enterprise-Rag-Assistant/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.11.0-005571.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-rank__bm25-6f42c1.svg)](https://github.com/dorianbrown/rank_bm25)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-5.1.0-orange.svg)](https://www.sbert.net/)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64.svg)](https://docs.astral.sh/ruff/)
[![Pytest](https://img.shields.io/badge/test-Pytest-0A9EDC.svg)](https://pytest.org/)

Trợ lý RAG tra cứu chính sách nội bộ bằng tiếng Việt. Hệ thống xử lý tài liệu
theo version và ngày hiệu lực, kiểm soát ACL trước retrieval, kết hợp Dense với
BM25, rerank bằng Cross-Encoder đa ngôn ngữ và chỉ tạo câu trả lời khi evidence
đạt contract an toàn.

## Bài Toán & Phạm Vi Ứng Dụng (Problem & Scope)

### Bài toán

Kho tài liệu nội bộ thường có các rủi ro mà một RAG tutorial đơn giản không xử
lý đầy đủ:

- Tài liệu cùng một policy có nhiều version, trong đó version cũ vẫn có thể
  được tìm thấy dù đã hết hiệu lực.
- Người dùng khác nhau có quyền đọc khác nhau; lọc ACL sau retrieval có thể
  làm lộ nội dung nhạy cảm qua candidate, score hoặc prompt.
- Truy vấn có thể dùng từ khóa chính xác, cách diễn đạt tương đương hoặc chứa
  mã/số liệu; chỉ Dense hoặc chỉ BM25 đều có vùng yếu.
- LLM có thể trả lời tự tin khi evidence thiếu, sai citation hoặc trộn thông
  tin từ tài liệu không còn active.

### Phạm vi hiện tại

Dự án triển khai một pipeline có thể kiểm tra và tái lập cho corpus policy mẫu
tiếng Việt:

- Catalog là nguồn sự thật cho `document_id`, `policy_key`, version, effective
  dates, trạng thái và `allowed_groups`.
- Chỉ tài liệu `ACTIVE` và còn hiệu lực tại ngày build được đưa vào production
  index.
- ACL được áp dụng ở shard trước retrieval; identity đến từ server-side API
  key.
- Kết quả cuối có thể là `ANSWER`, `ABSTAIN` hoặc `EVIDENCE_ONLY`.
- Mỗi câu factual trong câu trả lời tổng hợp phải có citation hợp lệ tới
  evidence đã truy xuất.

Ngoài phạm vi release này: OCR cho PDF scan, semantic entailment hoàn toàn tự
động, HA đa vùng và cơ chế cập nhật từng vector trong index production.

## Kiến Trúc Kỹ Thuật Duy Nhất (Canonical Technical Architecture)

Sơ đồ Mermaid dưới đây là quy trình kỹ thuật duy nhất chi phối mã nguồn, cấu
hình và báo cáo. Tên bước trong sơ đồ tương ứng với `src/catalog.py`,
`src/ingestion.py`, `src/index.py`, `src/retrieval.py`, `src/ranking.py`,
`src/generation.py`, `src/service.py` và `src/api.py`.

~~~mermaid
flowchart TD
    subgraph OFFLINE["OFFLINE INDEX PIPELINE"]
        D["data/raw policy documents"] --> C["Catalog validation"]
        M["knowledge_catalog.yaml"] --> C
        C --> Q["Document QA"]
        Q --> K["Structure-aware chunking"]
        K --> I["Dense + BM25 index"]
        I --> S["ACL shards"]
        S --> V["Release validation"]
        V --> P["Atomic promotion"]
        P --> R["models/rag_index/releases/<version>"]
    end

    subgraph ONLINE["ONLINE QUERY PIPELINE"]
        U["User + X-API-Key"] --> A["Server-side AccessContext"]
        A --> AR["ACL-filtered retrieval"]
        AR --> H["Dense + BM25"]
        H --> F["RRF"]
        F --> X["Multilingual Cross-Encoder"]
        X --> G{"Evidence Gate"}
        G -->|insufficient| AB["ABSTAIN"]
        G -->|sufficient| L["Grounded LLM"]
        L --> CV["Citation validator"]
        CV -->|valid| AN["ANSWER"]
        CV -->|invalid| EO["EVIDENCE_ONLY"]
        X -. unavailable .-> EO
        L -. error .-> EO
        AB --> OUT["REST response"]
        AN --> OUT
        EO --> OUT
    end

    R -. active release .-> AR
    E["data/evaluation/questions.json"] --> T["Dev tuning + locked Test"]
    T --> J["reports/*.json"]
    OUT --> FB["feedback/events.jsonl"]
~~~

Ý nghĩa vận hành:

- Offline luôn là full rebuild vào release mới. `active_index.json` chỉ được
  thay sau khi toàn bộ artifact và ACL shard đã validate.
- Online lấy identity từ API key do server quản lý; request body production chỉ
  chứa câu hỏi.
- Dense lấy Top-30, BM25 lấy Top-30. Hai nhánh được hợp nhất bằng RRF
  (`rrf_k=60`), sau đó giữ tối đa Top-20 để rerank.
- Cross-Encoder giữ raw `reranker_logit`; điểm này là tín hiệu xếp hạng, không
  phải xác suất.
- Evidence Gate được tune trên Dev. Evidence không đủ đi thẳng đến `ABSTAIN`.
- Reranker/LLM không sẵn sàng hoặc citation validation thất bại thì không tạo
  câu trả lời tổng hợp bình thường; hệ thống trả `EVIDENCE_ONLY`.

## Luồng dữ liệu và artifact

| Dữ liệu đầu vào | Thành phần xử lý | Artifact hoặc kết quả | Nơi sử dụng tiếp |
|---|---|---|---|
| Tài liệu trong `data/raw` và catalog | `catalog.py` + `ingestion.py` | Catalog hợp lệ, tài liệu active, chunks có metadata | `index.py` |
| Chunks active và ACL groups | `index.py` | Release immutable gồm FAISS, BM25, manifest và ACL shards | `retrieval.py` |
| API key server-side + câu hỏi | `security.py` + `api.py` | `AccessContext` và query nội bộ đã xác thực | `service.py` |
| Query + `AccessContext` | `retrieval.py` + `ranking.py` | Candidate Dense/BM25, RRF, reranked evidence | Evidence Gate |
| Evidence đã qua gate | `generation.py` + Ollama tùy chọn | `ANSWER`, `ABSTAIN` hoặc `EVIDENCE_ONLY` kèm citations | REST response |
| Benchmark Dev/Locked Test | `evaluate.py` | `reports/*.json`, không commit | Tuning và audit |
| Metadata telemetry không chứa nội dung thô | `service.py` | `feedback/events.jsonl` | Theo dõi vận hành |

Raw corpus mẫu được tạo bằng `scripts/download_data.py` và bị ignore khỏi Git;
CI luôn tái tạo corpus trước khi validate để một checkout mới không phụ thuộc
file cục bộ của máy phát triển.

## Benchmark & Evaluation Status

Bảng dưới đây là format benchmark canonical mà pipeline đánh giá sinh ra. Repo
chưa có một locked-test run tái lập thành công với đầy đủ runtime ML tại môi
trường hiện tại, vì vậy không điền số ước đoán.

> Current release status: evaluation pipeline implemented; headline benchmark pending reproducible locked-test run.

| Pipeline | Recall@K | MRR | nDCG@K | Citation validity | Answer coverage |
|---|---:|---:|---:|---:|---:|
| BM25 | Pending | Pending | Pending | Pending | Pending |
| Dense | Pending | Pending | Pending | Pending | Pending |
| Hybrid RRF | Pending | Pending | Pending | Pending | Pending |
| + Cross Encoder | Pending | Pending | Pending | Pending | Pending |

Khi đủ dependencies, `scripts/evaluate_dev.py` sẽ:

1. đọc benchmark từ `data/evaluation/questions.json`;
2. tune evidence threshold và lựa chọn pipeline chỉ trên `dev`;
3. chạy canonical pipeline đúng một lần trên `test` đã khóa;
4. ghi `reports/dev_metrics.json` và `reports/final_test_metrics.json`.

`Recall@K`, `MRR`, `nDCG@K`, true abstention, false answer và evidence metrics
được tính từ code đánh giá; nDCG được kiểm tra trong khoảng `[0, 1]`.
`scripts/ablation_experiments.py` chỉ phục vụ so sánh trên Dev, không dùng
locked test để chọn cấu hình.

Evaluator hiện sinh trực tiếp retrieval/evidence/gate metrics. Citation validity
ở runtime được kiểm tra bởi `validate_citation_references`; hàm
`evaluate_citation_metrics` dùng khi có generation harness cung cấp answer, còn
headline citation/answer coverage vẫn để `Pending` nếu chưa có locked generation
run. README không suy diễn các chỉ số này từ report cũ.

## Corpus / Data Snapshot

Snapshot dưới đây được kiểm tra trực tiếp từ catalog, raw corpus và benchmark
đang nằm trong repository; đây là quy mô dữ liệu mẫu, không phải headline
benchmark.

~~~yaml
Documents: 18 catalog records; 17 ACTIVE tại 2026-09-08
Pages: null với TXT/MD/DOCX; PDF số có page metadata khi parser đọc được
Chunks: 38 active chunks với cấu hình smoke structure-aware 250/40
Policies: 17 active policy_key
Languages: Vietnamese (vi)
Evaluation queries: 50 human-authored (12 dev, 38 test) + 4 synthetic regression (2 dev, 2 test)
~~~

~~~text
Corpus
├── Vietnamese internal policy documents
├── versioned policy_key
├── effective dates
└── ACL groups
~~~

Raw corpus mẫu được tạo bởi `scripts/download_data.py`, gồm các nhóm HR,
Finance và Security. `annual_leave_policy_2025.txt` là hard negative đã hết
hiệu lực; version hiện hành phải được chọn thông qua catalog và `as_of` date,
không chọn theo tên file.

## CI và tính tái lập

Workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) chạy trên Python
3.10 và 3.11 cho mỗi push hoặc pull request. CI thực hiện cùng một chuỗi kiểm
tra tối thiểu của repo:

1. cài dependencies từ `requirements.txt`;
2. tạo lại corpus mẫu bằng `scripts/download_data.py`;
3. validate catalog và coverage file;
4. chạy Ruff lint, Ruff format check và toàn bộ pytest.

CI chưa tự tải model embedding/reranker để build release production và chưa
điền số benchmark. Việc đó cần một locked-test run có model cache và được mô
tả rõ là `Pending` ở phần benchmark, không dùng số giả.

## Offline Index Pipeline

### 1. Catalog validation

`configs/knowledge_catalog.yaml` khai báo coverage cho từng tài liệu. Loader
kiểm tra:

- file phải nằm trong `data/raw`, không được là absolute path hoặc path
  traversal;
- `document_id`, `policy_key`, version, department và nhóm ACL phải hợp lệ;
- `effective_from <= as_of < effective_to`; `effective_to` rỗng nghĩa là chưa
  có ngày kết thúc;
- một `policy_key` không được có nhiều version `ACTIVE` cùng thời điểm;
- catalog không được suy luận ACL từ tên file.

Lệnh kiểm tra độc lập:

~~~bash
python scripts/validate_catalog.py
~~~

### 2. Document QA và chunking

`src/ingestion.py` hỗ trợ TXT, Markdown, PDF số qua `pypdf` và DOCX qua
`python-docx`. Parser giữ section, page khi có và metadata lấy từ catalog.
Document Quality Gate loại nội dung rỗng, chunk quá ngắn, thiếu metadata hoặc
chunk không thuộc tài liệu active.

Mặc định chunk theo `structure_aware` với 250 từ và overlap 40 từ. Chunk ID
ổn định theo dạng:

~~~text
{document_id}:{version}:{section_hash}:c{index:03d}
~~~

### 3. Build release

`src/index.py` tạo:

- global FAISS bundle cho Dense retrieval;
- global BM25 bundle;
- shard FAISS/BM25 cho từng nhóm ACL;
- catalog snapshot, document registry, manifest và quality report;
- `config.json` với `schema_version=3`.

Release được validate count giữa FAISS, BM25 và `chunks.json`. Sau đó
`active_index.json` được thay bằng `os.replace`, nên release cũ vẫn còn để
audit/rollback thủ công. Cấu trúc release:

~~~text
models/rag_index/
├── active_index.json
└── releases/<index_version>/
    ├── config.json
    ├── index.faiss
    ├── chunks.json
    ├── bm25_index.json
    ├── catalog_snapshot.json
    ├── document_registry.json
    ├── document_manifest.json
    ├── index_manifest.json
    ├── quality_report.json
    └── shards/<group>/
        ├── index.faiss
        ├── chunks.json
        └── bm25_index.json
~~~

Các artifact trong `models/` bị ignore khỏi Git và phải được tạo lại từ
catalog/corpus khi cần tái lập.

## Online Query Pipeline

### Retrieval và ranking

`src/retrieval.py` chọn shard theo `AccessContext.groups`, sau đó chỉ truy vấn
các chunk đã được ACL cho phép. Pipeline production dùng:

| Bước | Cấu hình |
|---|---:|
| Dense candidate pool | 30 |
| BM25 candidate pool | 30 |
| RRF constant | 60 |
| Rerank candidates | 20 |
| Context chunks | 4 |
| Evidence threshold | raw logit `1.72` mặc định, tune trên Dev |
| ACL mode | `pre_retrieval_shards` |

`src/ranking.py` dùng BM25, RRF và model
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Baseline BM25-only/Dense-only
chỉ dành cho evaluation; production không dùng linear fusion.

### Decision và grounded generation

`src/service.py` điều phối state machine:

~~~text
retrieval rỗng hoặc evidence không đủ
    -> ABSTAIN

evidence đủ
    -> grounded prompt XML chỉ chứa evidence đã truy xuất
    -> LLM trả câu trả lời có [C1], [C2], ...
    -> citation validator kiểm tra ID và coverage từng factual sentence
    -> ANSWER nếu hợp lệ
    -> EVIDENCE_ONLY nếu LLM/reranker/citation không sẵn sàng
~~~

`src/generation.py` không tuyên bố semantic entailment. Citation validator chỉ
kiểm tra citation ID có thuộc evidence và câu factual có citation hợp lệ. Đây
là guard cấu trúc runtime; faithfulness sâu hơn cần đánh giá offline bởi
human/judge.

## Ví dụ RAG hoàn chỉnh

Ví dụ dưới đây mô tả đúng các trạng thái mà code phải đi qua. Nội dung quote
minh họa lấy từ corpus mẫu; khi chạy thật, `chunk_id`, version và citation
được sinh từ release active.

~~~text
Question
  "Nhân viên chính thức có bao nhiêu ngày phép năm?"

Retrieved chunks
  - policy_leave.txt, ACTIVE, allowed_groups=[employee, hr, ...]
  - annual_leave_policy_2026.txt, ACTIVE, effective_from=2026-01-01
  - annual_leave_policy_2025.txt, ARCHIVED, bị loại trước retrieval

Reranked evidence
  - C1: policy_leave.txt / mục "Quyền lợi nghỉ phép"
  - C2: annual_leave_policy_2026.txt / mục "Tiêu chuẩn phép năm 2026"

Decision
  - AccessContext hợp lệ
  - evidence gate đạt
  - citation contract được áp dụng

Answer
  "Nhân viên chính thức được hưởng 12 ngày phép năm có lương từ năm 2026. [C1] [C2]"

Citations
  - [C1] trỏ về chunk active của policy_leave.txt
  - [C2] trỏ về chunk active của annual_leave_policy_2026.txt
~~~

Nếu không có chunk được phép, nếu evidence gate không đạt, hoặc nếu reranker/
LLM/citation validator lỗi, action tương ứng là `ABSTAIN` hoặc
`EVIDENCE_ONLY`, không được tự suy diễn thành `ANSWER`.

## Why This Is Enterprise RAG

- Effective-date filtering: không trộn policy cũ với policy đang hiệu lực.
- ACL before retrieval: không đưa chunk ngoài quyền vào candidate hoặc prompt.
- Hybrid retrieval: Dense xử lý ngữ nghĩa, BM25 giữ từ khóa/mã/số liệu, RRF
  hợp nhất theo thứ hạng.
- Abstention: evidence không đủ thì từ chối thay vì đoán.
- Immutable index release: build, validate và promote tách biệt; release đã
  phát hành không bị ghi đè tại chỗ.

## Contract An Toàn

Các invariant bắt buộc của release:

- ACL được kiểm tra trước retrieval.
- Chỉ document active và còn hiệu lực được đưa vào index production.
- Evidence không đủ phải trả `ABSTAIN`.
- `reranker_logit` là điểm xếp hạng, không phải probability.
- Citation validator không đồng nghĩa với semantic entailment.

Chi tiết API, header, payload và mã lỗi xem tại
[`docs/API.md`](docs/API.md). Kết quả đánh giá được sinh thành JSON trong
`reports/`; README là tài liệu canonical để tránh duy trì thêm một báo cáo
Markdown trùng lặp.

## Cấu Trúc Thư Mục Dự Án (Project Structure)

~~~text
Rag-Knowledge-Assistant/
├── configs/
│   ├── knowledge_catalog.yaml    # Nguồn sự thật document/version/ACL
│   ├── retrieval.yaml            # Dense, BM25, RRF, reranker, gate
│   └── generation.yaml           # Grounded generation/citation policy
├── data/
│   ├── raw/                      # Corpus TXT/MD/PDF/DOCX đầu vào
│   └── evaluation/               # Human benchmark và synthetic regression
├── docs/
│   └── API.md                    # Tài liệu REST API chi tiết
├── .github/
│   └── workflows/ci.yml          # Lint, format, catalog validation và test
├── models/
│   └── rag_index/                # Release sinh lúc build, không commit
├── reports/                      # JSON sinh lúc evaluate/ablation, không commit
├── feedback/                     # Telemetry runtime, không commit
├── scripts/
│   ├── download_data.py          # Tạo corpus mẫu
│   ├── validate_catalog.py       # Kiểm tra catalog/coverage
│   ├── build_index.py            # Build release schema v3
│   ├── activate_index.py         # Kiểm tra active pointer
│   ├── evaluate_dev.py           # Tune và đánh giá theo Dev/Test
│   ├── evaluate_final.py         # In locked-test artifact
│   └── ablation_experiments.py   # So sánh pipeline/chunk trên Dev
├── src/
│   ├── catalog.py                # Catalog và effective-date filter
│   ├── security.py               # API key -> AccessContext
│   ├── ingestion.py              # Reader, QA, chunking
│   ├── index.py                  # Immutable release và ACL shards
│   ├── ranking.py                # BM25, RRF, Cross-Encoder
│   ├── retrieval.py              # ACL-filtered retrieval
│   ├── generation.py             # Grounded prompt/citation validation
│   ├── service.py                # Decision state machine
│   ├── api.py                    # FastAPI endpoints
│   └── evaluate.py               # Metric, Dev tuning, locked test
├── tests/                        # Unit test và enterprise contract test
├── .env.example                  # Biến môi trường mẫu
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
~~~

## Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài dependencies

Yêu cầu Python `3.10+`:

~~~bash
python -m pip install -r requirements.txt
~~~

`faiss-cpu`, SentenceTransformers và model embedding/reranker cần có sẵn để
build hoặc chạy retrieval đầy đủ. Model có thể được tải ở lần chạy đầu tùy
cache môi trường.

### 2. Tạo và validate corpus

Nếu muốn tái tạo corpus mẫu từ đầu:

~~~bash
python scripts/download_data.py
python scripts/validate_catalog.py
~~~

### 3. Build immutable index

~~~bash
python scripts/build_index.py \
  --data-dir data/raw \
  --model-dir models/rag_index \
  --catalog-path configs/knowledge_catalog.yaml
~~~

Lệnh build dùng `structure_aware`, 250/40 và reranker mặc định. Để build
baseline phục vụ evaluation có thể thêm `--no-reranker`; không dùng baseline đó
làm production release canonical.

Sau build, kiểm tra active pointer và release vừa được promote:

~~~bash
python scripts/activate_index.py
~~~

`build_index.py` đã promote atomically sau khi validate. `activate_index.py`
chỉ đọc và in active pointer; script này không tự ý đổi release.

### 4. Chạy kiểm tra chất lượng code

~~~bash
python -m pytest -q
python -m ruff check --no-cache src scripts tests
python -m ruff format --check src scripts tests
~~~

### 5. Chạy evaluation

~~~bash
python scripts/evaluate_dev.py
python scripts/evaluate_final.py
python scripts/ablation_experiments.py
~~~

`evaluate_dev.py` tune threshold chỉ trên Dev rồi đánh giá Locked Test theo
policy đã khóa. `evaluate_final.py` chỉ đọc artifact final và dừng nếu artifact
ghi nhận đã tune trên test. Ablation chunk có thể tạo scratch release trong
`models/ablation_scratch/`, không phải active production release.

### 6. Chạy API

~~~bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

Development có key server-side mặc định `local-demo-employee` nếu chưa cấu hình
key. Production phải đặt `ENV=production` và cấu hình mapping thật:

~~~text
RAG_API_KEYS_JSON={"key":{"user_id":"u1","groups":["employee","hr"]}}
~~~

Hoặc:

~~~text
RAG_API_KEY=<key>
RAG_API_USER_ID=u1
RAG_API_GROUPS=employee,hr
RAG_ADMIN_API_KEY=<admin-key>
~~~

API đầy đủ và mã lỗi được mô tả tại [`docs/API.md`](docs/API.md).

## API Tóm Tắt

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/health` | Health tổng quan, response đã loại filesystem path |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Kiểm tra artifact, schema, count và reranker |
| `POST` | `/v1/query` | Endpoint hỏi đáp canonical, cần `X-API-Key` |
| `POST` | `/internal/debug/retrieve` | Debug retrieval, admin-only, non-production |
| `POST` | `/v1/feedback` | Ghi feedback JSONL |

Chi tiết request/response xem [docs/API.md](docs/API.md).

## Giới Hạn Hiện Tại

- PDF scan dạng ảnh chưa có OCR; PDF số được đọc qua `pypdf`.
- TXT/MD/DOCX không có số trang ổn định nên page citation có thể là `null`.
- Ollama local là integration tùy chọn, chưa phải kiến trúc HA đa vùng.
- Citation runtime kiểm tra reference ID và sentence coverage, không tự chứng minh
  semantic entailment.
- Artifact flat cũ không có ACL shards không được truy vấn; cần build lại
  release schema v3 trước khi dùng production.

## Phiên Bản Contract

~~~text
package: enterprise-rag-assistant 1.0.0
service: FastAPI 3.0.0
model: enterprise-rag-v1
retrieval policy: retrieval-v1
generation policy: grounded-v1
index schema: 3
~~~
