# Vietnamese Enterprise Policy RAG

Trợ lý tra cứu chính sách nội bộ bằng tiếng Việt. Hệ thống chỉ đưa tài liệu
đang hiệu lực và được cấp quyền vào retrieval; chỉ trả `ANSWER` khi evidence
và citation contract đạt yêu cầu.

## Luồng xử lý chuẩn

```text
Knowledge Catalog
  -> kiểm tra version, ngày hiệu lực, ACL và coverage file
  -> Document Quality Gate
  -> Structure-aware chunks
  -> immutable release schema v3
  -> ACL shards theo group
  -> Dense Top-30 + BM25 Top-30
  -> RRF (k=60), giữ Top-20
  -> multilingual Cross-Encoder, giữ raw reranker_logit
  -> Evidence Gate (ngưỡng tune trên Dev)
  -> Ollama grounded generation
  -> kiểm tra citation ở cấp factual sentence
```

Khi truy vấn production, `X-API-Key` được ánh xạ ở phía server thành
`AccessContext`. Request body chỉ có `question`; nhóm quyền, candidate size,
reranker và ngưỡng retrieval không do client điều khiển.

## Contract an toàn

- `configs/knowledge_catalog.yaml` là nguồn sự thật duy nhất cho
  `document_id`, `policy_key`, version, ngày hiệu lực, trạng thái và
  `allowed_groups`.
- Quy tắc hiệu lực là `effective_from <= as_of < effective_to`; `effective_to`
  rỗng nghĩa là không có ngày kết thúc. Mỗi `policy_key` chỉ được có một
  version `ACTIVE`.
- ACL được lọc trước retrieval bằng shard. ACL rỗng, thiếu `AccessContext` hoặc
  không có group hợp lệ đều bị từ chối mặc định.
- Không suy luận ACL từ tên file. Metadata ACL của chunk phải đến từ catalog.
- Reranker dùng model multilingual
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` và giữ điểm thô
  `reranker_logit`; không chuyển thành xác suất bằng sigmoid.
- Không có linear fusion. Hai nhánh Dense/BM25 được hợp nhất bằng RRF theo
  thứ hạng.
- Reranker hoặc LLM lỗi trả `EVIDENCE_ONLY`; evidence không đủ trả `ABSTAIN`.
  Không tự động biến fallback thành câu trả lời LLM bình thường.
- Citation validator chỉ kiểm tra ID citation có thuộc evidence và factual
  sentence có citation hợp lệ. Nó không chứng minh semantic entailment.
- `/internal/debug/retrieve` chỉ mở ở môi trường development và cần
  `RAG_ADMIN_API_KEY`; production trả `404`.

## Immutable index release

Lệnh build luôn tạo full rebuild trong thư mục tạm, validate count của FAISS,
BM25 và chunks, sau đó mới đổi `active_index.json` bằng replace nguyên tử.
Release cũ không bị ghi đè.

```text
models/rag_index/
├── active_index.json
└── releases/<index_version>/
    ├── config.json              # schema_version = 3
    ├── index.faiss              # global bundle
    ├── chunks.json
    ├── bm25_index.json
    ├── catalog_snapshot.json
    ├── document_registry.json
    ├── document_manifest.json
    ├── index_manifest.json
    ├── quality_report.json
    └── shards/<group>/          # bundle chỉ chứa chunk của group
```

Chunk production có định danh ổn định:

```text
{document_id}:{version}:{section_hash}:c{index:03d}
```

`document_id` và version không lấy từ nội dung chunk đầu tiên. Tài liệu
`ARCHIVED`, `RETIRED`, `DRAFT` hoặc hết hiệu lực vẫn có thể nằm trong catalog
nhưng không được đưa vào production index.

## API

### `GET /health`

Trả trạng thái tổng quan, không trả đường dẫn filesystem:

```json
{
  "status": "ok",
  "index_ready": true,
  "model_version": "enterprise-rag-v1",
  "index_version": "rag-<release>",
  "chunk_count": 38
}
```

Index chỉ được coi là ready khi active release có `schema_version >= 3` và thư
mục ACL shards.

### `GET /health/live`

Liveness probe, trả `status=alive` và timestamp UTC.

### `GET /health/ready`

Readiness probe. Endpoint kiểm tra artifact bắt buộc, count FAISS/BM25/chunk và
trạng thái reranker. Thiếu artifact hoặc release sai schema trả HTTP `503`.

### `POST /v1/query`

Alias `/query` vẫn được giữ để tương thích ngược. Request production:

```http
X-API-Key: <server-configured-key>
Content-Type: application/json
```

```json
{"question": "Nhân viên chính thức có bao nhiêu ngày phép năm?"}
```

Response gồm `request_id`, `answer`, `decision`, `retrieval`, `grounding`,
`citations`, `sources` và `versions`. `sources` chỉ chứa metadata công khai như
`chunk_id`, `document_id`, `source_path`, page, section và version; score debug
không được trả cho client.

Các action hợp lệ:

- `ANSWER`: evidence gate và citation contract đều đạt.
- `ABSTAIN`: không có evidence hoặc evidence không đủ.
- `EVIDENCE_ONLY`: reranker/LLM không sẵn sàng; response chỉ hiển thị evidence
  đã truy xuất để người dùng tự kiểm tra.

### `POST /internal/debug/retrieve`

Endpoint nghiên cứu trong development. Payload nhận `question`, `top_k`,
`candidate_k`, `min_score`, `use_reranker`; endpoint này không thuộc contract
production và yêu cầu admin key.

### `POST /v1/feedback`

Ghi phản hồi vào `feedback/feedback.jsonl` với `request_id`, `helpful`, thông tin
document/section kỳ vọng và ghi chú reviewer.

## Cấu hình chính

`configs/retrieval.yaml` mô tả policy retrieval chuẩn:

| Tham số | Giá trị |
|---|---:|
| Dense candidate | 30 |
| BM25 candidate | 30 |
| RRF k | 60 |
| Rerank candidates | 20 |
| Context chunks | 4 |
| Evidence threshold | raw logit `1.72` |
| ACL mode | `pre_retrieval_shards` |

`configs/generation.yaml` mô tả policy `grounded-v1`: citation bắt buộc ở cấp
factual sentence; lỗi citation, reranker hoặc LLM đều đi theo evidence-only
semantics.

## Cài đặt và chạy

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/validate_catalog.py
python scripts/build_index.py
```

`build_index.py` cần `faiss-cpu`, SentenceTransformers và model embedding. Sau
khi build thành công, kiểm tra và đánh giá:

```bash
python -m pytest -q
python -m ruff check --no-cache src scripts tests
python -m ruff format --check src scripts tests
python scripts/evaluate_dev.py
python scripts/evaluate_final.py
python scripts/ablation_experiments.py
```

`evaluate_dev.py` tune pipeline và evidence threshold chỉ trên Dev, sau đó chạy
canonical pipeline một lần trên Locked Test. Kết quả được ghi vào
`reports/dev_metrics.json` và `reports/final_test_metrics.json`; không dùng
metric lịch sử trong repository để mô tả release mới.

Khởi động API:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Trong development có thể dùng key mặc định `local-demo-employee`. Production
phải cấu hình mapping server-side bằng một trong hai cách:

```text
RAG_API_KEYS_JSON={"key":{"user_id":"u1","groups":["employee","hr"]}}
```

hoặc `RAG_API_KEY`, `RAG_API_USER_ID`, `RAG_API_GROUPS`. Admin debug dùng
`RAG_ADMIN_API_KEY` và không được truyền quyền từ request body.

## Cấu trúc dự án

```text
configs/           Catalog và retrieval/generation policy
data/raw/          Corpus tài liệu đầu vào
data/evaluation/   Benchmark human và regression
models/            Immutable index release sinh lúc build
reports/           Report JSON sinh lúc evaluate/ablation
scripts/           CLI build, validate, evaluate và ablation
src/               Catalog, security, ingestion, index, retrieval, service, API
tests/             Unit test và enterprise contract test
```

## Giới hạn hiện tại

- PDF scan dạng ảnh chưa có OCR; PDF số được đọc qua `pypdf`.
- DOCX/TXT không có số trang ổn định nên citation page có thể là `null`.
- Ollama local là fallback vận hành, chưa phải kiến trúc HA đa vùng.
- Citation runtime mới kiểm tra liên kết tham chiếu; semantic faithfulness cần
  đánh giá offline bằng human/judge.

## Phiên bản contract

```text
package: enterprise-rag-assistant 1.0.0
model: enterprise-rag-v1
retrieval policy: retrieval-v1
generation policy: grounded-v1
index schema: 3
```
