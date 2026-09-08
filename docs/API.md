# API Reference

Tài liệu này mô tả REST API đang được cài đặt trong `src/api.py`. README chỉ giữ
bảng tóm tắt; các chi tiết về payload, xác thực và mã lỗi được tập trung tại đây
để tránh lệch giữa tài liệu và code.

## Chạy dịch vụ

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Mặc định API chạy ở `http://127.0.0.1:8000`. OpenAPI UI có tại `/docs` khi
FastAPI được chạy ở môi trường development.

## Xác thực và nguyên tắc an toàn

- Endpoint production `/v1/query` yêu cầu header `X-API-Key`.
- Server ánh xạ API key thành `AccessContext`; client không được truyền nhóm
  quyền, `k`, ngưỡng điểm hoặc cờ reranker trong request production.
- API key có thể được cấu hình bằng JSON mapping:

  ```text
  RAG_API_KEYS_JSON={"key":{"user_id":"u1","groups":["employee","hr"]}}
  ```

- Hoặc dùng cấu hình đơn:

  ```text
  RAG_API_KEY=<key>
  RAG_API_USER_ID=u1
  RAG_API_GROUPS=employee,hr
  ```

- Trong development, nếu chưa cấu hình key, code cho phép key demo
  `local-demo-employee`. Production không có fallback này.
- `/internal/debug/retrieve` cần `RAG_ADMIN_API_KEY` và tự động trả `404` khi
  `ENV=production`.
- Không đưa điểm retrieval/reranker hoặc đường dẫn filesystem vào response
  production.

## Endpoint production

### `GET /health`

Health tổng quan, không cần xác thực. Response gồm trạng thái an toàn để dùng
cho kiểm tra cơ bản:

```json
{
  "status": "ok",
  "index_ready": true,
  "model_version": "enterprise-rag-v1",
  "index_version": "rag-<release>",
  "chunk_count": 38
}
```

`status` là `degraded` nếu active release chưa có đủ artifact hoặc chưa đạt
schema index yêu cầu.

### `GET /health/live`

Liveness probe cho container/orchestrator. Endpoint chỉ xác nhận tiến trình API
đang chạy:

```json
{
  "status": "alive",
  "timestamp": "2026-09-08T00:00:00+00:00"
}
```

### `GET /health/ready`

Readiness probe chuyên sâu. Code kiểm tra `config.json`, `index.faiss`,
`chunks.json`, `bm25_index.json`, schema release, ACL shards và count FAISS so
với số chunk. Response thành công có các trường:

```json
{
  "status": "ready",
  "model_version": "enterprise-rag-v1",
  "index_version": "rag-<release>",
  "vector_dimension": 384,
  "total_chunks": 38,
  "reranker_mode": "neural",
  "reranker_ready": true
}
```

Thiếu artifact, sai schema hoặc count không nhất quán trả HTTP `503`.

### `POST /v1/query`

Đây là endpoint hỏi đáp canonical. Alias `/query` vẫn tồn tại để tương thích
ngược nhưng không hiển thị trong OpenAPI schema.

Request:

```http
POST /v1/query
X-API-Key: local-demo-employee
Content-Type: application/json
```

```json
{
  "question": "Nhân viên chính thức có bao nhiêu ngày phép năm?"
}
```

`question` dài từ 2 đến 2.000 ký tự. Trường dư bị bỏ qua bởi schema input;
`user_groups`, `candidate_k`, `min_score`, `use_reranker` và các tham số pipeline
không thuộc contract production.

Response chuẩn hóa gồm:

- `request_id`: mã truy vết request;
- `answer`: câu trả lời grounded hoặc thông báo từ chối/fallback;
- `decision`: action, answerability và reason;
- `retrieval`: thống kê retrieval và chế độ reranker;
- `grounding`: evidence score, trạng thái gate và citation validity;
- `citations`: danh sách citation có `id`, document, version, section, page,
  `chunk_id` và quote;
- `sources`: metadata nguồn đã được ACL duyệt, không có score debug;
- `versions`: model, index và generation/retrieval metadata;
- các field tương thích `model_version`, `index_version`,
  `evidence_gate_passed`.

Ba action hợp lệ:

| Action | Ý nghĩa |
|---|---|
| `ANSWER` | Evidence gate đạt và citation contract hợp lệ. |
| `ABSTAIN` | Không có evidence hoặc evidence không đủ để trả lời. |
| `EVIDENCE_ONLY` | Reranker/LLM/citation validation không sẵn sàng; chỉ trả evidence đã được phép. |

Ví dụ response rút gọn:

```json
{
  "request_id": "req_abc123",
  "answer": "Nhân viên chính thức được hưởng ... [C1]",
  "decision": {"action": "ANSWER", "answerable": true, "reason": "evidence_sufficient"},
  "retrieval": {"candidate_count": 20, "reranker_mode": "neural"},
  "grounding": {"evidence_gate_passed": true, "citation_valid": true},
  "citations": [
    {
      "id": "C1",
      "document": "annual_leave_policy.txt",
      "document_id": "...",
      "version": "2025.1",
      "source_path": "annual_leave_policy.txt",
      "page": null,
      "section": "Phép năm",
      "chunk_id": "...",
      "quote": "..."
    }
  ],
  "sources": [],
  "versions": {"model_version": "enterprise-rag-v1", "index": "rag-<release>"},
  "model_version": "enterprise-rag-v1",
  "index_version": "rag-<release>",
  "evidence_gate_passed": true
}
```

## Endpoint nội bộ

### `POST /internal/debug/retrieve`

Endpoint nghiên cứu retrieval, không thuộc API production. Chỉ khả dụng khi
`ENV` khác `production` và header `X-API-Key` trùng `RAG_ADMIN_API_KEY`.

Request:

```json
{
  "question": "Quy định cấp quyền thiết bị là gì?",
  "top_k": 4,
  "candidate_k": 30,
  "min_score": null,
  "use_reranker": true
}
```

Response trả `question`, `count`, `hits` và `reranker_mode`. `hits` có thể chứa
điểm và trường debug để phân tích; không dùng response này làm contract cho
client production. Debug endpoint truyền nhóm kiểm thử cố định trong code và
không đọc quyền từ request body.

### `POST /v1/feedback`

Ghi feedback phục vụ vòng đánh giá liên tục. Endpoint yêu cầu `X-API-Key` hợp
lệ và nhận:

```json
{
  "request_id": "req_abc123",
  "helpful": true,
  "expected_document": "annual_leave_policy.txt",
  "expected_section": "Phép năm",
  "reviewer_note": "Citation phù hợp"
}
```

Code bổ sung `received_at` theo UTC và append một JSON object vào
`feedback/feedback.jsonl`. Thư mục feedback được tạo khi có request đầu tiên.

## Mã lỗi

| HTTP | Trường hợp |
|---:|---|
| `401` | Thiếu hoặc sai `X-API-Key` ở `/v1/query` hoặc `/v1/feedback`. |
| `403` | Không có quyền admin cho debug endpoint. |
| `404` | Debug endpoint bị ẩn trong production. |
| `422` | Payload không hợp lệ theo schema FastAPI/Pydantic. |
| `500` | Lỗi nội bộ khi xử lý query hoặc ghi feedback. |
| `503` | Active index chưa sẵn sàng hoặc artifact không nhất quán. |

## Kiểm tra nhanh

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "X-API-Key: local-demo-employee" \
  -H "Content-Type: application/json" \
  -d '{"question":"Nhân viên chính thức có bao nhiêu ngày phép năm?"}'
```
