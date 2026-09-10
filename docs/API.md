# API Reference

API được cài đặt trong `src/api.py`. README chỉ giữ bảng tóm tắt; tài liệu này
giữ payload và mã lỗi để tránh lệch với code.

## Chạy dịch vụ

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Xác thực

`POST /query` yêu cầu header `X-API-Key`. Server ánh xạ key thành
`AccessContext`; client không được gửi groups hoặc tham số retrieval trong body.

```text
RAG_API_KEYS_JSON={"key":{"user_id":"u1","groups":["employee","hr"]}}
```

Hoặc dùng cấu hình một key:

```text
RAG_API_KEY=<key>
RAG_API_USER_ID=u1
RAG_API_GROUPS=employee,hr
```

Trong development, key demo `local-demo-employee` được dùng khi chưa cấu hình
key. Production không có fallback này.

## Endpoint

### `GET /health`

Không cần xác thực. Endpoint chỉ kiểm tra artifact hiện hành có `config.json`
và các group index đầy đủ hay không; không tải embedding model.

```json
{
  "status": "degraded",
  "index_loaded": false,
  "total_chunks": 0,
  "groups": []
}
```

`status` là `ok` khi artifact có đủ `config.json` và các file
`index.faiss`, `chunks.json`, `bm25_index.json` cho mọi group.

### `POST /query`

Request chỉ có câu hỏi:

```http
POST /query
X-API-Key: local-demo-employee
Content-Type: application/json
```

```json
{"question":"Nhân viên chính thức có bao nhiêu ngày phép năm?"}
```

Schema từ chối trường thừa như `user_groups`, `min_score` hoặc `use_reranker`.
Response user-facing gồm `request_id`, `mode`, `answer`, `citations` và
`sources`:

```json
{
  "request_id": "req_abc123",
  "mode": "answer",
  "answer": "Nhân viên được hưởng 12 ngày phép năm. [C1]",
  "citations": [
    {
      "id": "C1",
      "document": "annual_leave_policy_2026.txt",
      "document_id": "...",
      "version": "v2026",
      "source_path": "annual_leave_policy_2026.txt",
      "page": null,
      "section": "Tiêu chuẩn phép năm",
      "chunk_id": "...",
      "quote": "..."
    }
  ],
  "sources": []
}
```

Các mode hợp lệ:

| Mode | Ý nghĩa |
|---|---|
| `answer` | Evidence gate đạt và citation reference/coverage hợp lệ. |
| `sources_only` | Có nguồn được phép nhưng không thể tạo câu trả lời tổng hợp an toàn. |
| `abstain` | Không có evidence hoặc evidence không đủ. |

### `POST /debug/retrieve`

Endpoint development-only, không xuất hiện trong OpenAPI schema. Endpoint trả
raw hits để debug retrieval khi `ENV` khác `production` và key trùng
`RAG_ADMIN_API_KEY`. Admin key cũng phải có trong `RAG_API_KEYS_JSON` để server
tạo AccessContext.

```json
{
  "question":"Quy định cấp quyền thiết bị là gì?",
  "top_k":4,
  "candidate_k":30,
  "min_score":null,
  "use_dense":true,
  "use_bm25":true,
  "use_reranker":true
}
```

## Mã lỗi

| HTTP | Trường hợp |
|---:|---|
| `401` | Thiếu hoặc sai `X-API-Key`. |
| `403` | Không có quyền admin cho debug endpoint. |
| `404` | Debug endpoint bị tắt trong production. |
| `422` | Payload không hợp lệ hoặc có trường ngoài contract. |
| `500` | Lỗi nội bộ khi xử lý query. |

## Kiểm tra nhanh

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H "X-API-Key: local-demo-employee" \
  -H "Content-Type: application/json" \
  -d '{"question":"Nhân viên chính thức có bao nhiêu ngày phép năm?"}'
```
