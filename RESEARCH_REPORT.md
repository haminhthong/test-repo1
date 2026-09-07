# Báo cáo kỹ thuật — Vietnamese Enterprise Policy RAG

## Trạng thái số liệu

Các số liệu chỉ được công bố sau khi chạy `scripts/evaluate_dev.py`. Script ghi
`reports/dev_metrics.json` và `reports/final_test_metrics.json`; Markdown này
không hard-code metric cũ và không sử dụng locked test để chọn cấu hình.

## Contract V1

```text
Approved documents
  -> Knowledge Catalog (document_id, policy_key, version, effective dates, ACL)
  -> Document Quality Gate
  -> Structure-aware chunks
  -> ACTIVE + effective-date filter
  -> ACL-specific FAISS/BM25 shards
  -> Dense Top-30 + BM25 Top-30
  -> RRF(k=60), top-20
  -> Multilingual reranker, raw reranker_logit
  -> Dev-calibrated evidence gate
  -> grounded Ollama generation
  -> factual-sentence citation validation
```

## Quyết định reliability

- Catalog thiếu tài liệu, version active trùng hoặc ACL rỗng: build thất bại.
- Tài liệu hết hiệu lực không đi vào production index.
- Người dùng chỉ tìm trên shard mà AccessContext server cấp quyền.
- Reranker/LLM lỗi: `EVIDENCE_ONLY`, không tạo câu trả lời LLM bình thường.
- Citation ID sai hoặc factual sentence thiếu citation: chặn câu trả lời tổng hợp.
- Runtime chỉ kiểm tra cấu trúc citation; semantic faithfulness phải đánh giá
  offline bằng human/judge.

## Đánh giá

Tách rõ:

- Dev: chọn chunk strategy, candidate/rerank config và evidence threshold.
- Locked test: chạy canonical pipeline đã đóng băng đúng một lần.
- Security: đo Unauthorized Evidence Rate, unauthorized answer, prompt
  injection, system prompt leakage và fake secret propagation.

Các report JSON là artifact sinh ra, không phải nội dung viết tay. Nếu chưa có
artifact mới, không được dùng các con số trong report lịch sử để mô tả hệ thống.
