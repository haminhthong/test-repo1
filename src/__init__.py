"""Vietnamese Evidence-Grounded Knowledge Assistant — Hybrid RAG Platform.

Gói thư viện chính của dự án. Cung cấp:
- Cấu hình hệ thống (`config`).
- Pipeline nạp tài liệu & phân đoạn theo cấu trúc (`ingestion`).
- Chỉ mục FAISS + BM25 (`index`, `ranking`).
- Bộ truy xuất lai canonical (`retrieval`).
- Bộ sinh câu trả lời căn thực & kiểm định trích dẫn (`generation`).
- Dịch vụ REST API (`api`).
- Đánh giá benchmark đa tầng (`evaluate`).
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = [
    "config",
    "ingestion",
    "ranking",
    "retrieval",
    "generation",
    "api",
    "evaluate",
]
