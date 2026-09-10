"""Vietnamese Policy RAG — Dense/BM25 retrieval có ACL và citation.

Gói thư viện chính của dự án. Cung cấp:
- Cấu hình hệ thống (`config`).
- Pipeline nạp tài liệu & phân đoạn theo cấu trúc (`ingestion`).
- Artifact FAISS + BM25 (`index`, `ranking`).
- Bộ truy xuất hybrid có ACL (`retrieval`).
- Bộ sinh câu trả lời có citation (`generation`).
- Dịch vụ REST API (`api`).
- Đánh giá benchmark đa tầng (`evaluate`).
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "api",
    "config",
    "evaluate",
    "generation",
    "ingestion",
    "ranking",
    "retrieval",
]
