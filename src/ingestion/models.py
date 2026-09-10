"""Mô hình dữ liệu dùng chung cho parser và chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Một đoạn văn bản có định danh và metadata lấy từ catalog."""

    chunk_id: str
    text: str
    source: str
    document_id: str
    document_version: str
    source_path: str
    page: int | None = None
    checksum: str | None = None
    word_count: int = 0
    section: str | None = None
    chunk_index: int = 0
    content_hash: str = ""
    policy_key: str = ""
    policy_status: str = "active"
    department: str = ""
    allowed_groups: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Kiểm tra identity bắt buộc và bổ sung metadata dẫn xuất."""
        if not self.chunk_id:
            raise ValueError("chunk_id không được rỗng.")
        if not self.document_id:
            raise ValueError("document_id bắt buộc; identity phải đến từ catalog.")
        if not self.document_version:
            raise ValueError("document_version bắt buộc; chunk phải thuộc một version của catalog.")
        if not self.source_path:
            raise ValueError("source_path không được rỗng.")
        self.text = self.text.strip()
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.content_hash and self.text:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        self.allowed_groups = [
            group.strip().lower() for group in self.allowed_groups if group.strip()
        ]


@dataclass(frozen=True)
class DocumentBlock:
    """Khối nguyên tử từ parser: heading, paragraph hoặc table row."""

    text: str
    page: int | None = None
    block_type: str = "paragraph"
    heading_level: int | None = None
