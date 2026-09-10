"""Pipeline nạp tài liệu có catalog và structure-aware chunking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..catalog import CatalogEntry, active_catalog
from ..utils import compute_file_checksum
from .chunking import chunk_text, structure_aware_chunk, validate_chunk_quality_contract
from .models import Chunk, DocumentBlock
from .parsers import SUPPORTED_SUFFIXES, check_document_quality, read_document_blocks

LOGGER = logging.getLogger("rag_knowledge_assistant.ingestion")

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Chunk",
    "DocumentBlock",
    "check_document_quality",
    "chunk_text",
    "ingest_folder",
    "read_document_blocks",
    "structure_aware_chunk",
    "validate_chunk_quality_contract",
]


def ingest_folder(
    folder: str | Path,
    chunk_words: int = 250,
    overlap_words: int = 40,
    strategy: str = "structure_aware",
    catalog: list[CatalogEntry] | None = None,
    as_of: Any | None = None,
) -> list[Chunk]:
    """Index chỉ các file ACTIVE trong catalog tại ngày build.

    Catalog là bắt buộc trong pipeline chuẩn để document identity, version
    và ACL không bị suy luận từ đường dẫn hoặc nội dung tài liệu.
    """
    data_path = Path(folder)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_path.resolve()}")
    if catalog is None:
        raise ValueError("Catalog bắt buộc cho pipeline nạp tài liệu chuẩn.")

    entries = {entry.file: entry for entry in active_catalog(catalog, as_of=as_of)}
    chunks: list[Chunk] = []
    for relative_path, entry in sorted(entries.items()):
        path = data_path / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Catalog trỏ tới file không tồn tại: {relative_path}")

        file_hash = compute_file_checksum(path)
        blocks = read_document_blocks(path)
        if path.suffix.lower() == ".pdf":
            pages = [(block.text, block.page) for block in blocks]
        else:
            lines = [
                f"# {block.text}" if block.block_type == "heading" else block.text
                for block in blocks
            ]
            pages = [("\n".join(lines), None)]
        passed, reason = check_document_quality(path, pages)
        if not passed:
            raise ValueError(f"Document QA thất bại với {relative_path}: {reason}")

        document_chunks: list[Chunk] = []
        next_chunk_index = 0
        for content, page in pages:
            if not content.strip():
                continue
            page_chunks = chunk_text(
                content,
                source=path.name,
                page=page,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                checksum=file_hash,
                strategy=strategy,
                source_path=relative_path,
                document_id=entry.document_id,
                base_chunk_index=next_chunk_index,
                policy_key=entry.policy_key,
                document_version=entry.version,
                policy_status=entry.policy_status,
                department=entry.department,
                allowed_groups=entry.allowed_groups,
            )
            document_chunks.extend(page_chunks)
            next_chunk_index += len(page_chunks)
        chunks.extend(document_chunks)

    validate_chunk_quality_contract(chunks)
    LOGGER.info("Đã tạo %d chunks từ %d tài liệu ACTIVE.", len(chunks), len(entries))
    return chunks
