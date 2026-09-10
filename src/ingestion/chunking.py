"""Structure-aware chunking và các kiểm tra chất lượng chunk."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import Chunk
from .parsers import HEADING_REGEX

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9])", flags=re.UNICODE)


def _section_hash(section: str) -> str:
    """Tạo mã section ổn định, không phụ thuộc số trang."""
    normalized = " ".join(section.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:6]


def _chunk_id(document_id: str, version: str, section: str, index: int) -> str:
    """Định danh chunk duy nhất trong một document version."""
    return f"{document_id}:{version}:{_section_hash(section)}:c{index:03d}"


def split_into_sentences(text: str) -> list[str]:
    """Tách câu ở các ranh giới cơ bản, giữ nguyên câu tiếng Việt."""
    sentences = [part.strip() for part in SENTENCE_SPLIT_REGEX.split(text.strip()) if part.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _bounded_sentences(sentences: list[str], target_words: int) -> list[str]:
    """Chia câu bất thường quá dài để chunk không phình vô hạn."""
    bounded: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= target_words:
            bounded.append(sentence)
            continue
        bounded.extend(
            " ".join(words[start : start + target_words])
            for start in range(0, len(words), target_words)
        )
    return bounded


def validate_chunk_quality_contract(chunks: list[Chunk]) -> dict[str, Any]:
    """Kiểm tra chunk rỗng, ID trùng và nội dung trùng trước khi lập chỉ mục."""
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    content_hashes = [chunk.content_hash for chunk in chunks]
    report = {
        "total_chunks": len(chunks),
        "empty_chunks": sum(not chunk.text.strip() for chunk in chunks),
        "duplicate_chunk_ids": len(chunk_ids) - len(set(chunk_ids)),
        "duplicate_content": len(content_hashes) - len(set(content_hashes)),
    }
    report["passed"] = report["empty_chunks"] == 0 and report["duplicate_chunk_ids"] == 0
    if not report["passed"]:
        raise ValueError(f"Kiểm tra chất lượng chunk thất bại: {report}")
    return report


def _make_chunk(
    text: str,
    source: str,
    source_path: str,
    document_id: str | None,
    document_version: str | None,
    section: str,
    page: int | None,
    checksum: str | None,
    index: int,
    policy_key: str,
    policy_status: str,
    department: str,
    allowed_groups: list[str] | tuple[str, ...] | None,
) -> Chunk:
    """Tạo chunk với identity bắt buộc từ catalog."""
    if not document_id or not document_version:
        raise ValueError("document_id và document_version phải đến từ catalog.")
    return Chunk(
        chunk_id=_chunk_id(document_id, document_version, section, index),
        text=text,
        source=source,
        document_id=document_id,
        document_version=document_version,
        source_path=source_path or source,
        page=page,
        checksum=checksum,
        word_count=len(text.split()),
        section=section,
        chunk_index=index,
        policy_key=policy_key,
        policy_status=policy_status,
        department=department,
        allowed_groups=list(allowed_groups or []),
    )


def structure_aware_chunk(
    text: str,
    source: str,
    source_path: str = "",
    page: int | None = None,
    target_words: int = 250,
    overlap_words: int = 40,
    checksum: str | None = None,
    document_id: str | None = None,
    base_chunk_index: int = 0,
    policy_key: str = "",
    document_version: str | None = None,
    policy_status: str = "active",
    department: str = "",
    allowed_groups: list[str] | tuple[str, ...] | None = None,
) -> list[Chunk]:
    """Chia theo heading và câu, có overlap ở ranh giới chunk."""
    if target_words <= 0:
        raise ValueError("target_words phải lớn hơn 0.")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words phải thuộc [0, target_words).")
    if not document_id or not document_version:
        raise ValueError("document_id và document_version phải đến từ catalog.")
    if not text.strip():
        return []

    sections: list[tuple[str, list[str]]] = []
    current_title = "Nội dung chung"
    current_lines: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if HEADING_REGEX.search(stripped):
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = re.sub(r"^#{1,6}\s+", "", stripped).strip()
        else:
            current_lines.append(stripped)
    if current_lines:
        sections.append((current_title, current_lines))

    chunks: list[Chunk] = []
    next_index = base_chunk_index
    for title, lines in sections:
        sentences = _bounded_sentences(split_into_sentences(" ".join(lines)), target_words)
        buffer: list[str] = []
        buffer_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if buffer and buffer_words + sentence_words > target_words:
                content = " ".join(buffer)
                if title != "Nội dung chung":
                    content = f"{title}: {content}"
                chunks.append(
                    _make_chunk(
                        content,
                        source,
                        source_path,
                        document_id,
                        document_version,
                        title,
                        page,
                        checksum,
                        next_index,
                        policy_key,
                        policy_status,
                        department,
                        allowed_groups,
                    )
                )
                next_index += 1
                overlap: list[str] = []
                overlap_count = 0
                for previous in reversed(buffer):
                    previous_words = len(previous.split())
                    if overlap_count + previous_words <= overlap_words or not overlap:
                        overlap.insert(0, previous)
                        overlap_count += previous_words
                    else:
                        break
                buffer = overlap
                buffer_words = overlap_count
            buffer.append(sentence)
            buffer_words += sentence_words

        if buffer:
            content = " ".join(buffer)
            if len(content.split()) >= 15 or not chunks:
                if title != "Nội dung chung":
                    content = f"{title}: {content}"
                chunks.append(
                    _make_chunk(
                        content,
                        source,
                        source_path,
                        document_id,
                        document_version,
                        title,
                        page,
                        checksum,
                        next_index,
                        policy_key,
                        policy_status,
                        department,
                        allowed_groups,
                    )
                )
                next_index += 1
    return chunks


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_words: int = 220,
    overlap_words: int = 30,
    checksum: str | None = None,
    strategy: str = "structure_aware",
    source_path: str = "",
    document_id: str | None = None,
    base_chunk_index: int = 0,
    policy_key: str = "",
    document_version: str | None = None,
    policy_status: str = "active",
    department: str = "",
    allowed_groups: list[str] | tuple[str, ...] | None = None,
) -> list[Chunk]:
    """Chia tài liệu theo strategy; structure-aware là cấu hình chuẩn."""
    if chunk_words <= 0:
        raise ValueError("chunk_words phải lớn hơn 0.")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words phải thuộc [0, chunk_words).")
    if strategy == "structure_aware":
        return structure_aware_chunk(
            text,
            source,
            source_path,
            page,
            chunk_words,
            overlap_words,
            checksum,
            document_id,
            base_chunk_index,
            policy_key,
            document_version,
            policy_status,
            department,
            allowed_groups,
        )
    if strategy != "sliding_window":
        raise ValueError(f"Chiến lược chunking không hợp lệ: {strategy!r}.")
    if not document_id or not document_version:
        raise ValueError("document_id và document_version phải đến từ catalog.")

    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step), start=base_chunk_index):
        part = words[start : start + chunk_words]
        if len(part) < 15 and len(words) > 15:
            continue
        chunks.append(
            _make_chunk(
                " ".join(part),
                source,
                source_path,
                document_id,
                document_version,
                "Nội dung chung",
                page,
                checksum,
                index,
                policy_key,
                policy_status,
                department,
                allowed_groups,
            )
        )
    return chunks
