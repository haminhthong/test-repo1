"""Xây dựng và Lưu trữ chỉ mục tri thức chính quy (Enterprise Indexing Module).

Tệp này quản lý:
1. Document Registry theo dõi trạng thái tài liệu (DocumentRecord: path, checksum, version, ACL scope, status).
2. Phát hiện thay đổi gia tăng (Incremental Change Detection: NEW, CHANGED, UNCHANGED, DELETED/TOMBSTONE).
3. Parser Quality Gate & Chunk Quality Contract (chặn empty chunks, duplicate chunk_ids, parse rác).
4. Vectorize bằng SentenceTransformers và lập FAISS FlatIP Dense Index.
5. Lập BM25Okapi Lexical Index độc lập.
6. Đóng gói Versioned Index Artifacts (index.faiss, chunks.json, bm25_index.json, document_registry.json, quality_report.json, config.json).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import IndexConfig
from .ingestion import (
    Chunk,
    create_document_manifest,
    ingest_folder,
    validate_chunk_quality_contract,
)
from .ranking import BM25Index
from .utils import compute_file_checksum, load_json, save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.index")


@dataclass
class DocumentRecord:
    """Bản ghi đăng ký tài liệu theo chuẩn Enterprise Knowledge Registry."""

    document_id: str
    source_path: str
    checksum: str
    document_version: str
    parser_version: str = "structure-aware-v2"
    security_scope: list[str] = field(default_factory=lambda: ["public", "employee"])
    status: str = "ACTIVE"
    ingested_at: str = ""
    chunk_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ingested_at:
            self.ingested_at = datetime.now(timezone.utc).isoformat()


def detect_corpus_changes(
    data_dir: Path,
    existing_registry: dict[str, Any],
) -> dict[str, list[Path]]:
    """Phân loại tài liệu thành 4 nhóm: NEW, CHANGED, UNCHANGED, DELETED."""
    current_files = sorted([p for p in data_dir.rglob("*") if p.is_file()])
    current_rel_paths = {p.relative_to(data_dir).as_posix(): p for p in current_files}

    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_files: list[Path] = []
    deleted_paths: list[str] = []

    for rel_path, abs_path in current_rel_paths.items():
        if rel_path not in existing_registry:
            new_files.append(abs_path)
        else:
            old_record = existing_registry[rel_path]
            current_checksum = compute_file_checksum(abs_path)
            if old_record.get("checksum") != current_checksum or old_record.get("status") != "ACTIVE":
                changed_files.append(abs_path)
            else:
                unchanged_files.append(abs_path)

    for rel_path, old_record in existing_registry.items():
        if rel_path not in current_rel_paths and old_record.get("status") == "ACTIVE":
            deleted_paths.append(rel_path)

    return {
        "new": new_files,
        "changed": changed_files,
        "unchanged": unchanged_files,
        "deleted": deleted_paths,
    }


def build_index(
    config: IndexConfig | None = None,
    incremental: bool = False,
) -> dict[str, Any]:
    """Xây dựng chỉ mục tri thức hoàn chỉnh, hỗ trợ chế độ Incremental Indexing.

    Args:
        config (Optional[IndexConfig]): Cấu hình Index.
        incremental (bool): Kích hoạt chế độ cập nhật gia tăng (mặc định: False - rebuild đầy đủ).

    Returns:
        Dict[str, Any]: Báo cáo kết quả quá trình indexing.
    """
    config = config or IndexConfig()
    config.validate()
    setup_logging()

    data_dir = Path(config.data_dir)
    output_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = output_dir / "document_registry.json"
    chunks_path = output_dir / "chunks.json"
    existing_registry: dict[str, Any] = {}
    if registry_path.exists():
        try:
            existing_registry = load_json(registry_path)
        except Exception:  # noqa: BLE001
            existing_registry = {}

    LOGGER.info(
        "Bắt đầu quy trình Indexing từ '%s' (strategy='%s', chunk_words=%d, incremental=%s)...",
        config.data_dir,
        config.strategy,
        config.chunk_words,
        incremental,
    )

    # 1. Phát hiện thay đổi gia tăng nếu chạy ở chế độ incremental
    if incremental and existing_registry and chunks_path.exists():
        changes = detect_corpus_changes(data_dir, existing_registry)
        LOGGER.info(
            "Phát hiện Corpus Changes: %d new, %d changed, %d unchanged, %d deleted.",
            len(changes["new"]),
            len(changes["changed"]),
            len(changes["unchanged"]),
            len(changes["deleted"]),
        )
        if not changes["new"] and not changes["changed"] and not changes["deleted"]:
            LOGGER.info("Toàn bộ tài liệu không thay đổi. Bỏ qua re-indexing để tiết kiệm tài nguyên.")
            return {"status": "SKIPPED_NO_CHANGES", "total_chunks": len(load_json(chunks_path))}

    # 2. Ingest toàn bộ chunks
    chunks = ingest_folder(
        config.data_dir,
        chunk_words=config.chunk_words,
        overlap_words=config.overlap_words,
        strategy=config.strategy,
    )

    if not chunks:
        raise RuntimeError(
            f"Không tìm thấy tài liệu hợp lệ trong '{config.data_dir}'. "
            "Hãy kiểm tra lại thư mục hoặc chạy 'python scripts/download_data.py'."
        )

    # 3. Thực thi Chunk Quality Contract
    quality_report = validate_chunk_quality_contract(chunks)
    LOGGER.info("Chunk Quality Contract PASSED: %s", quality_report)
    save_json(output_dir / "quality_report.json", quality_report)

    # 4. Xây dựng Document Registry
    manifest = create_document_manifest(chunks, config.data_dir)
    document_registry: dict[str, Any] = {}
    timestamp_now = datetime.now(timezone.utc).isoformat()

    for doc_key, doc_info in manifest.get("documents", {}).items():
        rec = DocumentRecord(
            document_id=doc_info["document_id"],
            source_path=doc_key,
            checksum=doc_info["checksum"] or "",
            document_version=timestamp_now[:10],
            ingested_at=timestamp_now,
            chunk_count=doc_info["chunk_count"],
            chunk_ids=doc_info["chunk_ids"],
            security_scope=getattr(chunks[0], "security_scope", ["public", "employee"]),
        )
        document_registry[doc_key] = asdict(rec)

    save_json(output_dir / "document_registry.json", document_registry)
    save_json(output_dir / "document_manifest.json", manifest)

    # 5. Xây dựng Dense Embeddings & FAISS Index
    LOGGER.info("Khởi tạo mô hình Embedding: %s", config.embedding_model)
    encoder = SentenceTransformer(config.embedding_model)

    texts = [c.text for c in chunks]
    LOGGER.info("Vectorize %d chunks văn bản...", len(texts))

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(embeddings, dtype="float32")
    vector_dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(vectors)

    faiss_path = output_dir / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    LOGGER.info("Đã ghi FAISS index (%d vectors) tại %s", index.ntotal, faiss_path.name)

    # 6. Xây dựng BM25 Lexical Index
    LOGGER.info("Xây dựng BM25Okapi Lexical Index cho %d chunks...", len(texts))
    bm25_index = BM25Index.from_texts(texts)
    save_json(output_dir / "bm25_index.json", bm25_index.to_dict())

    # 7. Lưu Chunks Metadata
    chunks_dict_list = [c.__dict__ for c in chunks]
    save_json(chunks_path, chunks_dict_list)

    # 8. Tính toán Corpus Hash & Index Versioning
    corpus_hasher = hashlib.sha256()
    for text in texts:
        corpus_hasher.update(text.encode("utf-8"))
    corpus_hash = corpus_hasher.hexdigest()[:16]

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    index_version = f"{timestamp_str}-{corpus_hash[:6]}"

    config_metadata = {
        "schema_version": 2,
        "model_version": "rag-evidence-v2",
        "index_version": index_version,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "vector_dimension": vector_dimension,
        "chunk_count": len(chunks),
        "document_count": manifest["total_documents"],
        "chunk_words": config.chunk_words,
        "overlap_words": config.overlap_words,
        "strategy": config.strategy,
        "corpus_hash": corpus_hash,
        "candidate_pool_k": config.candidate_pool_k,
        "rrf_k": config.rrf_k,
        "rerank_top_k": config.rerank_top_k,
        "evidence_gate_threshold": config.evidence_gate_threshold,
        "data_dir": str(config.data_dir),
    }
    save_json(output_dir / "config.json", config_metadata)

    LOGGER.info(
        "Chỉ mục hoàn tất! Version: %s, Tài liệu: %d, Chunks: %d",
        index_version,
        manifest["total_documents"],
        len(chunks),
    )

    return {
        "status": "SUCCESS",
        "index_version": index_version,
        "document_count": manifest["total_documents"],
        "chunk_count": len(chunks),
        "quality_report": quality_report,
    }


def main() -> None:
    """CLI Entrypoint hỗ trợ cả build toàn diện và incremental."""
    parser = argparse.ArgumentParser(description="Build RAG Knowledge Index Artifacts")
    parser.add_argument("--data-dir", type=str, default=None, help="Thư mục tài liệu gốc")
    parser.add_argument("--model-dir", type=str, default=None, help="Thư mục lưu trữ artifact")
    parser.add_argument("--incremental", action="store_true", help="Chế độ cập nhật gia tăng")
    args = parser.parse_args()

    cfg = IndexConfig()
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    if args.model_dir:
        cfg.model_dir = Path(args.model_dir)

    build_index(cfg, incremental=args.incremental)


if __name__ == "__main__":
    main()
