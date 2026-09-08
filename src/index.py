"""Xây dựng full rebuild bất biến và phát hành ACL index shards."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - môi trường test nhẹ không cần ML runtime
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = Any  # type: ignore[misc,assignment]

from .catalog import CatalogEntry, active_catalog, catalog_snapshot, load_catalog
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
    """Registry record lấy trực tiếp từ catalog, không lấy từ chunk đầu tiên."""

    document_id: str
    policy_key: str
    source_path: str
    checksum: str
    document_version: str
    effective_from: str
    effective_to: str | None
    department: str
    allowed_groups: list[str]
    status: str = "ACTIVE"
    parser_version: str = "structure-aware-v3"
    ingested_at: str = ""
    chunk_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    # Bí danh để các thành phần cũ vẫn đọc được ACL; giá trị gốc vẫn là catalog.
    security_scope: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ingested_at:
            self.ingested_at = datetime.now(UTC).isoformat()
        if not self.security_scope:
            self.security_scope = list(self.allowed_groups)


def detect_corpus_changes(
    data_dir: Path, existing_registry: dict[str, Any]
) -> dict[str, list[Path] | list[str]]:
    """Phân loại thay đổi để audit; ``incremental`` không còn là vector update."""
    current_files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}
    )
    current = {path.relative_to(data_dir).as_posix(): path for path in current_files}
    registry_by_path: dict[str, dict[str, Any]] = {}
    for key, value in existing_registry.items():
        if not isinstance(value, dict):
            continue
        registry_by_path[str(value.get("source_path", key))] = value

    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_files: list[Path] = []
    deleted: list[str] = []
    for relative_path, absolute_path in current.items():
        old = registry_by_path.get(relative_path)
        if old is None:
            new_files.append(absolute_path)
        elif (
            old.get("checksum") != compute_file_checksum(absolute_path)
            or old.get("status") != "ACTIVE"
        ):
            changed_files.append(absolute_path)
        else:
            unchanged_files.append(absolute_path)
    for relative_path, old in registry_by_path.items():
        if relative_path not in current and old.get("status") == "ACTIVE":
            deleted.append(relative_path)
    return {
        "new": new_files,
        "changed": changed_files,
        "unchanged": unchanged_files,
        "deleted": deleted,
    }


def _write_bundle(directory: Path, chunks: list[Chunk], vectors: np.ndarray) -> dict[str, Any]:
    """Ghi một global bundle hoặc shard và kiểm tra count ngay sau khi ghi."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để xây dựng index.")
    directory.mkdir(parents=True, exist_ok=True)
    subset = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(subset.shape[1])
    index.add(subset)
    faiss.write_index(index, str(directory / "index.faiss"))
    save_json(directory / "chunks.json", [chunk.__dict__ for chunk in chunks])
    bm25 = BM25Index.from_texts([chunk.text for chunk in chunks])
    save_json(directory / "bm25_index.json", bm25.to_dict())
    if index.ntotal != len(chunks) or len(bm25.tokenized_corpus) != len(chunks):
        raise ValueError(f"Bundle {directory} có count không khớp sau khi ghi.")
    return {"chunk_count": len(chunks), "vector_count": int(index.ntotal)}


def _validate_release(directory: Path) -> None:
    """Smoke validation cho global bundle và toàn bộ ACL shards."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để validate index.")
    config = load_json(directory / "config.json")
    global_chunks = load_json(directory / "chunks.json")
    global_index = faiss.read_index(str(directory / "index.faiss"))
    global_bm25 = load_json(directory / "bm25_index.json")
    if global_index.ntotal != len(global_chunks):
        raise ValueError("FAISS count != chunks count trong release.")
    if int(global_bm25.get("corpus_size", -1)) != len(global_chunks):
        raise ValueError("BM25 count != chunks count trong release.")
    if config.get("chunk_count") != len(global_chunks):
        raise ValueError("config.chunk_count != chunks count trong release.")
    shard_root = directory / "shards"
    if not shard_root.is_dir():
        raise ValueError("Release schema v3 thiếu thư mục ACL shards.")
    shard_dirs = sorted(path for path in shard_root.iterdir() if path.is_dir())
    if not shard_dirs:
        raise ValueError("Release schema v3 phải có ít nhất một ACL shard.")
    for shard in shard_dirs:
        shard_chunks = load_json(shard / "chunks.json")
        shard_index = faiss.read_index(str(shard / "index.faiss"))
        shard_bm25 = load_json(shard / "bm25_index.json")
        if shard_index.ntotal != len(shard_chunks):
            raise ValueError(f"FAISS count != chunks count tại shard {shard.name}.")
        if int(shard_bm25.get("corpus_size", -1)) != len(shard_chunks):
            raise ValueError(f"BM25 count != chunks count tại shard {shard.name}.")


def _make_version(chunks: list[Chunk], catalog: list[CatalogEntry]) -> str:
    hasher = hashlib.sha256()
    for item in catalog_snapshot(catalog):
        hasher.update(str(item).encode("utf-8"))
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(chunk.content_hash.encode("utf-8"))
    digest = hasher.hexdigest()[:8]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S%f")
    return f"rag-{stamp}-{digest}"


def _activate_release(root: Path, release_path: Path, index_version: str) -> None:
    """Đổi active pointer bằng replace nguyên tử sau khi release đã validate."""
    pointer = {
        "schema_version": 1,
        "index_version": index_version,
        "release_path": str(release_path.relative_to(root)).replace("\\", "/"),
        "activated_at": datetime.now(UTC).isoformat(),
    }
    pointer_tmp = root / ".active_index.json.tmp"
    save_json(pointer_tmp, pointer)
    os.replace(pointer_tmp, root / "active_index.json")


def build_index(config: IndexConfig | None = None, incremental: bool = False) -> dict[str, Any]:
    """Build và validate một full release; chỉ sau đó mới đổi active pointer."""
    config = config or IndexConfig()
    config.validate()
    setup_logging()
    data_dir = Path(config.data_dir)
    root = Path(config.model_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "releases").mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(config.catalog_path, data_dir=data_dir)
    build_date = date.today()
    active_entries = active_catalog(catalog, as_of=build_date)
    if not active_entries:
        raise RuntimeError("Catalog không có tài liệu ACTIVE đang có hiệu lực.")
    LOGGER.info(
        "Build full release: %d active/%d catalog records; incremental=%s (change-aware full rebuild).",
        len(active_entries),
        len(catalog),
        incremental,
    )

    chunks = ingest_folder(
        data_dir,
        chunk_words=config.chunk_words,
        overlap_words=config.overlap_words,
        strategy=config.strategy,
        catalog=catalog,
        as_of=build_date,
    )
    if not chunks:
        raise RuntimeError("Không có chunk hợp lệ sau Document Quality Gate.")
    validate_chunk_quality_contract(chunks)

    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
    for entry in active_entries:
        if not chunks_by_document.get(entry.document_id):
            raise ValueError(f"Tài liệu ACTIVE không tạo được chunk: {entry.document_id}.")

    index_version = _make_version(chunks, catalog)
    release_root = root / "releases"
    temporary = Path(tempfile.mkdtemp(prefix=f".{index_version}-", dir=str(release_root)))
    try:
        if faiss is None:
            raise RuntimeError("Cần cài faiss-cpu để xây dựng index.")
        encoder = SentenceTransformer(config.embedding_model)
        vectors = np.asarray(
            encoder.encode(
                [chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False
            ),
            dtype="float32",
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embedding count không khớp với chunk count.")

        manifest = create_document_manifest(chunks, data_dir)
        timestamp = datetime.now(UTC).isoformat()
        records: dict[str, dict[str, Any]] = {}
        for entry in active_entries:
            doc_chunks = chunks_by_document[entry.document_id]
            source_path = data_dir / entry.file
            record = DocumentRecord(
                document_id=entry.document_id,
                policy_key=entry.policy_key,
                source_path=entry.file,
                checksum=compute_file_checksum(source_path),
                document_version=entry.version,
                effective_from=entry.effective_from.isoformat(),
                effective_to=entry.effective_to.isoformat() if entry.effective_to else None,
                department=entry.department,
                allowed_groups=list(entry.allowed_groups),
                status=entry.status,
                ingested_at=timestamp,
                chunk_count=len(doc_chunks),
                chunk_ids=[chunk.chunk_id for chunk in doc_chunks],
            )
            records[entry.document_id] = asdict(record)

        global_stats = _write_bundle(temporary, chunks, vectors)
        shard_stats: dict[str, dict[str, Any]] = {}
        shard_groups = sorted({group for entry in active_entries for group in entry.allowed_groups})
        for group in shard_groups:
            selected_indices = [
                index for index, chunk in enumerate(chunks) if group in set(chunk.allowed_groups)
            ]
            if not selected_indices:
                raise ValueError(f"ACL group {group!r} không có chunk trong release.")
            shard_chunks = [chunks[index] for index in selected_indices]
            shard_vectors = vectors[selected_indices]
            shard_stats[group] = _write_bundle(
                temporary / "shards" / group,
                shard_chunks,
                shard_vectors,
            )

        corpus_hash = hashlib.sha256(
            "".join(chunk.content_hash for chunk in chunks).encode("utf-8")
        ).hexdigest()[:16]
        release_config = {
            "schema_version": 3,
            "model_version": "enterprise-rag-v1",
            "index_version": index_version,
            "embedding_model": config.embedding_model,
            "reranker_model": config.reranker_model,
            "vector_dimension": int(vectors.shape[1]),
            "chunk_count": len(chunks),
            "document_count": len(active_entries),
            "active_as_of": build_date.isoformat(),
            "chunk_words": config.chunk_words,
            "overlap_words": config.overlap_words,
            "strategy": config.strategy,
            "corpus_hash": corpus_hash,
            "candidate_pool_k": config.candidate_pool_k,
            "dense_k": config.candidate_pool_k,
            "sparse_k": config.candidate_pool_k,
            "rrf_k": config.rrf_k,
            "rerank_candidates": config.rerank_top_k,
            "context_k": config.context_k,
            "evidence_gate_threshold": config.evidence_gate_threshold,
            "retrieval_policy_version": "retrieval-v1",
            "generation_policy_version": "grounded-v1",
            "acl_mode": "pre_retrieval_shards",
            "rebuild_mode": "change-aware-full-rebuild",
        }
        save_json(temporary / "config.json", release_config)
        save_json(temporary / "catalog_snapshot.json", catalog_snapshot(catalog))
        save_json(temporary / "document_registry.json", records)
        save_json(temporary / "document_manifest.json", manifest)
        save_json(
            temporary / "index_manifest.json",
            {
                "schema_version": 1,
                "index_version": index_version,
                "global": global_stats,
                "shards": shard_stats,
                "active_groups": shard_groups,
                "catalog_records": len(catalog),
            },
        )
        save_json(
            temporary / "quality_report.json",
            {"passed": True, "total_chunks": len(chunks), "active_documents": len(active_entries)},
        )
        _validate_release(temporary)

        final_release = release_root / index_version
        temporary.rename(final_release)
        _activate_release(root, final_release, index_version)
    except Exception:
        # Chỉ xóa thư mục build tạm; active release cũ không bị đụng tới.
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "SUCCESS",
        "index_version": index_version,
        "release_path": str(final_release),
        "document_count": len(active_entries),
        "chunk_count": len(chunks),
        "shards": shard_stats,
        "mode": "change-aware-full-rebuild",
    }


def main() -> None:
    """CLI build index; không mutate ``IndexConfig(frozen=True)``."""
    parser = argparse.ArgumentParser(description="Build immutable Enterprise RAG index release")
    parser.add_argument("--data-dir", type=str, default=None, help="Thư mục tài liệu gốc")
    parser.add_argument("--model-dir", type=str, default=None, help="Thư mục gốc lưu release")
    parser.add_argument("--catalog-path", type=str, default=None, help="Knowledge catalog YAML")
    parser.add_argument(
        "--incremental", action="store_true", help="Bí danh audit; vẫn full rebuild"
    )
    args = parser.parse_args()
    config = IndexConfig(
        data_dir=args.data_dir or "data/raw",
        model_dir=args.model_dir or "models/rag_index",
        catalog_path=args.catalog_path or "configs/knowledge_catalog.yaml",
    )
    build_index(config, incremental=args.incremental)


if __name__ == "__main__":
    main()
