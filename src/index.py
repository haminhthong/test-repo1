"""Xây dựng artifact Dense/BM25 từ corpus ACTIVE hiện tại."""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - môi trường chỉ chạy unit test
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - môi trường chỉ chạy unit test
    SentenceTransformer = Any  # type: ignore[misc,assignment]

from .catalog import CatalogEntry, active_catalog, load_catalog
from .config import IndexConfig
from .ingestion import Chunk, ingest_folder
from .ranking import BM25Index
from .utils import compute_file_checksum, load_json, save_json, setup_logging

LOGGER = logging.getLogger("rag_knowledge_assistant.index")


@dataclass
class DocumentRecord:
    """Metadata tối thiểu của tài liệu đã được đưa vào artifact."""

    document_id: str
    policy_key: str
    source_path: str
    checksum: str
    version: str
    effective_from: str
    effective_to: str | None
    department: str
    allowed_groups: list[str]
    status: str
    chunk_count: int
    chunk_ids: list[str]


def _write_group(directory: Path, chunks: list[Chunk], vectors: np.ndarray) -> dict[str, int]:
    """Ghi ba artifact cần thiết cho một nhóm ACL."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để xây dựng index.")
    if not chunks:
        raise ValueError(f"Không thể ghi group rỗng: {directory.name}")

    directory.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError(f"Số vector không khớp số chunk tại group {directory.name}.")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(directory / "index.faiss"))
    save_json(directory / "chunks.json", [asdict(chunk) for chunk in chunks])
    bm25 = BM25Index.from_texts([chunk.text for chunk in chunks])
    save_json(directory / "bm25_index.json", bm25.to_dict())
    return {"chunk_count": len(chunks), "vector_count": int(index.ntotal)}


def _validate_group(directory: Path) -> None:
    """Kiểm tra count giữa FAISS, BM25 và chunks sau khi ghi."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để validate index.")
    required = ("index.faiss", "chunks.json", "bm25_index.json")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise ValueError(f"Artifact group {directory.name} thiếu: {', '.join(missing)}")

    chunks = load_json(directory / "chunks.json")
    index = faiss.read_index(str(directory / "index.faiss"))
    bm25 = load_json(directory / "bm25_index.json")
    if index.ntotal != len(chunks):
        raise ValueError(f"FAISS count không khớp tại group {directory.name}.")
    if int(bm25.get("corpus_size", -1)) != len(chunks):
        raise ValueError(f"BM25 count không khớp tại group {directory.name}.")


def _corpus_hash(chunks: list[Chunk]) -> str:
    """Tạo fingerprint ngắn để biết artifact thuộc corpus nào."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.content_hash.encode("utf-8"))
    return digest.hexdigest()[:16]


def _document_records(
    chunks: list[Chunk], active_entries: list[CatalogEntry], data_dir: Path
) -> list[dict[str, Any]]:
    """Ghi một file metadata duy nhất thay cho chuỗi manifest trùng lặp."""
    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    records: list[dict[str, Any]] = []
    for entry in active_entries:
        document_chunks = chunks_by_document.get(entry.document_id, [])
        records.append(
            asdict(
                DocumentRecord(
                    document_id=entry.document_id,
                    policy_key=entry.policy_key,
                    source_path=entry.file,
                    checksum=compute_file_checksum(data_dir / entry.file),
                    version=entry.version,
                    effective_from=entry.effective_from.isoformat(),
                    effective_to=(
                        entry.effective_to.isoformat() if entry.effective_to is not None else None
                    ),
                    department=entry.department,
                    allowed_groups=list(entry.allowed_groups),
                    status=entry.status,
                    chunk_count=len(document_chunks),
                    chunk_ids=[chunk.chunk_id for chunk in document_chunks],
                )
            )
        )
    return records


def build_index(config: IndexConfig | None = None) -> dict[str, Any]:
    """Xây dựng lại toàn bộ artifact từ tài liệu ACTIVE hiện tại.

    Artifact luôn được ghi vào thư mục ``artifacts`` hiện tại. Muốn cập nhật
    chỉ cần build lại từ corpus và catalog hiện hành.
    """
    config = config or IndexConfig()
    config.validate()
    setup_logging()
    data_dir = Path(config.data_dir)
    artifact_dir = Path(config.model_dir)
    catalog = load_catalog(config.catalog_path, data_dir=data_dir)
    build_date = date.today()
    active_entries = active_catalog(catalog, as_of=build_date)
    if not active_entries:
        raise RuntimeError("Catalog không có tài liệu ACTIVE đang có hiệu lực.")

    chunks = ingest_folder(
        data_dir,
        chunk_words=config.chunk_words,
        overlap_words=config.overlap_words,
        strategy=config.strategy,
        catalog=catalog,
        as_of=build_date,
    )
    if not chunks:
        raise RuntimeError("Không tạo được chunk từ corpus ACTIVE.")
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để xây dựng index.")

    LOGGER.info("Encode %d chunks từ %d tài liệu ACTIVE.", len(chunks), len(active_entries))
    encoder = SentenceTransformer(config.embedding_model)
    vectors = np.asarray(
        encoder.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype="float32",
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding count không khớp chunk count.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = artifact_dir / "groups"
    if groups_dir.exists():
        shutil.rmtree(groups_dir)
    groups_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_group: dict[str, list[Chunk]] = {}
    vectors_by_group: dict[str, list[np.ndarray]] = {}
    for chunk, vector in zip(chunks, vectors, strict=True):
        for group in chunk.allowed_groups:
            chunks_by_group.setdefault(group, []).append(chunk)
            vectors_by_group.setdefault(group, []).append(vector)
    if not chunks_by_group:
        raise ValueError("Corpus ACTIVE không có ACL group.")

    group_stats: dict[str, dict[str, int]] = {}
    for group in sorted(chunks_by_group):
        group_stats[group] = _write_group(
            groups_dir / group,
            chunks_by_group[group],
            np.asarray(vectors_by_group[group], dtype="float32"),
        )
        _validate_group(groups_dir / group)

    config_data = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "vector_dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "document_count": len(active_entries),
        "active_as_of": build_date.isoformat(),
        "chunk_words": config.chunk_words,
        "overlap_words": config.overlap_words,
        "strategy": config.strategy,
        "candidate_pool_k": config.candidate_pool_k,
        "rrf_k": config.rrf_k,
        "rerank_top_k": config.rerank_top_k,
        "context_k": config.context_k,
        "evidence_gate_threshold": config.evidence_gate_threshold,
        "groups": sorted(chunks_by_group),
        "corpus_hash": _corpus_hash(chunks),
    }
    save_json(artifact_dir / "config.json", config_data)
    save_json(artifact_dir / "documents.json", _document_records(chunks, active_entries, data_dir))

    return {
        "status": "ok",
        "artifact_dir": str(artifact_dir),
        "document_count": len(active_entries),
        "chunk_count": len(chunks),
        "groups": group_stats,
    }


def main() -> None:
    """CLI build artifact hiện hành."""
    import argparse

    # Windows mặc định có thể dùng code page không chứa tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Xây dựng artifact cho Vietnamese Policy RAG")
    parser.add_argument("--data-dir", default="data/raw", help="Thư mục tài liệu gốc")
    parser.add_argument("--model-dir", default="artifacts", help="Thư mục artifact")
    parser.add_argument(
        "--catalog-path", default="configs/knowledge_catalog.yaml", help="Catalog YAML"
    )
    parser.add_argument("--chunk-words", type=int, default=250, help="Số từ mục tiêu mỗi chunk")
    parser.add_argument("--overlap-words", type=int, default=40, help="Số từ overlap")
    parser.add_argument(
        "--strategy",
        choices=["structure_aware", "sliding_window"],
        default="structure_aware",
        help="structure_aware là cấu hình chuẩn; sliding_window chỉ dùng làm baseline",
    )
    parser.add_argument("--embedding-model", default=IndexConfig.embedding_model)
    parser.add_argument("--candidate-pool-k", type=int, default=30)
    args = parser.parse_args()
    build_index(
        IndexConfig(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            catalog_path=args.catalog_path,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            strategy=args.strategy,
            embedding_model=args.embedding_model,
            candidate_pool_k=args.candidate_pool_k,
        )
    )


if __name__ == "__main__":
    main()
