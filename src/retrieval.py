"""Authorized Dense + BM25 + RRF retrieval trên ACL group indexes."""

from __future__ import annotations

import logging
import unicodedata
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

from .config import DEFAULT_EVIDENCE_GATE_THRESHOLD
from .ranking import BM25Index, CrossEncoderReranker, reciprocal_rank_fusion
from .security import AccessContext
from .utils import load_json

LOGGER = logging.getLogger("rag_knowledge_assistant.retrieval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def normalize_query(query: str) -> str:
    """Chuẩn hóa Unicode và thu gọn khoảng trắng của câu hỏi."""
    normalized = unicodedata.normalize("NFKC", str(query).strip())
    return " ".join(normalized.split())


def _load_group(directory: Path) -> dict[str, Any]:
    """Nạp và kiểm tra một group index."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để nạp index.")
    required = ("index.faiss", "chunks.json", "bm25_index.json")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Group {directory.name} thiếu: {', '.join(missing)}")

    index = faiss.read_index(str(directory / "index.faiss"))
    chunks = load_json(directory / "chunks.json")
    bm25 = BM25Index.from_dict(load_json(directory / "bm25_index.json"))
    if index.ntotal != len(chunks):
        raise ValueError(f"FAISS count không khớp tại group {directory.name}.")
    if len(bm25.tokenized_corpus) != len(chunks):
        raise ValueError(f"BM25 count không khớp tại group {directory.name}.")
    return {"index": index, "chunks": chunks, "bm25": bm25}


class Retriever:
    """Chỉ tìm trong các group mà AccessContext được phép đọc."""

    def __init__(self, model_dir: str | Path | None = None, use_reranker: bool = True) -> None:
        self.index_dir = Path(model_dir) if model_dir else DEFAULT_ARTIFACT_DIR
        config_path = self.index_dir / "config.json"
        groups_dir = self.index_dir / "groups"
        if not config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config artifact tại '{self.index_dir}'.")
        if not groups_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy ACL groups tại '{groups_dir}'.")

        self.config: dict[str, Any] = load_json(config_path)
        group_dirs = sorted(path for path in groups_dir.iterdir() if path.is_dir())
        if not group_dirs:
            raise ValueError("Artifact phải có ít nhất một ACL group.")
        configured_groups = {
            str(group).strip().lower()
            for group in self.config.get("groups", [])
            if str(group).strip()
        }
        actual_groups = {directory.name.lower() for directory in group_dirs}
        if configured_groups != actual_groups:
            raise ValueError("Danh sách ACL group trong config.json không khớp thư mục groups/.")
        self.groups = {directory.name: _load_group(directory) for directory in group_dirs}

        # Một chunk có thể xuất hiện ở nhiều group; chỉ giữ một bản trong thuộc tính
        # tiện ích để kiểm tra và đánh giá, còn retrieval hợp nhất theo chunk_id.
        chunks_by_id: dict[str, dict[str, Any]] = {}
        for bundle in self.groups.values():
            for chunk in bundle["chunks"]:
                chunks_by_id.setdefault(str(chunk.get("chunk_id", "")), chunk)
        self.chunks = [chunk for chunk_id, chunk in chunks_by_id.items() if chunk_id]

        self.default_candidate_k = int(self.config.get("candidate_pool_k", 30))
        self.rrf_k = int(self.config.get("rrf_k", 60))
        self.rerank_top_k = int(self.config.get("rerank_top_k", 20))
        self.evidence_gate_threshold = float(
            self.config.get("evidence_gate_threshold", DEFAULT_EVIDENCE_GATE_THRESHOLD)
        )

        embedding_model = str(
            self.config.get(
                "embedding_model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        )
        reranker_model = str(
            self.config.get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        )
        LOGGER.info("Đang tải embedding model '%s'...", embedding_model)
        self.encoder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoderReranker(model_name=reranker_model, enabled=use_reranker)
        LOGGER.info("Retriever sẵn sàng: %d chunks, %d groups.", len(self.chunks), len(self.groups))

    @staticmethod
    def _authorized(chunk: dict[str, Any], groups: tuple[str, ...]) -> bool:
        """Mặc định từ chối nếu chunk thiếu ACL hoặc không giao quyền."""
        allowed = {
            str(group).strip().lower()
            for group in (chunk.get("allowed_groups") or [])
            if str(group).strip()
        }
        return bool(allowed and set(groups) & allowed)

    def _search_group(
        self,
        bundle: dict[str, Any],
        query_vector: np.ndarray | None,
        query: str,
        pool_size: int,
        use_dense: bool,
        use_bm25: bool,
        groups: tuple[str, ...],
        dense_ranks: dict[str, int],
        dense_scores: dict[str, float],
        sparse_ranks: dict[str, int],
        sparse_scores: dict[str, float],
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> None:
        """Tìm và lọc ACL ngay trong từng group trước khi fusion."""
        chunks = bundle["chunks"]
        if use_dense and query_vector is not None:
            scores, indices = bundle["index"].search(query_vector, min(pool_size, len(chunks)))
            for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
                if 0 <= int(index) < len(chunks):
                    chunk = chunks[int(index)]
                    if not self._authorized(chunk, groups):
                        continue
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if chunk_id and rank < dense_ranks.get(chunk_id, 10**9):
                        dense_ranks[chunk_id] = rank
                        dense_scores[chunk_id] = float(score)
                        chunks_by_id[chunk_id] = dict(chunk)

        if use_bm25:
            for rank, (index, score) in enumerate(
                bundle["bm25"].search(query, top_k=pool_size), start=1
            ):
                if 0 <= int(index) < len(chunks):
                    chunk = chunks[int(index)]
                    if not self._authorized(chunk, groups):
                        continue
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if chunk_id and rank < sparse_ranks.get(chunk_id, 10**9):
                        sparse_ranks[chunk_id] = rank
                        sparse_scores[chunk_id] = float(score)
                        chunks_by_id[chunk_id] = dict(chunk)

    def search(
        self,
        query: str,
        k: int = 4,
        candidate_k: int | None = None,
        min_score: float | None = None,
        use_dense: bool = True,
        use_bm25: bool = True,
        use_reranker: bool = True,
        access_context: AccessContext | None = None,
        max_context_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """ACL -> Dense/BM25 -> RRF -> Cross-Encoder.

        ``use_dense=False`` hoặc ``use_bm25=False`` chỉ phục vụ đánh giá
        baseline. Pipeline trực tuyến luôn dùng cả hai nhánh.
        """
        clean_query = normalize_query(query)
        if not clean_query or k <= 0 or not (use_dense or use_bm25):
            return []
        if access_context is None or not access_context.groups:
            return []

        pool_size = max(1, int(candidate_k or self.default_candidate_k))
        query_vector = None
        if use_dense:
            query_vector = np.asarray(
                self.encoder.encode([clean_query], normalize_embeddings=True), dtype="float32"
            )

        groups = access_context.groups
        dense_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        sparse_ranks: dict[str, int] = {}
        sparse_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict[str, Any]] = {}
        for group in sorted(set(groups) & self.groups.keys()):
            self._search_group(
                self.groups[group],
                query_vector,
                clean_query,
                pool_size,
                use_dense,
                use_bm25,
                groups,
                dense_ranks,
                dense_scores,
                sparse_ranks,
                sparse_scores,
                chunks_by_id,
            )
        if not chunks_by_id:
            return []

        rrf_scores = reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=self.rrf_k)
        candidates: list[dict[str, Any]] = []
        for chunk_id, rrf_score in rrf_scores.items():
            item = dict(chunks_by_id[chunk_id])
            item.update(
                {
                    "dense_score": round(dense_scores.get(chunk_id, 0.0), 4),
                    "bm25_score": round(sparse_scores.get(chunk_id, 0.0), 4),
                    "rrf_score": round(rrf_score, 6),
                    "dense_rank": dense_ranks.get(chunk_id),
                    "bm25_rank": sparse_ranks.get(chunk_id),
                }
            )
            candidates.append(item)
        candidates.sort(key=lambda item: (-item["rrf_score"], item.get("chunk_id", "")))
        candidates = candidates[: min(self.rerank_top_k, len(candidates))]

        reranker_available = False
        if use_reranker:
            candidates = self.reranker.rerank(clean_query, candidates, top_k=len(candidates))
            reranker_available = self.reranker.mode == "neural"
        else:
            for item in candidates:
                item["reranker_logit"] = None
                item["reranker_score"] = None
                item["evidence_score"] = item["rrf_score"]
                item["retrieval_score"] = item["rrf_score"]

        selected: list[dict[str, Any]] = []
        for candidate in candidates:
            tokens = set(str(candidate.get("text", "")).casefold().split())
            duplicate = any(
                candidate.get("document_id") == chosen.get("document_id")
                and candidate.get("section") == chosen.get("section")
                and len(tokens & set(str(chosen.get("text", "")).casefold().split()))
                / max(1, len(tokens | set(str(chosen.get("text", "")).casefold().split())))
                > 0.85
                for chosen in selected
            )
            if duplicate:
                continue
            selected.append(candidate)
            if len(selected) >= k:
                break

        if max_context_tokens is not None:
            bounded: list[dict[str, Any]] = []
            used = 0
            for item in selected:
                estimate = max(1, int(len(str(item.get("text", "")).split()) * 1.3))
                if used + estimate > max_context_tokens and bounded:
                    break
                bounded.append(item)
                used += estimate
            selected = bounded

        threshold = self.evidence_gate_threshold if min_score is None else float(min_score)
        results: list[dict[str, Any]] = []
        for item in selected:
            score = float(item.get("evidence_score", item.get("rrf_score", 0.0)))
            item["score"] = score
            item["retrieval_score"] = score
            item["evidence_score"] = score
            item["reranker_available"] = reranker_available
            item["gate_passed"] = bool(reranker_available and score >= self.evidence_gate_threshold)
            if score >= threshold:
                results.append(item)
        return results
