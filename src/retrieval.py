"""Authorized Dense + BM25 + RRF retrieval trên ACL shards."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - unit test có thể chỉ mock Retriever
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = Any  # type: ignore[misc,assignment]

from .config import DEFAULT_EVIDENCE_GATE_THRESHOLD
from .ranking import BM25Index, CrossEncoderReranker
from .security import AccessContext
from .utils import load_json

LOGGER = logging.getLogger("rag_knowledge_assistant.retrieval")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_query(query: str) -> str:
    """Chuẩn hóa Unicode và thu gọn khoảng trắng của câu hỏi."""
    normalized = unicodedata.normalize("NFKC", str(query).strip())
    return " ".join(normalized.split())


def _load_bundle(directory: Path) -> dict[str, Any]:
    """Nạp và kiểm tra một index bundle (global hoặc shard)."""
    if faiss is None:
        raise RuntimeError("Cần cài faiss-cpu để nạp index.")
    required = ("index.faiss", "chunks.json", "bm25_index.json")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle {directory} thiếu artifact: {', '.join(missing)}")
    index = faiss.read_index(str(directory / "index.faiss"))
    chunks = load_json(directory / "chunks.json")
    bm25 = BM25Index.from_dict(load_json(directory / "bm25_index.json"))
    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS count ({index.ntotal}) != chunks count ({len(chunks)}) tại {directory}."
        )
    if len(bm25.tokenized_corpus) != len(chunks):
        raise ValueError(
            f"BM25 count ({len(bm25.tokenized_corpus)}) != chunks count ({len(chunks)}) tại {directory}."
        )
    return {"index": index, "chunks": chunks, "bm25": bm25}


def _resolve_release(root: Path) -> Path:
    pointer = root / "active_index.json"
    if not pointer.exists():
        return root
    metadata = load_json(pointer)
    release_path = Path(str(metadata.get("release_path", "")))
    if not release_path.is_absolute():
        release_path = root / release_path
    root_resolved = root.resolve()
    release_resolved = release_path.resolve()
    if not release_resolved.is_relative_to(root_resolved):
        raise ValueError("Active index pointer trỏ ra ngoài model_dir.")
    if not release_resolved.exists():
        raise FileNotFoundError(f"Active index trỏ tới release không tồn tại: {release_path}")
    return release_resolved


class Retriever:
    """Bộ truy xuất chỉ tìm trên shard được AccessContext cấp quyền."""

    def __init__(self, model_dir: str | Path | None = None, use_reranker: bool = True) -> None:
        root = Path(model_dir) if model_dir else PROJECT_ROOT / "models/rag_index"
        index_dir = _resolve_release(root)
        config_path = index_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config index tại '{index_dir}'.")

        self.root_dir = root
        self.index_dir = index_dir
        self.config: dict[str, Any] = load_json(config_path)
        shard_root = index_dir / "shards"
        shard_dirs = (
            sorted(path for path in shard_root.iterdir() if path.is_dir())
            if shard_root.is_dir()
            else []
        )
        self.versioned_acl_release = bool(
            int(self.config.get("schema_version", 0)) >= 3 and shard_dirs
        )
        if int(self.config.get("schema_version", 0)) >= 3 and not self.versioned_acl_release:
            raise ValueError("Release schema v3 phải có ít nhất một ACL shard.")
        self.evidence_gate_threshold = float(
            self.config.get("evidence_gate_threshold", DEFAULT_EVIDENCE_GATE_THRESHOLD)
        )
        self.default_candidate_k = int(self.config.get("candidate_pool_k", 30))
        self.rrf_k = int(self.config.get("rrf_k", 60))
        self.rerank_top_k = int(
            self.config.get("rerank_candidates", self.config.get("rerank_top_k", 20))
        )

        embedding_model = self.config.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        reranker_model = self.config.get(
            "reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        LOGGER.info("Đang tải embedding model '%s'...", embedding_model)
        self.encoder = SentenceTransformer(embedding_model)

        # Bundle toàn cục dùng cho health/tương thích; production search dùng shards.
        global_bundle = _load_bundle(index_dir)
        self.index = global_bundle["index"]
        self.chunks: list[dict[str, Any]] = global_bundle["chunks"]
        self.bm25_index: BM25Index = global_bundle["bm25"]

        self.shards: dict[str, dict[str, Any]] = {}
        for shard_dir in shard_dirs:
            self.shards[shard_dir.name] = _load_bundle(shard_dir)

        self.reranker = CrossEncoderReranker(model_name=reranker_model, enabled=use_reranker)
        LOGGER.info(
            "Retriever sẵn sàng: %d chunks, %d ACL shards, policy=%s.",
            len(self.chunks),
            len(self.shards),
            self.config.get("retrieval_policy_version", "unknown"),
        )

    @staticmethod
    def _authorized(chunk: dict[str, Any], groups: tuple[str, ...]) -> bool:
        """Defense-in-depth: ACL rỗng hoặc thiếu metadata luôn bị từ chối."""
        raw_scopes = chunk.get("allowed_groups", chunk.get("security_scope", []))
        scopes = {str(scope).strip().lower() for scope in (raw_scopes or []) if str(scope).strip()}
        return bool(scopes and set(groups) & scopes)

    def _search_bundle(
        self,
        bundle: dict[str, Any],
        query_vector: np.ndarray,
        query: str,
        pool_size: int,
        dense_enabled: bool,
        sparse_enabled: bool,
        groups: tuple[str, ...],
        dense_ranks: dict[str, int],
        dense_scores: dict[str, float],
        sparse_ranks: dict[str, int],
        sparse_scores: dict[str, float],
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> None:
        """Tìm trong một shard rồi lọc ACL trước khi đưa vào candidate union."""
        chunks = bundle["chunks"]
        if dense_enabled:
            scores, indices = bundle["index"].search(query_vector, min(pool_size, len(chunks)))
            for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
                if 0 <= int(index) < len(chunks):
                    chunk = chunks[int(index)]
                    if not self._authorized(chunk, groups):
                        continue
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if not chunk_id:
                        continue
                    if rank < dense_ranks.get(chunk_id, 10**9):
                        dense_ranks[chunk_id] = rank
                        dense_scores[chunk_id] = float(score)
                        chunks_by_id[chunk_id] = dict(chunk)

        if sparse_enabled:
            for rank, (index, score) in enumerate(
                bundle["bm25"].search(query, top_k=pool_size), start=1
            ):
                if 0 <= int(index) < len(chunks):
                    chunk = chunks[int(index)]
                    if not self._authorized(chunk, groups):
                        continue
                    chunk_id = str(chunk.get("chunk_id", ""))
                    if not chunk_id:
                        continue
                    if rank < sparse_ranks.get(chunk_id, 10**9):
                        sparse_ranks[chunk_id] = rank
                        sparse_scores[chunk_id] = float(score)
                        chunks_by_id[chunk_id] = dict(chunk)

    def search(
        self,
        query: str,
        k: int = 4,
        candidate_k: int | None = None,
        min_score: float | None = None,
        dense_weight: float | None = None,
        use_reranker: bool = True,
        user_groups: tuple[str, ...] | list[str] | None = None,
        access_context: AccessContext | None = None,
        max_context_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Dense/BM25 -> RRF -> reranker trên authorized shards.

        ``user_groups`` chỉ còn để tương thích với caller nội bộ cũ. API
        production truyền ``AccessContext`` do server tạo, không truyền body.
        ``dense_weight`` chỉ hỗ trợ 0/1 cho baseline; không còn linear fusion.
        """
        clean_query = normalize_query(query)
        if not clean_query:
            return []
        if getattr(self, "versioned_acl_release", None) is False:
            LOGGER.error(
                "Từ chối truy vấn artifact cũ chưa có ACL shards; cần build release schema v3."
            )
            return []
        if dense_weight is not None and dense_weight not in {0.0, 1.0}:
            raise ValueError(
                "dense_weight không còn dùng cho linear fusion; chỉ nhận 0.0 hoặc 1.0."
            )

        if access_context is not None:
            groups = access_context.groups
        elif user_groups is not None:
            groups = tuple(
                str(group).strip().lower() for group in user_groups if str(group).strip()
            )
        else:
            groups = ()
        if not groups:
            return []

        dense_enabled = dense_weight != 0.0
        sparse_enabled = dense_weight != 1.0
        pool_size = max(1, int(candidate_k or self.default_candidate_k))
        query_vector: np.ndarray | None = None
        if dense_enabled:
            query_vector = np.asarray(
                self.encoder.encode([clean_query], normalize_embeddings=True), dtype="float32"
            )

        dense_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        sparse_ranks: dict[str, int] = {}
        sparse_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict[str, Any]] = {}

        configured_shards = getattr(self, "shards", None)
        shard_map = (
            configured_shards if isinstance(configured_shards, dict) and configured_shards else {}
        )
        if shard_map:
            for group in sorted(set(groups)):
                bundle = shard_map.get(group)
                if bundle is not None:
                    self._search_bundle(
                        bundle,
                        query_vector
                        if query_vector is not None
                        else np.empty((1, 0), dtype="float32"),
                        clean_query,
                        pool_size,
                        dense_enabled,
                        sparse_enabled,
                        groups,
                        dense_ranks,
                        dense_scores,
                        sparse_ranks,
                        sparse_scores,
                        chunks_by_id,
                    )
        else:
            # Fallback đọc artifact cũ; vẫn deny-by-default và lọc trước fusion.
            self._search_bundle(
                {"index": self.index, "chunks": self.chunks, "bm25": self.bm25_index},
                query_vector if query_vector is not None else np.empty((1, 0), dtype="float32"),
                clean_query,
                pool_size,
                dense_enabled,
                sparse_enabled,
                groups,
                dense_ranks,
                dense_scores,
                sparse_ranks,
                sparse_scores,
                chunks_by_id,
            )

        if not chunks_by_id:
            return []

        # Tính RRF theo chunk_id để không làm sai khi shard có bản sao.
        rrf_scores: dict[str, float] = {}
        for chunk_id in set(dense_ranks) | set(sparse_ranks):
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        candidates: list[dict[str, Any]] = []
        for chunk_id in rrf_scores:
            item = dict(chunks_by_id[chunk_id])
            item.update(
                {
                    "dense_score": round(dense_scores.get(chunk_id, 0.0), 4),
                    "bm25_score": round(sparse_scores.get(chunk_id, 0.0), 4),
                    "rrf_score": round(rrf_scores[chunk_id], 6),
                    "dense_rank": dense_ranks.get(chunk_id),
                    "bm25_rank": sparse_ranks.get(chunk_id),
                    "allowed_groups": item.get("allowed_groups", item.get("security_scope", [])),
                }
            )
            candidates.append(item)
        candidates.sort(key=lambda item: (-item["rrf_score"], item.get("chunk_id", "")))
        candidates = candidates[: min(self.rerank_top_k, len(candidates))]

        reranker_available = False
        if use_reranker:
            reranked = self.reranker.rerank(clean_query, candidates, top_k=len(candidates))
            reranker_available = self.reranker.mode == "neural"
        else:
            reranked = candidates
            for item in reranked:
                item["reranker_logit"] = None
                item["reranker_score"] = None
                item["evidence_score"] = item["rrf_score"]
                item["retrieval_score"] = item["rrf_score"]

        selected: list[dict[str, Any]] = []
        for candidate in reranked:
            tokens = set(str(candidate.get("text", "")).casefold().split())
            duplicate = False
            for chosen in selected:
                if chosen.get("document_id") == candidate.get("document_id") and chosen.get(
                    "section"
                ) == candidate.get("section"):
                    chosen_tokens = set(str(chosen.get("text", "")).casefold().split())
                    overlap = len(tokens & chosen_tokens) / max(1, len(tokens | chosen_tokens))
                    if overlap > 0.85:
                        duplicate = True
                        break
            if not duplicate:
                selected.append(candidate)
            if len(selected) >= max(1, k):
                break

        if max_context_tokens is not None:
            budget: list[dict[str, Any]] = []
            used = 0
            for item in selected:
                estimate = max(1, int(len(str(item.get("text", "")).split()) * 1.3))
                if used + estimate > max_context_tokens and budget:
                    break
                budget.append(item)
                used += estimate
            selected = budget

        threshold = self.evidence_gate_threshold if min_score is None else float(min_score)
        result: list[dict[str, Any]] = []
        for item in selected:
            score = float(item.get("evidence_score", item.get("rrf_score", 0.0)))
            item["score"] = score
            item["retrieval_score"] = score
            item["evidence_score"] = score
            item["reranker_available"] = reranker_available
            item["gate_passed"] = bool(reranker_available and score >= threshold)
            if min_score is not None and score < threshold:
                continue
            result.append(item)
        return result
