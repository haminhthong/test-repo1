"""Điều phối online pipeline: ACL -> retrieval -> gate -> generation."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .generation import ABSTAIN_PHRASE, generate_grounded_response
from .security import AccessContext

if TYPE_CHECKING:
    from .retrieval import Retriever


@dataclass(frozen=True)
class QueryRequest:
    """Yêu cầu nội bộ đã gắn AccessContext do server xác thực."""

    question: str
    access_context: AccessContext
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question không được rỗng.")
        if not self.request_id:
            object.__setattr__(self, "request_id", f"req_{uuid.uuid4().hex[:12]}")


@dataclass
class RAGResult:
    """Kết quả gọn cho API; thông tin chẩn đoán nằm trong ``debug``."""

    request_id: str
    mode: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả sang payload JSON."""
        return asdict(self)


def _public_source(hit: dict[str, Any]) -> dict[str, Any]:
    """Chỉ trả metadata nguồn cần cho người dùng, không trả điểm xếp hạng."""
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "document_id": hit.get("document_id", ""),
        "source": hit.get("source", ""),
        "source_path": hit.get("source_path", hit.get("source", "")),
        "version": hit.get("document_version", ""),
        "section": hit.get("section"),
        "page": hit.get("page"),
    }


class RAGService:
    """Điều phối kết quả answer, sources_only và abstain."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.config = retriever.config

    def answer(self, request: QueryRequest) -> RAGResult:
        """Chỉ gọi LLM khi evidence và Cross-Encoder cùng đạt điều kiện."""
        started = time.perf_counter()
        context_k = int(self.config.get("context_k", 4))
        hits = self.retriever.search(
            request.question,
            k=context_k,
            candidate_k=int(self.config.get("candidate_pool_k", 30)),
            use_dense=True,
            use_bm25=True,
            use_reranker=True,
            access_context=request.access_context,
            # Không dùng raw reranker threshold ở retrieval. Khi reranker không
            # khả dụng, điểm còn lại là RRF và không cùng thang với logit.
            min_score=float("-inf"),
            max_context_tokens=1500,
        )
        reranker_mode = self.retriever.reranker.mode
        reranker_ready = reranker_mode == "neural"
        top_score = max(
            (float(hit.get("evidence_score", 0.0)) for hit in hits),
            default=0.0,
        )

        if not hits:
            return self._result(
                request.request_id,
                "abstain",
                ABSTAIN_PHRASE,
                [],
                [],
                "NO_EVIDENCE_FOUND",
                top_score,
                reranker_mode,
                started,
            )

        if not reranker_ready:
            answer, citations, _, metadata = generate_grounded_response(
                request.question,
                hits,
                max_chunks=context_k,
                reranker_available=False,
            )
            return self._result(
                request.request_id,
                str(metadata.get("action", "sources_only")),
                answer,
                hits,
                citations,
                str(metadata.get("reason", "RERANKER_UNAVAILABLE")),
                top_score,
                reranker_mode,
                started,
                metadata,
            )

        passed_hits = [hit for hit in hits if hit.get("gate_passed", False)]
        if not passed_hits:
            return self._result(
                request.request_id,
                "abstain",
                ABSTAIN_PHRASE,
                hits,
                [],
                "INSUFFICIENT_EVIDENCE",
                top_score,
                reranker_mode,
                started,
            )

        answer, citations, _, metadata = generate_grounded_response(
            request.question,
            passed_hits,
            max_chunks=context_k,
            reranker_available=True,
        )
        return self._result(
            request.request_id,
            str(metadata.get("action", "abstain")),
            answer,
            hits,
            citations,
            str(metadata.get("reason", "UNKNOWN")),
            top_score,
            reranker_mode,
            started,
            metadata,
        )

    def _result(
        self,
        request_id: str,
        mode: str,
        answer: str,
        hits: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        reason: str,
        top_score: float,
        reranker_mode: str,
        started: float,
        metadata: dict[str, Any] | None = None,
    ) -> RAGResult:
        """Tạo response gọn và giữ metrics chẩn đoán ở debug."""
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        grounding = metadata or {}
        debug = {
            "reason": reason,
            "retrieval": {
                "pipeline": "acl_dense_bm25_rrf_cross_encoder",
                "candidate_count": len(hits),
                "evidence_count": sum(bool(hit.get("gate_passed")) for hit in hits),
                "reranker_mode": reranker_mode,
            },
            "grounding": {"evidence_score": top_score, **grounding},
            "latency_ms": elapsed,
        }
        return RAGResult(
            request_id=request_id,
            mode=mode.lower(),
            answer=answer,
            citations=citations,
            sources=[_public_source(hit) for hit in hits],
            debug=debug,
        )
