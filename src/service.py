"""Điều phối online pipeline: ACL -> retrieval -> gate -> generation."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generation import ABSTAIN_PHRASE, generate_grounded_response
from .security import AccessContext

if TYPE_CHECKING:
    from .retrieval import Retriever

LOGGER = logging.getLogger("rag_knowledge_assistant.service")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = PROJECT_ROOT / "feedback"


@dataclass(frozen=True)
class QueryRequest:
    """Yêu cầu nội bộ đã gắn AccessContext tin cậy."""

    question: str
    access_context: AccessContext | None = None
    request_id: str = ""
    # Chỉ giữ để caller prototype cũ chuyển tiếp; API không expose trường này.
    user_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", f"req_{uuid.uuid4().hex[:12]}")
        if self.access_context is None and self.user_groups:
            object.__setattr__(
                self,
                "access_context",
                AccessContext(
                    user_id="legacy-internal", groups=self.user_groups, auth_method="legacy"
                ),
            )

    @property
    def resolved_access_context(self) -> AccessContext:
        """Không có context thì deny-by-default."""
        return self.access_context or AccessContext.deny_all()


@dataclass
class RAGResult:
    """Response contract công khai của dịch vụ."""

    request_id: str
    answer: str
    decision: dict[str, Any]
    retrieval: dict[str, Any]
    grounding: dict[str, Any]
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _public_source(hit: dict[str, Any]) -> dict[str, Any]:
    """Ẩn điểm debug khỏi response user-facing."""
    return {
        "chunk_id": hit.get("chunk_id", ""),
        "document_id": hit.get("document_id", ""),
        "document": hit.get("source", ""),
        "source": hit.get("source", ""),
        "source_path": hit.get("source_path", hit.get("source", "")),
        "version": hit.get("document_version", hit.get("version", "")),
        "section": hit.get("section"),
        "page": hit.get("page"),
    }


class RAGService:
    """Orchestrator trung tâm, không cho LLM chạy khi policy không pass."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.config = retriever.config
        self.versions = {
            "index": str(self.config.get("index_version", "unknown")),
            "retrieval_policy": str(self.config.get("retrieval_policy_version", "retrieval-v1")),
            "generation_policy": str(self.config.get("generation_policy_version", "grounded-v1")),
            "embedding_model": str(self.config.get("embedding_model", "unknown")),
            "reranker_model": str(self.config.get("reranker_model", "unknown")),
        }
        # Alias phục vụ consumer cũ; contract mới dùng index/retrieval_policy.
        self.versions["index_version"] = self.versions["index"]
        self.versions["model_version"] = str(self.config.get("model_version", "enterprise-rag-v1"))

    def answer(self, request: QueryRequest) -> RAGResult:
        """Thực thi đúng state machine ANSWER/ABSTAIN/EVIDENCE_ONLY."""
        started = time.perf_counter()
        request_id = request.request_id
        context = request.resolved_access_context
        hits = self.retriever.search(
            request.question,
            k=int(self.config.get("context_k", 4)),
            candidate_k=int(self.config.get("candidate_pool_k", 30)),
            use_reranker=True,
            access_context=context,
            max_context_tokens=1500,
        )
        reranker_mode = self.retriever.reranker.mode
        reranker_ready = reranker_mode == "neural"
        passed_hits = [hit for hit in hits if hit.get("gate_passed", False)]
        top_score = max((float(hit.get("evidence_score", 0.0)) for hit in hits), default=0.0)

        # Reranker hỏng không được đi qua gate bằng điểm heuristic. Vẫn có thể
        # trả evidence-only để người dùng tự kiểm tra nguồn, nhưng tuyệt đối
        # không gọi LLM như một câu trả lời bình thường.
        if hits and not reranker_ready:
            answer, citations, gate_passed, grounding_meta = generate_grounded_response(
                request.question,
                hits,
                max_chunks=int(self.config.get("context_k", 4)),
                reranker_available=False,
            )
            result = self._result(
                request_id,
                answer,
                "EVIDENCE_ONLY",
                "RERANKER_UNAVAILABLE",
                hits,
                citations,
                {
                    "evidence_score": top_score,
                    **grounding_meta,
                    "evidence_gate_passed": gate_passed,
                },
                started,
                reranker_mode,
            )
            self._record_telemetry(result)
            return result

        if not passed_hits:
            result = self._result(
                request_id,
                ABSTAIN_PHRASE,
                "ABSTAIN",
                "NO_EVIDENCE_FOUND" if not hits else "INSUFFICIENT_EVIDENCE",
                hits,
                [],
                {"evidence_score": top_score, "evidence_gate_passed": False},
                started,
                reranker_mode,
            )
            self._record_telemetry(result)
            return result

        answer, citations, gate_passed, grounding_meta = generate_grounded_response(
            request.question,
            passed_hits,
            max_chunks=int(self.config.get("context_k", 4)),
            reranker_available=reranker_ready,
        )
        action = str(grounding_meta.get("action", "ABSTAIN")).upper()
        result = self._result(
            request_id,
            answer,
            action,
            str(grounding_meta.get("reason", "UNKNOWN")),
            hits,
            citations,
            {
                "evidence_score": top_score,
                **grounding_meta,
                "evidence_gate_passed": gate_passed,
                "reranker_logit": max(
                    (
                        float(hit["reranker_logit"])
                        for hit in hits
                        if hit.get("reranker_logit") is not None
                    ),
                    default=None,
                ),
            },
            started,
            reranker_mode,
        )
        self._record_telemetry(result)
        return result

    def _result(
        self,
        request_id: str,
        answer: str,
        action: str,
        reason: str,
        hits: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        grounding: dict[str, Any],
        started: float,
        reranker_mode: str,
    ) -> RAGResult:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return RAGResult(
            request_id=request_id,
            answer=answer,
            decision={
                "action": action,
                "answerable": action == "ANSWER",
                "reason": reason,
            },
            retrieval={
                "pipeline": "authorized_dense_bm25_rrf_reranker",
                "candidate_count": len(hits),
                "evidence_count": len([hit for hit in hits if hit.get("gate_passed", False)]),
                "reranker_mode": reranker_mode,
                "acl_mode": self.config.get("acl_mode", "pre_retrieval_shards"),
            },
            grounding=grounding,
            citations=citations,
            sources=[_public_source(hit) for hit in hits],
            versions=self.versions,
            latency_ms=elapsed,
        )

    def _record_telemetry(self, result: RAGResult) -> None:
        """Ghi telemetry không lưu câu hỏi, context hoặc answer thô."""
        event = {
            "request_id": result.request_id,
            "timestamp": time.time(),
            "latency_ms": result.latency_ms,
            "decision": result.decision.get("action"),
            "reason": result.decision.get("reason"),
            "evidence_count": result.retrieval.get("evidence_count"),
            "citation_count": len(result.citations),
            "citation_valid": result.grounding.get("citation_valid"),
            "index_version": result.versions.get("index"),
        }
        try:
            TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
            with (TELEMETRY_DIR / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Không thể ghi telemetry: %s", exc)
