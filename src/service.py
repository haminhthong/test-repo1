"""Module điều phối dịch vụ RAG chính quy (RAG Service Orchestrator).

Chịu trách nhiệm kết nối toàn trình (End-to-End Online Pipeline):
1. Chuẩn hóa truy vấn và xác thực danh tính / thẩm quyền (Access Context).
2. Tìm kiếm ứng viên kép Authorized Hybrid Retrieval (FAISS + BM25 + RRF).
3. Xếp hạng lại bằng Champion Cross-Encoder (kèm cơ chế Fallback).
4. Kiểm soát chất lượng bằng chứng (Evidence Quality Gate) -> Early Abstention nếu bằng chứng yếu.
5. Đóng gói ngữ cảnh với ngân sách token (Context Builder & Delimiters).
6. Sinh câu trả lời căn thực (Grounded Generation) và kiểm định trích dẫn (Citation Validation).
7. Ghi nhận Telemetry có cấu trúc phục vụ giám sát vận hành.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .generation import ABSTAIN_PHRASE, generate_grounded_response
from .utils import save_json

if TYPE_CHECKING:
    from .retrieval import Retriever

LOGGER = logging.getLogger("rag_knowledge_assistant.service")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = PROJECT_ROOT / "feedback"


@dataclass
class QueryRequest:
    """Yêu cầu hỏi đáp đầu vào chuẩn hóa."""

    question: str
    user_groups: tuple[str, ...] = ("public", "employee")
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", f"req_{uuid.uuid4().hex[:12]}")


@dataclass
class RAGResult:
    """Kết quả trả lời tri thức chuẩn hóa của toàn hệ thống."""

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
        """Chuyển thành từ điển xuất bản API."""
        return asdict(self)


class RAGService:
    """Orchestrator trung tâm quản lý luồng dữ liệu truy vấn tri thức."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.config = retriever.config

        self.versions = {
            "model_version": str(self.config.get("model_version", "rag-evidence-v2")),
            "index_version": str(self.config.get("index_version", "unknown")),
            "embedding_model": str(self.config.get("embedding_model", "unknown")),
            "reranker_model": str(self.config.get("reranker_model", "unknown")),
            "reranker_mode": self.retriever.reranker.mode,
        }

    def answer(self, request: QueryRequest) -> RAGResult:
        """Thực thi chu trình Canonical Online RAG khép kín cho một truy vấn."""
        start_time = time.perf_counter()
        req_id = request.request_id

        LOGGER.info("[req=%s] Bắt đầu xử lý câu hỏi: '%s'", req_id, request.question)

        # 1. Truy xuất bằng chứng có kiểm soát thẩm quyền (Authorized Hybrid Search)
        hits = self.retriever.search(
            query=request.question,
            k=int(self.config.get("rerank_top_k", 4)),
            candidate_k=int(self.config.get("candidate_pool_k", 30)),
            use_reranker=True,
            user_groups=request.user_groups,
            max_context_tokens=1500,
        )

        # 2. Đánh giá Evidence Gate
        passed_hits = [h for h in hits if h.get("gate_passed", True)]
        top_score = max((float(h.get("evidence_score", 0.0)) for h in hits), default=0.0)

        # Early Abstain nếu không có bằng chứng vượt qua ngưỡng
        if not passed_hits:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            LOGGER.info("[req=%s] Early Abstain kích hoạt: Không đủ bằng chứng tin cậy.", req_id)
            result = RAGResult(
                request_id=req_id,
                answer=ABSTAIN_PHRASE,
                decision={
                    "action": "abstain",
                    "answerable": False,
                    "reason": "INSUFFICIENT_EVIDENCE",
                },
                retrieval={
                    "pipeline": "hybrid_rrf_reranker",
                    "candidate_count": len(hits),
                    "evidence_count": 0,
                    "reranker_mode": self.retriever.reranker.mode,
                },
                grounding={
                    "evidence_score": round(top_score, 4),
                    "evidence_gate_passed": False,
                    "citation_valid": False,
                    "citation_coverage": 0.0,
                },
                citations=[],
                sources=hits,
                versions=self.versions,
                latency_ms=elapsed,
            )
            self._record_telemetry(result)
            return result

        # 3. Grounded Generation & Citation Verification
        answer_text, citations, gate_passed, ground_meta = generate_grounded_response(
            question=request.question,
            hits=passed_hits,
            max_chunks=int(self.config.get("rerank_top_k", 4)),
        )

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        result = RAGResult(
            request_id=req_id,
            answer=answer_text,
            decision={
                "action": ground_meta.get("action", "answer"),
                "answerable": gate_passed and answer_text != ABSTAIN_PHRASE,
                "reason": ground_meta.get("reason", "SUFFICIENT_EVIDENCE"),
            },
            retrieval={
                "pipeline": "hybrid_rrf_reranker",
                "candidate_count": len(hits),
                "evidence_count": len(passed_hits),
                "reranker_mode": self.retriever.reranker.mode,
            },
            grounding={
                "evidence_score": round(top_score, 4),
                "evidence_gate_passed": gate_passed,
                "citation_valid": ground_meta.get("citation_valid", False),
                "citation_coverage": ground_meta.get("citation_coverage", 0.0),
            },
            citations=citations,
            sources=hits,
            versions=self.versions,
            latency_ms=elapsed,
        )

        self._record_telemetry(result)
        return result

    def _record_telemetry(self, result: RAGResult) -> None:
        """Ghi nhận log telemetry có cấu trúc (bảo mật, không lưu văn bản nội bộ thô)."""
        telemetry_event = {
            "request_id": result.request_id,
            "timestamp": time.time(),
            "latency_ms": result.latency_ms,
            "decision": result.decision.get("action"),
            "answerable": result.decision.get("answerable"),
            "evidence_score": result.grounding.get("evidence_score"),
            "evidence_count": result.retrieval.get("evidence_count"),
            "citation_count": len(result.citations),
            "citation_valid": result.grounding.get("citation_valid"),
            "reranker_mode": result.retrieval.get("reranker_mode"),
            "index_version": result.versions.get("index_version"),
        }

        try:
            TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
            log_file = TELEMETRY_DIR / "events.jsonl"
            import json

            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry_event, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Không thể ghi file telemetry: %s", exc)
