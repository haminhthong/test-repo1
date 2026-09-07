"""Dịch vụ HTTP REST API cho Evidence-Grounded Enterprise Knowledge Assistant (FastAPI Service).

Cung cấp các endpoint chuẩn doanh nghiệp:
- GET /health: Liveness & Readiness tổng quan.
- GET /health/live: Liveness probe cho container/orchestrator.
- GET /health/ready: Readiness probe chuyên sâu kiểm tra toàn vẹn FAISS, BM25, và reranker_mode.
- POST /v1/query (và alias /query): Endpoint production nghiêm ngặt, chỉ nhận question & security scope.
  Khóa cứng cấu hình retrieval phía server, ngăn client bypass canonical policy.
- POST /internal/debug/retrieve: Endpoint nghiên cứu nội bộ cho phép tùy biến tham số ablation & debug.
- POST /v1/feedback: Thu thập phản hồi người dùng phục vụ Continuous Evaluation Loop.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .service import QueryRequest, RAGService
from .utils import load_json, setup_logging

if TYPE_CHECKING:
    from .retrieval import Retriever

setup_logging()
LOGGER = logging.getLogger("rag_knowledge_assistant.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "models/rag_index"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"

app = FastAPI(
    title="Evidence-Grounded Enterprise Knowledge Assistant",
    description=(
        "REST API tra cứu tri thức doanh nghiệp căn thực bằng chứng, "
        "kết hợp Dense FAISS, BM25Okapi, RRF, Cross-Encoder Reranking, "
        "Evidence Quality Gate và Claim-Level Citations."
    ),
    version="2.1.0",
)

_retriever: Retriever | None = None
_rag_service: RAGService | None = None


def get_retriever() -> Retriever:
    """Khởi tạo hoặc trả về thể hiện Singleton của Retriever."""
    global _retriever
    if _retriever is None:
        LOGGER.info("Khởi tạo Retriever lần đầu tiên (Lazy Loading)...")
        from .retrieval import Retriever

        _retriever = Retriever(model_dir=INDEX_DIR)
    return _retriever


def get_rag_service() -> RAGService:
    """Khởi tạo hoặc trả về thể hiện Singleton của RAGService."""
    global _rag_service
    if _rag_service is None:
        retriever_inst = get_retriever()
        _rag_service = RAGService(retriever_inst)
    return _rag_service


# ==============================================================================
# SCHEMAS
# ==============================================================================


class QueryIn(BaseModel):
    """Payload đầu vào cho yêu cầu hỏi đáp production /v1/query.

    Client CHỈ được truyền câu hỏi và thẩm quyền danh tính (Access Scope).
    Toàn bộ tham số pipeline (k, reranker, threshold, fusion) do Server áp đặt.
    """

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Câu hỏi bằng tiếng Việt cần tra cứu tri thức",
        json_schema_extra={"example": "Nhân viên chính thức có bao nhiêu ngày phép năm?"},
    )
    user_groups: list[str] = Field(
        default=["public", "employee"],
        description="Danh sách nhóm thẩm quyền bảo mật của người dùng (Access Context)",
    )


class CitationItem(BaseModel):
    """Chi tiết trích dẫn minh chứng."""

    id: str = Field(..., description="Mã trích dẫn (ví dụ: C1, C2)")
    document: str = Field(..., description="Tên tệp tài liệu gốc")
    source_path: str = Field(..., description="Đường dẫn tương đối của tài liệu")
    page: int | None = Field(None, description="Số trang tương ứng (khi khả dụng)")
    section: str | None = Field(None, description="Tên phần/chương/mục chứa trích dẫn")
    chunk_id: str = Field(..., description="Mã định danh duy nhất của chunk")
    quote: str = Field(..., description="Trích đoạn ngắn minh chứng sự thật")


class SourceItem(BaseModel):
    """Thông tin chi tiết của từng đoạn tài liệu nguồn thu thập được."""

    chunk_id: str = Field(..., description="Mã định danh duy nhất của chunk")
    document_id: str = Field(..., description="Mã hash định danh tài liệu")
    source: str = Field(..., description="Tên tệp tài liệu gốc")
    source_path: str = Field(..., description="Đường dẫn tương đối của tệp")
    page: int | None = Field(None, description="Số trang tương ứng (khi có)")
    section: str | None = Field(None, description="Tiêu đề mục chứa chunk")
    retrieval_score: float = Field(
        ..., description="Điểm xếp hạng độ liên quan chuẩn hóa [0.0 - 1.0]"
    )
    score: float = Field(..., description="Điểm số tương thích ngược")
    dense_score: float = Field(..., description="Điểm tương đồng Cosine ngữ nghĩa")
    bm25_score: float = Field(..., description="Điểm từ khóa BM25")
    evidence_score: float | None = Field(None, description="Điểm đánh giá bằng chứng")
    rerank_score: float | None = Field(None, description="Điểm sau Cross-Encoder Reranking")


class QueryOut(BaseModel):
    """Payload đầu ra chuẩn hóa cho câu trả lời và trích dẫn theo Architecture v2."""

    request_id: str = Field(..., description="Mã truy vết duy nhất của request")
    answer: str = Field(
        ..., description="Câu trả lời tổng hợp căn thực hoặc thông báo từ chối"
    )
    decision: dict[str, Any] = Field(
        ..., description="Quyết định hệ thống (action, answerable, reason)"
    )
    retrieval: dict[str, Any] = Field(
        ..., description="Thống kê ứng viên và chế độ reranker"
    )
    grounding: dict[str, Any] = Field(
        ..., description="Chỉ số căn thực bằng chứng (evidence_score, gate_passed, citation_valid)"
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Danh sách các trích dẫn có cấu trúc [C1], [C2]",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Danh sách toàn bộ các nguồn tài liệu ứng viên được duyệt",
    )
    versions: dict[str, str] = Field(
        ..., description="Phiên bản model, index và reranker_mode"
    )
    # Các trường tương thích ngược (Backward compatibility)
    model_version: str = Field(default="rag-evidence-v2")
    index_version: str = Field(default="unknown")
    evidence_gate_passed: bool = Field(default=True)


class DebugRetrieveIn(BaseModel):
    """Payload cho endpoint nghiên cứu nội bộ /internal/debug/retrieve."""

    question: str
    top_k: int = 4
    candidate_k: int | None = None
    min_score: float | None = None
    dense_weight: float | None = None
    use_reranker: bool = True
    user_groups: list[str] | None = None


class FeedbackIn(BaseModel):
    """Payload thu thập phản hồi từ người dùng hoặc người thẩm định."""

    request_id: str
    helpful: bool
    expected_document: str | None = None
    expected_section: str | None = None
    reviewer_note: str | None = None


# ==============================================================================
# ENDPOINTS
# ==============================================================================


@app.get(
    "/health",
    summary="Kiểm tra tổng quát sức khỏe dịch vụ và chỉ mục",
    response_model=dict[str, Any],
)
def health() -> dict[str, Any]:
    """Kiểm tra tổng quan trạng thái, không làm lộ đường dẫn filesystem."""
    required_files = ("config.json", "index.faiss", "chunks.json")
    ready = all((INDEX_DIR / filename).exists() for filename in required_files)

    model_version = "not_trained"
    index_version = "none"
    chunk_count = 0

    if ready:
        try:
            config_data = load_json(INDEX_DIR / "config.json")
            model_version = config_data.get("model_version", "rag-evidence-v2")
            index_version = config_data.get("index_version", "unknown")
            chunk_count = config_data.get("chunk_count", 0)
        except Exception:  # noqa: BLE001
            ready = False

    return {
        "status": "ok" if ready else "degraded",
        "index_ready": ready,
        "model_version": model_version,
        "index_version": index_version,
        "chunk_count": chunk_count,
    }


@app.get(
    "/health/live",
    summary="Liveness probe cho orchestrator",
    response_model=dict[str, Any],
)
def health_live() -> dict[str, Any]:
    """Kiểm tra tiến trình API đang hoạt động bình thường."""
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/health/ready",
    summary="Readiness probe kiểm tra tính sẵn sàng của toàn bộ artifacts",
    response_model=dict[str, Any],
)
def health_ready() -> dict[str, Any]:
    """Kiểm tra chuyên sâu: FAISS index, chunks.json, BM25, và reranker_mode."""
    required_files = ("config.json", "index.faiss", "chunks.json", "bm25_index.json")
    missing = [f for f in required_files if not (INDEX_DIR / f).exists()]

    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Hệ thống chưa sẵn sàng. Thiếu các artifact: {', '.join(missing)}",
        )

    try:
        config_data = load_json(INDEX_DIR / "config.json")
        chunks_data = load_json(INDEX_DIR / "chunks.json")
        retriever_inst = get_retriever()

        if retriever_inst.index.ntotal != len(chunks_data):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Bất nhất artifact: FAISS ({retriever_inst.index.ntotal}) != "
                    f"chunks.json ({len(chunks_data)})"
                ),
            )

        reranker_mode = retriever_inst.reranker.mode

        return {
            "status": "ready",
            "model_version": config_data.get("model_version", "rag-evidence-v2"),
            "index_version": config_data.get("index_version", "unknown"),
            "vector_dimension": config_data.get("vector_dimension"),
            "total_chunks": len(chunks_data),
            "reranker_mode": reranker_mode,
            "reranker_ready": reranker_mode != "disabled",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Lỗi khi kiểm tra tính sẵn sàng: {exc}",
        ) from exc


@app.post(
    "/v1/query",
    summary="Gửi câu hỏi tra cứu tri thức nội bộ (Production Canonical Endpoint)",
    response_model=QueryOut,
)
@app.post(
    "/query",
    summary="Alias tương thích ngược cho endpoint hỏi đáp",
    response_model=QueryOut,
    include_in_schema=False,
)
def query(payload: QueryIn, request: Request) -> QueryOut:
    """Production Endpoint: Nhận câu hỏi, áp đặt toàn bộ chính sách retrieval từ server."""
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"

    try:
        rag_svc = get_rag_service()
        req_obj = QueryRequest(
            question=payload.question,
            user_groups=tuple(payload.user_groups),
            request_id=request_id,
        )
        res = rag_svc.answer(req_obj)

        return QueryOut(
            request_id=res.request_id,
            answer=res.answer,
            decision=res.decision,
            retrieval=res.retrieval,
            grounding=res.grounding,
            citations=[CitationItem(**c) for c in res.citations],
            sources=[
                SourceItem(
                    chunk_id=s.get("chunk_id", ""),
                    document_id=s.get("document_id", ""),
                    source=s.get("source", ""),
                    source_path=s.get("source_path", s.get("source", "")),
                    page=s.get("page"),
                    section=s.get("section"),
                    retrieval_score=float(s.get("retrieval_score", 0.0)),
                    score=float(s.get("score", s.get("retrieval_score", 0.0))),
                    dense_score=float(s.get("dense_score", 0.0)),
                    bm25_score=float(s.get("bm25_score", 0.0)),
                    evidence_score=float(s.get("evidence_score", 0.0)) if s.get("evidence_score") is not None else None,
                    rerank_score=float(s.get("rerank_score", 0.0)) if s.get("rerank_score") is not None else None,
                )
                for s in res.sources
            ],
            versions=res.versions,
            model_version=res.versions.get("model_version", "rag-evidence-v2"),
            index_version=res.versions.get("index_version", "unknown"),
            evidence_gate_passed=bool(res.grounding.get("evidence_gate_passed", True)),
        )
    except Exception as exc:
        LOGGER.exception("[req=%s] Lỗi không mong muốn khi xử lý query: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ dịch vụ RAG.") from exc


@app.post(
    "/internal/debug/retrieve",
    summary="Endpoint nghiên cứu & debug: Cho phép tùy biến toàn bộ cờ retrieval",
    response_model=dict[str, Any],
)
def debug_retrieve(payload: DebugRetrieveIn) -> dict[str, Any]:
    """Endpoint dùng cho thử nghiệm, ablation benchmark, và kiểm thử kỹ thuật."""
    retriever_inst = get_retriever()
    hits = retriever_inst.search(
        query=payload.question,
        k=payload.top_k,
        candidate_k=payload.candidate_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        use_reranker=payload.use_reranker,
        user_groups=payload.user_groups,
    )
    return {
        "question": payload.question,
        "count": len(hits),
        "hits": hits,
        "reranker_mode": retriever_inst.reranker.mode,
    }


@app.post(
    "/v1/feedback",
    summary="Ghi nhận phản hồi người dùng phục vụ Continuous Improvement Loop",
    response_model=dict[str, str],
)
def record_feedback(payload: FeedbackIn) -> dict[str, str]:
    """Thu thập tín hiệu phản hồi vào feedback/feedback.jsonl."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    feedback_file = FEEDBACK_DIR / "feedback.jsonl"

    record = payload.model_dump()
    record["received_at"] = datetime.now(timezone.utc).isoformat()

    try:
        with feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"status": "recorded", "request_id": payload.request_id}
    except Exception as exc:
        LOGGER.error("Lỗi khi ghi feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Không thể lưu phản hồi.") from exc
