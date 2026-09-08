"""Dịch vụ HTTP REST API cho Evidence-Grounded Enterprise Knowledge Assistant (FastAPI Service).

Cung cấp các endpoint chuẩn doanh nghiệp:
- GET /health: Liveness & Readiness tổng quan.
- GET /health/live: Liveness probe cho container/orchestrator.
- GET /health/ready: Readiness probe chuyên sâu kiểm tra toàn vẹn FAISS, BM25, và reranker_mode.
- POST /v1/query (và alias /query): Endpoint production chỉ nhận question;
  identity/ACL đến từ X-API-Key và mapping server-side.
- POST /internal/debug/retrieve: Endpoint admin-only, bị tắt trong production.
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
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .security import (
    AccessContext,
    AuthenticationError,
    access_context_from_api_key,
    is_admin_api_key,
)
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
    version="3.0.0",
)

_retriever: Retriever | None = None
_rag_service: RAGService | None = None


def _effective_index_dir() -> Path:
    """Lấy release mà active pointer đang trỏ tới để health không đọc artifact cũ."""
    pointer = INDEX_DIR / "active_index.json"
    if not pointer.exists():
        return INDEX_DIR
    try:
        target = Path(str(load_json(pointer).get("release_path", "")))
        target = target if target.is_absolute() else INDEX_DIR / target
        index_root = INDEX_DIR.resolve()
        resolved_target = target.resolve()
        return resolved_target if resolved_target.is_relative_to(index_root) else INDEX_DIR
    except Exception:  # noqa: BLE001
        return INDEX_DIR


def _has_acl_shards(index_dir: Path) -> bool:
    """Kiểm tra release có shard ACL thật, không chỉ có thư mục rỗng."""
    shard_root = index_dir / "shards"
    return shard_root.is_dir() and any(path.is_dir() for path in shard_root.iterdir())


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


def get_access_context(request: Request) -> AccessContext:
    """Tạo trusted AccessContext từ header, không đọc quyền trong request body."""
    try:
        return access_context_from_api_key(request.headers.get("X-API-Key"))
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401, detail="X-API-Key không hợp lệ hoặc bị thiếu."
        ) from exc


# ==============================================================================
# SCHEMAS
# ==============================================================================


class QueryIn(BaseModel):
    """Payload đầu vào cho yêu cầu hỏi đáp production /v1/query.

    Client CHỈ được truyền câu hỏi. Danh tính và quyền được server suy ra từ
    X-API-Key; không có trường user_groups trong contract.
    Toàn bộ tham số pipeline (k, reranker, threshold, fusion) do Server áp đặt.
    """

    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Câu hỏi bằng tiếng Việt cần tra cứu tri thức",
        json_schema_extra={"example": "Nhân viên chính thức có bao nhiêu ngày phép năm?"},
    )
    model_config = ConfigDict(extra="ignore")


class CitationItem(BaseModel):
    """Chi tiết trích dẫn minh chứng."""

    id: str = Field(..., description="Mã trích dẫn (ví dụ: C1, C2)")
    document: str = Field(..., description="Tên tệp tài liệu gốc")
    document_id: str = Field(default="", description="ID ổn định của tài liệu")
    version: str = Field(default="", description="Version lấy từ catalog")
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
    version: str = Field(default="", description="Version lấy từ catalog")


class QueryOut(BaseModel):
    """Payload đầu ra chuẩn hóa cho câu trả lời và trích dẫn theo contract v3."""

    request_id: str = Field(..., description="Mã truy vết duy nhất của request")
    answer: str = Field(..., description="Câu trả lời tổng hợp căn thực hoặc thông báo từ chối")
    decision: dict[str, Any] = Field(
        ..., description="Quyết định hệ thống (action, answerable, reason)"
    )
    retrieval: dict[str, Any] = Field(..., description="Thống kê ứng viên và chế độ reranker")
    grounding: dict[str, Any] = Field(
        ..., description="Chỉ số căn thực bằng chứng (evidence_score, gate_passed, citation_valid)"
    )
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Danh sách các trích dẫn có cấu trúc [C1], [C2]",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Metadata nguồn đã được ACL duyệt, không gồm điểm debug",
    )
    versions: dict[str, str] = Field(..., description="Phiên bản model, index và reranker_mode")
    # Các trường tương thích ngược.
    model_version: str = Field(default="enterprise-rag-v1")
    index_version: str = Field(default="unknown")
    evidence_gate_passed: bool = Field(default=False)


class DebugRetrieveIn(BaseModel):
    """Payload cho endpoint nghiên cứu nội bộ /internal/debug/retrieve."""

    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    min_score: FiniteFloat | None = None
    use_reranker: bool = True


class FeedbackIn(BaseModel):
    """Payload thu thập phản hồi từ người dùng hoặc người thẩm định."""

    request_id: str = Field(..., min_length=2, max_length=128)
    helpful: bool
    expected_document: str | None = Field(default=None, max_length=256)
    expected_section: str | None = Field(default=None, max_length=256)
    reviewer_note: str | None = Field(default=None, max_length=2000)


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
    effective_dir = _effective_index_dir()
    required_files = ("config.json", "index.faiss", "chunks.json", "bm25_index.json")
    ready = all((effective_dir / filename).exists() for filename in required_files)

    model_version = "not_trained"
    index_version = "none"
    chunk_count = 0

    if ready:
        try:
            config_data = load_json(effective_dir / "config.json")
            model_version = config_data.get("model_version", "enterprise-rag-v1")
            index_version = config_data.get("index_version", "unknown")
            chunk_count = config_data.get("chunk_count", 0)
            ready = (
                ready
                and int(config_data.get("schema_version", 0)) >= 3
                and _has_acl_shards(effective_dir)
            )
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
    effective_dir = _effective_index_dir()
    required_files = ("config.json", "index.faiss", "chunks.json", "bm25_index.json")
    missing = [f for f in required_files if not (effective_dir / f).exists()]

    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Hệ thống chưa sẵn sàng. Thiếu các artifact: {', '.join(missing)}",
        )

    try:
        config_data = load_json(effective_dir / "config.json")
        chunks_data = load_json(effective_dir / "chunks.json")
        if int(config_data.get("schema_version", 0)) < 3 or not _has_acl_shards(effective_dir):
            raise HTTPException(
                status_code=503, detail="Active index chưa phải release schema v3 có ACL shards."
            )
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
            "model_version": config_data.get("model_version", "enterprise-rag-v1"),
            "index_version": config_data.get("index_version", "unknown"),
            "vector_dimension": config_data.get("vector_dimension"),
            "total_chunks": len(chunks_data),
            "reranker_mode": reranker_mode,
            "reranker_ready": reranker_mode == "neural",
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
    """Production endpoint với AccessContext được xác thực từ X-API-Key."""
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:12]}"

    try:
        access_context = get_access_context(request)
        rag_svc = get_rag_service()
        req_obj = QueryRequest(
            question=payload.question,
            access_context=access_context,
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
                    version=str(s.get("version", "")),
                )
                for s in res.sources
            ],
            versions=res.versions,
            model_version=res.versions.get("model_version", "enterprise-rag-v1"),
            index_version=res.versions.get("index", "unknown"),
            evidence_gate_passed=bool(res.grounding.get("evidence_gate_passed", False)),
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("[req=%s] Lỗi không mong muốn khi xử lý query: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ dịch vụ RAG.") from exc


@app.post(
    "/internal/debug/retrieve",
    summary="Endpoint admin nghiên cứu & debug (tắt trong production)",
    response_model=dict[str, Any],
)
def debug_retrieve(payload: DebugRetrieveIn, request: Request) -> dict[str, Any]:
    """Endpoint admin-only; production không expose debug retrieval."""
    import os

    if os.getenv("ENV", "development").lower() == "production":
        raise HTTPException(status_code=404, detail="Endpoint không tồn tại trong production.")
    if not is_admin_api_key(request.headers.get("X-API-Key")):
        raise HTTPException(status_code=403, detail="Chỉ admin mới được dùng endpoint debug.")
    retriever_inst = get_retriever()
    hits = retriever_inst.search(
        query=payload.question,
        k=payload.top_k,
        candidate_k=payload.candidate_k,
        min_score=payload.min_score,
        use_reranker=payload.use_reranker,
        user_groups=("employee", "hr", "finance", "security"),
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
def record_feedback(payload: FeedbackIn, request: Request) -> dict[str, str]:
    """Thu thập tín hiệu phản hồi vào feedback/feedback.jsonl sau khi xác thực."""
    get_access_context(request)
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
