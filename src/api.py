"""HTTP API tối thiểu cho Vietnamese Policy RAG."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

app = FastAPI(
    title="Vietnamese Policy RAG",
    description=(
        "Tra cứu chính sách nội bộ tiếng Việt bằng Dense + BM25 + RRF, "
        "Cross-Encoder reranking và citation reference validation."
    ),
    version="1.0.0",
)

_retriever: Retriever | None = None
_rag_service: RAGService | None = None


class QueryIn(BaseModel):
    """Payload chuẩn: client chỉ gửi câu hỏi."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Câu hỏi bằng tiếng Việt",
        json_schema_extra={"example": "Nhân viên chính thức có bao nhiêu ngày phép năm?"},
    )


class CitationItem(BaseModel):
    """Citation reference và quote đi kèm."""

    id: str
    document: str
    document_id: str = ""
    version: str = ""
    source_path: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
    quote: str


class SourceItem(BaseModel):
    """Metadata nguồn đã qua ACL."""

    chunk_id: str
    document_id: str
    source: str
    source_path: str
    page: int | None = None
    section: str | None = None
    version: str = ""


class QueryOut(BaseModel):
    """Response user-facing gọn: answer, sources_only hoặc abstain."""

    request_id: str
    mode: Literal["answer", "sources_only", "abstain"]
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)


class DebugRetrieveIn(BaseModel):
    """Payload debug chỉ bật ngoài production và cần admin API key."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    min_score: FiniteFloat | None = None
    use_dense: bool = True
    use_bm25: bool = True
    use_reranker: bool = True


def _artifact_health() -> dict[str, Any]:
    """Kiểm tra artifact hiện hành mà không tải model nặng."""
    config_path = ARTIFACT_DIR / "config.json"
    documents_path = ARTIFACT_DIR / "documents.json"
    groups_dir = ARTIFACT_DIR / "groups"
    required = ("index.faiss", "chunks.json", "bm25_index.json")
    groups = (
        sorted(path for path in groups_dir.iterdir() if path.is_dir())
        if groups_dir.is_dir()
        else []
    )
    loaded = config_path.is_file() and documents_path.is_file() and bool(groups)
    if loaded:
        try:
            config = load_json(config_path)
            configured_groups = {
                str(group).strip().lower()
                for group in config.get("groups", [])
                if str(group).strip()
            }
        except (OSError, TypeError, ValueError):
            configured_groups = set()
            loaded = False
        loaded = loaded and configured_groups == {group.name for group in groups}
        loaded = loaded and all(
            all((group / name).is_file() for name in required) for group in groups
        )

    chunk_count = 0
    if config_path.is_file():
        try:
            chunk_count = int(load_json(config_path).get("chunk_count", 0))
        except (OSError, TypeError, ValueError):
            loaded = False
    return {
        "status": "ok" if loaded else "degraded",
        "index_loaded": loaded,
        "total_chunks": chunk_count,
        "groups": [group.name for group in groups],
    }


def get_retriever() -> Retriever:
    """Tải retriever một lần khi có request cần truy xuất."""
    global _retriever
    if _retriever is None:
        from .retrieval import Retriever

        _retriever = Retriever(model_dir=ARTIFACT_DIR)
    return _retriever


def get_rag_service() -> RAGService:
    """Tải service một lần sau retriever."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(get_retriever())
    return _rag_service


def get_access_context(request: Request) -> AccessContext:
    """Lấy identity và ACL từ server-side API key."""
    try:
        return access_context_from_api_key(request.headers.get("X-API-Key"))
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401, detail="X-API-Key không hợp lệ hoặc bị thiếu."
        ) from exc


@app.get("/health", response_model=dict[str, Any], summary="Kiểm tra API và artifact")
def health() -> dict[str, Any]:
    """Trả trạng thái đơn giản, không làm lộ đường dẫn máy chủ."""
    return _artifact_health()


@app.post("/query", response_model=QueryOut, summary="Tra cứu chính sách nội bộ")
def query(payload: QueryIn, request: Request) -> QueryOut:
    """Nhận câu hỏi và tạo AccessContext từ API key, không tin ACL trong body."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    try:
        result = get_rag_service().answer(
            QueryRequest(
                question=payload.question,
                access_context=get_access_context(request),
                request_id=request_id,
            )
        )
        return QueryOut(
            request_id=result.request_id,
            mode=result.mode,
            answer=result.answer,
            citations=[CitationItem(**citation) for citation in result.citations],
            sources=[SourceItem(**source) for source in result.sources],
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[req=%s] Query thất bại: %s", request_id, exc)
        raise HTTPException(status_code=500, detail="Lỗi nội bộ dịch vụ RAG.") from exc


@app.post("/debug/retrieve", response_model=dict[str, Any], include_in_schema=False)
def debug_retrieve(payload: DebugRetrieveIn, request: Request) -> dict[str, Any]:
    """Debug retrieval ngoài production bằng AccessContext của admin."""
    if os.getenv("ENV", "development").lower() == "production":
        raise HTTPException(status_code=404, detail="Endpoint không tồn tại trong production.")
    if not is_admin_api_key(request.headers.get("X-API-Key")):
        raise HTTPException(status_code=403, detail="Chỉ admin mới được dùng endpoint debug.")
    context = get_access_context(request)
    retriever = get_retriever()
    hits = retriever.search(
        payload.question,
        k=payload.top_k,
        candidate_k=payload.candidate_k,
        min_score=payload.min_score,
        use_dense=payload.use_dense,
        use_bm25=payload.use_bm25,
        use_reranker=payload.use_reranker,
        access_context=context,
    )
    return {"question": payload.question, "count": len(hits), "hits": hits}
