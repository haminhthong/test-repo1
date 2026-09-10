from __future__ import annotations

from types import SimpleNamespace

from src.generation import ABSTAIN_PHRASE
from src.security import AccessContext
from src.service import QueryRequest, RAGService


def _hit() -> dict[str, object]:
    return {
        "chunk_id": "policy:v1:section:c000",
        "document_id": "policy",
        "document_version": "v1",
        "source": "policy.txt",
        "source_path": "policy.txt",
        "text": "Nhân viên chính thức được nghỉ 12 ngày phép năm.",
        "allowed_groups": ["employee"],
        "section": "Quyền lợi",
        "page": None,
    }


class _Retriever:
    def __init__(self, hits: list[dict[str, object]], reranker_mode: str) -> None:
        self.config = {"context_k": 4, "candidate_pool_k": 30}
        self.reranker = SimpleNamespace(mode=reranker_mode)
        self.hits = hits
        self.kwargs: dict[str, object] = {}

    def search(self, query: str, **kwargs: object) -> list[dict[str, object]]:
        self.kwargs = kwargs
        return [dict(hit) for hit in self.hits]


def test_reranker_unavailable_returns_sources_only_without_logit_filter() -> None:
    retriever = _Retriever([_hit()], reranker_mode="unavailable")
    result = RAGService(retriever).answer(
        QueryRequest(
            question="Có bao nhiêu ngày phép?",
            access_context=AccessContext(user_id="u1", groups=("employee",)),
        )
    )

    assert result.mode == "sources_only"
    assert result.citations
    assert result.answer != ABSTAIN_PHRASE
    assert retriever.kwargs["min_score"] == float("-inf")
