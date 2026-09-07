"""Smoke test kiểm tra import module, health check endpoints, citation validation và bảo mật ACL."""

from __future__ import annotations

import json

from src.api import INDEX_DIR, QueryIn, health, health_live
from src.generation import (
    format_context_for_prompt,
    validate_citation_references,
    validate_citations,
)


def test_health_reports_sanitized_status():
    """Kiểm tra hàm health() trả về trạng thái hợp lệ và KHÔNG làm lộ đường dẫn ổ đĩa."""
    response = health()
    assert response["status"] in {"ok", "degraded"}
    assert isinstance(response["index_ready"], bool)
    assert "artifacts_directory" not in response


def test_health_live_endpoint():
    """Kiểm tra liveness endpoint."""
    res = health_live()
    assert res["status"] == "alive"
    assert "timestamp" in res


def test_index_config_has_reproducibility_metadata():
    """Kiểm tra tệp config.json nếu index đã được xây dựng."""
    config_file = INDEX_DIR / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert config["schema_version"] in {1, 2, 3}
        assert config["chunk_count"] >= 0
        assert config["embedding_model"]
        assert 0 <= config["overlap_words"] < config["chunk_words"]


def test_citation_validation_logic():
    """Kiểm tra CitationValidator bóc tách đúng thẻ [C1] và gắn trích đoạn minh chứng."""
    hits = [
        {
            "source": "policy_leave.txt",
            "page": None,
            "section": "Nghỉ phép",
            "text": "12 ngày phép năm",
        },
        {
            "source": "security_guide.txt",
            "page": 1,
            "section": "Bảo mật",
            "text": "Bắt buộc bật 2FA",
        },
    ]
    answer = "Nhân viên có 12 ngày phép [C1] và bắt buộc 2FA [C2]."

    citations, is_clean = validate_citations(answer, hits)
    assert is_clean is True
    assert len(citations) == 2
    assert citations[0]["id"] == "C1"
    assert citations[0]["document"] == "policy_leave.txt"
    assert citations[1]["id"] == "C2"
    assert citations[1]["document"] == "security_guide.txt"


def test_missing_citation_is_not_auto_attached():
    """Tuyệt đối KHÔNG tự ý gán [C1] giả nếu câu trả lời không có thẻ trích dẫn."""
    hits = [
        {"source": "policy_leave.txt", "text": "Công ty hỗ trợ 12 ngày phép."},
    ]
    # LLM trả lời nhưng không gắn trích dẫn
    raw_answer = "Công ty hỗ trợ 20 triệu đồng mỗi tháng."

    citations, _is_clean, meta = validate_citation_references(raw_answer, hits)
    assert citations == []
    assert meta["citation_coverage"] == 0.0
    assert meta["citation_valid"] is False


def test_invalid_citation_causes_validation_failure():
    """Trích dẫn ảo [C99] vượt quá phạm vi evidence phải bị phát hiện và đánh cờ không hợp lệ."""
    hits = [
        {"source": "policy_leave.txt", "text": "Quy chế nghỉ phép năm."},
    ]
    hallucinated_answer = "Theo quy định [C99], nhân viên được nghỉ 30 ngày."

    _citations, is_clean, meta = validate_citation_references(hallucinated_answer, hits)
    assert is_clean is False
    assert meta["has_invalid_citation"] is True
    assert meta["citation_valid"] is False


def test_client_cannot_override_production_retrieval_policy():
    """Production QueryIn chỉ chấp nhận question và bỏ qua quyền do client tự gửi."""
    payload_data = {
        "question": "Quy định công tác phí thế nào?",
        "user_groups": ["employee"],
    }
    query_obj = QueryIn(**payload_data)
    assert query_obj.question == "Quy định công tác phí thế nào?"

    # Client cố tình truyền min_score=0.0 hoặc use_reranker=False sẽ bị Pydantic bỏ qua/không nhận
    bad_payload = {
        "question": "Thử nghiệm hack",
        "min_score": 0.0,
        "use_reranker": False,
    }
    # QueryIn không khai báo các trường này, nên không thể can thiệp vào pipeline
    parsed = QueryIn(**bad_payload)
    assert not hasattr(parsed, "min_score")
    assert not hasattr(parsed, "use_reranker")


def test_context_respects_token_budget():
    """Context Builder phải tuân thủ ngân sách token, không nhồi nhét quá mức."""
    hits = [
        {"source": "doc1.txt", "text": "Từ " * 200},
        {"source": "doc2.txt", "text": "Từ " * 200},
        {"source": "doc3.txt", "text": "Từ " * 200},
    ]
    # Ngân sách 300 tokens chỉ đủ chứa tối đa 1 chunk (~260 tokens)
    formatted = format_context_for_prompt(hits, max_chunks=3, max_tokens=300)
    assert '<evidence id="C1"' in formatted
    assert '<evidence id="C3"' not in formatted


def test_unauthorized_document_never_reaches_retrieval():
    """Tài liệu thuộc nhóm quyền restricted/confidential không bao giờ lọt vào ứng viên khi user chỉ có quyền employee."""
    from unittest.mock import MagicMock

    import numpy as np

    from src.retrieval import Retriever

    retriever = MagicMock(spec=Retriever)
    retriever.chunks = [
        {
            "chunk_id": "c1",
            "text": "Quy chế an ninh",
            "source": "security.txt",
            "security_scope": ["public", "employee"],
        },
        {
            "chunk_id": "c2",
            "text": "Bảng lương tuyệt mật",
            "source": "salary.txt",
            "security_scope": ["confidential", "hr"],
        },
    ]
    retriever.default_candidate_k = 10
    retriever.rrf_k = 60
    retriever.evidence_gate_threshold = 0.25
    retriever.bm25_index = MagicMock()
    retriever.bm25_index.search.return_value = [(0, 5.0), (1, 6.0)]
    retriever.encoder = MagicMock()
    retriever.encoder.encode.return_value = np.zeros((1, 384), dtype="float32")
    retriever.index = MagicMock()
    retriever.index.search.return_value = (
        np.array([[0.8, 0.9]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )
    retriever.reranker = MagicMock()
    retriever.reranker.rerank.side_effect = lambda q, pool, top_k: pool[:top_k]

    results = Retriever.search(retriever, "bảng lương tuyệt mật", k=2, user_groups=["employee"])
    retrieved_ids = [r["chunk_id"] for r in results]
    assert "c2" not in retrieved_ids, (
        "LỖI BẢO MẬT ACL: Tài liệu bí mật lọt vào retrieval của user không có thẩm quyền!"
    )
