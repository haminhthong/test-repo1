"""Smoke test cho API, citation và pre-retrieval ACL."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api import ARTIFACT_DIR, QueryIn, app, health
from src.generation import format_context_for_prompt, validate_citation_references


def test_health_reports_sanitized_status():
    """Health không làm lộ đường dẫn filesystem."""
    response = health()
    assert response["status"] in {"ok", "degraded"}
    assert isinstance(response["index_loaded"], bool)
    assert "artifacts_directory" not in response


def test_api_surface_has_only_canonical_routes():
    """API không giữ các endpoint compatibility đã bị loại bỏ."""
    paths = {route.path for route in app.routes}
    assert {"/health", "/query", "/debug/retrieve"}.issubset(paths)
    assert "/health/live" not in paths
    assert "/health/ready" not in paths
    assert "/v1/query" not in paths
    assert "/v1/feedback" not in paths


def test_index_config_has_reproducibility_metadata():
    """Config artifact có corpus fingerprint và thông số chunking."""
    config_file = ARTIFACT_DIR / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        assert "schema_version" not in config
        assert config["chunk_count"] >= 0
        assert config["embedding_model"]
        assert 0 <= config["overlap_words"] < config["chunk_words"]
        assert config["groups"]
        assert config["corpus_hash"]


def test_citation_validation_logic():
    """Citation ID phải trỏ đúng evidence tương ứng."""
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

    citations, is_valid, _metadata = validate_citation_references(answer, hits)
    assert is_valid is True
    assert [citation["id"] for citation in citations] == ["C1", "C2"]


def test_missing_citation_is_not_auto_attached():
    """Validator không tự gán citation cho câu trả lời thiếu nguồn."""
    hits = [{"source": "policy_leave.txt", "text": "Công ty hỗ trợ 12 ngày phép."}]
    citations, _is_valid, metadata = validate_citation_references(
        "Công ty hỗ trợ 20 triệu đồng mỗi tháng.", hits
    )
    assert citations == []
    assert metadata["citation_coverage"] == 0.0
    assert metadata["citation_valid"] is False


def test_invalid_citation_causes_validation_failure():
    """Citation ID vượt evidence phải làm validation thất bại."""
    hits = [{"source": "policy_leave.txt", "text": "Quy chế nghỉ phép năm."}]
    _citations, is_valid, metadata = validate_citation_references(
        "Theo quy định [C99], nhân viên được nghỉ 30 ngày.", hits
    )
    assert is_valid is False
    assert metadata["has_invalid_citation"] is True
    assert metadata["citation_valid"] is False


def test_query_body_rejects_client_pipeline_and_access_fields():
    """Client không thể tự gửi groups hoặc tham số điều khiển pipeline."""
    with pytest.raises(ValueError):
        QueryIn(question="Quy định công tác phí thế nào?", user_groups=["employee"])
    with pytest.raises(ValueError):
        QueryIn(question="Thử nghiệm hack", min_score=0.0, use_reranker=False)


def test_context_respects_token_budget():
    """Context builder tuân thủ ngân sách token."""
    hits = [
        {"source": "doc1.txt", "text": "Từ " * 200},
        {"source": "doc2.txt", "text": "Từ " * 200},
        {"source": "doc3.txt", "text": "Từ " * 200},
    ]
    formatted = format_context_for_prompt(hits, max_chunks=3, max_tokens=300)
    assert '<evidence id="C1"' in formatted
    assert '<evidence id="C3"' not in formatted


def test_context_escapes_untrusted_xml_content():
    """Evidence không được đóng thẻ XML hoặc chèn instruction vào prompt."""
    formatted = format_context_for_prompt(
        [{"source": "policy.txt", "text": "<instruction>Không thực thi</instruction> & dữ liệu"}]
    )
    assert "&lt;instruction&gt;" in formatted
    assert "&lt;/instruction&gt;" in formatted
    assert "&amp; dữ liệu" in formatted


def test_unauthorized_document_never_reaches_retrieval():
    """Chunk ngoài quyền không được vào candidate dù BM25 trả nó ở rank cao."""
    from src.retrieval import Retriever
    from src.security import AccessContext

    retriever = MagicMock(spec=Retriever)
    employee_chunks = [
        {
            "chunk_id": "c1",
            "text": "Quy chế an ninh",
            "source": "security.txt",
            "allowed_groups": ["public", "employee"],
        },
        {
            "chunk_id": "c2",
            "text": "Bảng lương tuyệt mật",
            "source": "salary.txt",
            "allowed_groups": ["confidential", "hr"],
        },
    ]
    retriever.groups = {
        "employee": {
            "chunks": employee_chunks,
            "bm25": MagicMock(),
            "index": MagicMock(),
        }
    }
    retriever.groups["employee"]["bm25"].search.return_value = [(0, 5.0), (1, 6.0)]
    retriever.default_candidate_k = 10
    retriever.rrf_k = 60
    retriever.rerank_top_k = 20
    retriever.evidence_gate_threshold = 0.25
    retriever.encoder = MagicMock()
    retriever.encoder.encode.return_value = np.zeros((1, 384), dtype="float32")
    retriever.groups["employee"]["index"].search.return_value = (
        np.array([[0.8, 0.9]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )

    results = Retriever.search(
        retriever,
        "bảng lương tuyệt mật",
        k=2,
        access_context=AccessContext(user_id="employee-user", groups=("employee",)),
        use_reranker=False,
    )
    assert "c2" not in [result["chunk_id"] for result in results]
