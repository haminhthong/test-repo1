"""Kiểm thử các invariant P0 của Enterprise RAG V1."""

from __future__ import annotations

from datetime import date

import pytest

from src.api import QueryIn
from src.catalog import CatalogError, active_catalog, load_catalog
from src.generation import validate_citation_references
from src.security import AccessContext, AuthenticationError, access_context_from_api_key


def test_catalog_excludes_expired_version() -> None:
    entries = load_catalog("configs/knowledge_catalog.yaml", data_dir="data/raw")
    active = active_catalog(entries, date(2026, 9, 7))
    assert all(entry.status == "ACTIVE" for entry in active)
    assert all(entry.file != "annual_leave_policy_2025.txt" for entry in active)


def test_catalog_rejects_two_active_versions(tmp_path) -> None:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "documents:\n"
        "  - {document_id: a, policy_key: p, file: a.txt, department: hr, version: '1', effective_from: '2026-01-01', effective_to: null, status: ACTIVE, allowed_groups: [employee]}\n"
        "  - {document_id: b, policy_key: p, file: b.txt, department: hr, version: '2', effective_from: '2026-02-01', effective_to: null, status: ACTIVE, allowed_groups: [employee]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="nhiều active version"):
        load_catalog(catalog)


def test_catalog_rejects_invalid_effective_interval(tmp_path) -> None:
    """Catalog không được chứa khoảng hiệu lực rỗng hoặc ngược."""
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "documents:\n"
        "  - {document_id: a, policy_key: p, file: a.txt, department: hr, version: '1', "
        "effective_from: '2026-01-01', effective_to: '2026-01-01', status: ACTIVE, "
        "allowed_groups: [employee]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="effective_to phải sau"):
        load_catalog(catalog)


def test_catalog_rejects_path_traversal(tmp_path) -> None:
    """Catalog không được trỏ file ra ngoài thư mục dữ liệu."""
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "documents:\n"
        "  - {document_id: a, policy_key: p, file: ../secret.txt, department: hr, "
        "version: '1', effective_from: '2026-01-01', status: ACTIVE, "
        "allowed_groups: [employee]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="không được trỏ ra ngoài"):
        load_catalog(catalog)


def test_catalog_rejects_absolute_posix_path(tmp_path) -> None:
    """Catalog không được biến đường dẫn tuyệt đối thành đường dẫn tương đối."""
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "documents:\n"
        "  - {document_id: a, policy_key: p, file: /outside/secret.txt, department: hr, "
        "version: '1', effective_from: '2026-01-01', status: ACTIVE, "
        "allowed_groups: [employee]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="không được trỏ ra ngoài"):
        load_catalog(catalog)


def test_catalog_rejects_unsafe_acl_group(tmp_path) -> None:
    """Tên ACL không được trở thành đường dẫn shard ngoài release."""
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "documents:\n"
        "  - {document_id: a, policy_key: p, file: a.txt, department: hr, "
        "version: '1', effective_from: '2026-01-01', status: ACTIVE, "
        "allowed_groups: ['../admin']}\n",
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="tên nhóm an toàn"):
        load_catalog(catalog)


def test_query_body_does_not_create_access_groups() -> None:
    payload = QueryIn(question="Câu hỏi hợp lệ", user_groups=["security"])
    assert not hasattr(payload, "user_groups")


def test_api_key_mapping_is_server_side() -> None:
    context = access_context_from_api_key(
        "key",
        env={
            "RAG_API_KEYS_JSON": '{"key": {"user_id": "u1", "groups": ["hr"]}}',
            "ENV": "production",
        },
    )
    assert context == AccessContext(user_id="u1", groups=("hr",))


def test_explicit_empty_environment_does_not_fallback_to_process_env(monkeypatch) -> None:
    """Test isolation không được vô tình đọc key từ environment của process."""
    monkeypatch.setenv("RAG_API_KEY", "process-key")
    with pytest.raises(AuthenticationError):
        access_context_from_api_key("process-key", env={})


def test_citation_coverage_is_sentence_level() -> None:
    hits = [{"source": "policy.txt", "text": "12 ngày phép"}]
    _, valid, metadata = validate_citation_references(
        "Có 12 ngày phép [C1]. Câu này không có nguồn.", hits
    )
    assert valid is False
    assert metadata["citation_reference_validity"] is True
    assert metadata["factual_sentence_citation_coverage"] == 0.5
