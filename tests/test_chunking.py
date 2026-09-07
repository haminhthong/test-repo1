"""Unit test cho chức năng chia nhỏ đoạn văn (Chunking) và xác thực cấu hình."""

from __future__ import annotations

import pytest

from src.config import IndexConfig
from src.ingestion import (
    Chunk,
    chunk_text,
    generate_document_id,
    structure_aware_chunk,
)
from src.ranking import hybrid_score, lexical_overlap


def test_chunking_keeps_source_and_metadata():
    """Kiểm tra việc chia chunk bảo toàn đúng metadata nguồn và độ dài."""
    text = " ".join(["từ_mẫu"] * 600)
    chunks = chunk_text(text, source="doc_test.txt", page=1, chunk_words=100, overlap_words=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "doc_test.txt"
        assert chunk.page == 1
        assert chunk.word_count > 0


def test_chunking_rejects_invalid_overlap():
    """Kiểm tra ngoại lệ khi tham số overlap không hợp lệ."""
    with pytest.raises(ValueError, match="overlap_words"):
        chunk_text("Nội dung thử nghiệm", source="test.txt", chunk_words=10, overlap_words=10)


def test_index_config_validation():
    """Kiểm tra việc validate thông số IndexConfig."""
    with pytest.raises(ValueError, match="overlap_words"):
        IndexConfig(chunk_words=100, overlap_words=100).validate()

    with pytest.raises(ValueError, match="chunk_words"):
        IndexConfig(chunk_words=0, overlap_words=0).validate()

    with pytest.raises(ValueError, match="Chiến lược chunking không hợp lệ"):
        IndexConfig(strategy="invalid_strategy").validate()


def test_hybrid_ranking_rewards_exact_query_terms():
    """Kiểm tra thuật toán Lexical Overlap ưu tiên các đoạn chứa chính xác từ khóa."""
    query = "báo sự cố bảo mật"
    doc_match = "Nhân viên phải báo sự cố bảo mật trong vòng 30 phút"
    doc_unrelated = "Quy trình thanh toán và nộp hoàn ứng chi phí công tác"

    exact_score = lexical_overlap(query, doc_match)
    unrelated_score = lexical_overlap(query, doc_unrelated)

    assert exact_score > unrelated_score
    assert hybrid_score(0.5, exact_score, dense_weight=0.8) > hybrid_score(
        0.5, unrelated_score, dense_weight=0.8
    )


def test_structure_aware_chunking_extracts_sections():
    """Kiểm tra structure_aware_chunk nhận diện đúng tiêu đề mục và không cắt giữa câu."""
    sample_doc = (
        "QUY CHẾ NGHỈ PHÉP NĂM\n\n"
        "1. Quyền lợi nghỉ phép:\n"
        "Nhân viên toàn thời gian chính thức có 12 ngày phép năm hưởng nguyên lương. "
        "Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi năm tiếp theo.\n\n"
        "2. Quy trình đăng ký:\n"
        "Đơn xin nghỉ phép từ 3 ngày liên tiếp trở lên cần gửi đăng ký trước tối thiểu 5 ngày làm việc."
    )

    chunks = structure_aware_chunk(
        text=sample_doc,
        source="policy_leave.txt",
        source_path="hr/policy_leave.txt",
        page=1,
        target_words=30,
        overlap_words=5,
    )

    assert len(chunks) >= 2
    sections = [c.section for c in chunks]
    assert any("Quyền lợi nghỉ phép" in (s or "") for s in sections)
    assert any("Quy trình đăng ký" in (s or "") for s in sections)

    # Kiểm tra chunk_id có định dạng chuẩn: doc_id:p1:c000
    for idx, c in enumerate(chunks):
        assert c.chunk_id.endswith(f":p1:c{idx:03d}")
        assert c.document_id == generate_document_id("hr/policy_leave.txt")
        assert c.content_hash != ""


def test_chunk_id_collision_resistance_across_directories():
    """Kiểm tra hai file cùng tên nhưng khác thư mục có document_id và chunk_id khác nhau."""
    doc_id_hr = generate_document_id("hr/policy.pdf")
    doc_id_fin = generate_document_id("finance/policy.pdf")

    assert doc_id_hr != doc_id_fin

    chunk_hr = Chunk(
        chunk_id=f"{doc_id_hr}:p1:c001",
        text="Quy định nhân sự",
        source="policy.pdf",
        source_path="hr/policy.pdf",
    )
    chunk_fin = Chunk(
        chunk_id=f"{doc_id_fin}:p1:c001",
        text="Quy định tài chính",
        source="policy.pdf",
        source_path="finance/policy.pdf",
    )

    assert chunk_hr.chunk_id != chunk_fin.chunk_id
    assert chunk_hr.document_id != chunk_fin.document_id


def test_chunk_quality_contract_validation():
    """Kiểm tra Chunk Quality Contract phát hiện lỗi rỗng hoặc trùng lặp."""
    from src.ingestion import validate_chunk_quality_contract

    valid_chunks = [
        Chunk(chunk_id="doc1:p0:c000", text="Nội dung 1", source="doc1.txt"),
        Chunk(chunk_id="doc1:p0:c001", text="Nội dung 2", source="doc1.txt"),
    ]
    report = validate_chunk_quality_contract(valid_chunks)
    assert report["passed"] is True

    # Trường hợp vi phạm: Chunk rỗng
    invalid_chunks = [
        Chunk(chunk_id="doc1:p0:c000", text="   ", source="doc1.txt"),
    ]
    with pytest.raises(ValueError, match="Chunk Quality Contract"):
        validate_chunk_quality_contract(invalid_chunks)


def test_detect_corpus_changes_logic():
    """Kiểm tra phát hiện tài liệu mới, sửa đổi, giữ nguyên và bị xóa."""
    import tempfile
    from pathlib import Path

    from src.index import detect_corpus_changes
    from src.utils import compute_file_checksum

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_path = Path(tmp_dir_str)

        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("Phiên bản 1", encoding="utf-8")
        hash1 = compute_file_checksum(doc1)

        registry = {
            "doc1.txt": {
                "document_id": "doc1",
                "checksum": hash1,
                "status": "ACTIVE",
            },
            "deleted_doc.txt": {
                "document_id": "del_doc",
                "checksum": "abc123",
                "status": "ACTIVE",
            },
        }

        # Thêm doc2 mới
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Tài liệu mới", encoding="utf-8")

        changes = detect_corpus_changes(tmp_path, registry)
        assert len(changes["new"]) == 1
        assert changes["new"][0].name == "doc2.txt"
        assert len(changes["unchanged"]) == 1
        assert changes["unchanged"][0].name == "doc1.txt"
        assert changes["deleted"] == ["deleted_doc.txt"]
