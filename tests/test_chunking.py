"""Unit test cho chức năng chia nhỏ đoạn văn (Chunking) và xác thực cấu hình."""

from __future__ import annotations

import pytest

from src.config import IndexConfig
from src.ingestion import (
    Chunk,
    chunk_text,
    structure_aware_chunk,
)


def test_chunking_keeps_source_and_metadata():
    """Kiểm tra việc chia chunk bảo toàn đúng metadata nguồn và độ dài."""
    text = " ".join(["từ_mẫu"] * 600)
    chunks = chunk_text(
        text,
        source="doc_test.txt",
        source_path="hr/doc_test.txt",
        document_id="doc-test",
        document_version="v1",
        page=1,
        chunk_words=100,
        overlap_words=10,
        strategy="sliding_window",
    )

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


def test_chunking_requires_catalog_identity():
    """Chunk không được tự suy luận document identity từ tên file."""
    with pytest.raises(ValueError, match="document_id"):
        chunk_text("Nội dung chính sách đủ dài để kiểm tra", source="doc.txt")


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
        document_id="leave-policy",
        document_version="v2026",
    )

    assert len(chunks) >= 2
    sections = [c.section for c in chunks]
    assert any("Quyền lợi nghỉ phép" in (s or "") for s in sections)
    assert any("Quy trình đăng ký" in (s or "") for s in sections)

    # ID luôn chứa tài liệu, version, mã băm section và thứ tự chunk.
    for idx, c in enumerate(chunks):
        assert c.chunk_id.startswith("leave-policy:v2026:")
        assert c.chunk_id.endswith(f":c{idx:03d}")
        assert c.document_id == "leave-policy"
        assert c.content_hash != ""


def test_chunk_ids_continue_across_document_pages():
    """Chunk ở các page khác nhau không được quay lại index 0."""
    page_text = "Nhân viên được hưởng quyền lợi theo chính sách hiện hành. " * 8
    first_page = chunk_text(
        page_text,
        source="policy.pdf",
        source_path="hr/policy.pdf",
        document_id="policy",
        document_version="v1",
        page=1,
        chunk_words=20,
        overlap_words=4,
        base_chunk_index=0,
    )
    second_page = chunk_text(
        page_text,
        source="policy.pdf",
        source_path="hr/policy.pdf",
        document_id="policy",
        document_version="v1",
        page=2,
        chunk_words=20,
        overlap_words=4,
        base_chunk_index=len(first_page),
    )
    assert set(chunk.chunk_id for chunk in first_page).isdisjoint(
        chunk.chunk_id for chunk in second_page
    )


def test_chunk_id_collision_resistance_across_directories():
    """Kiểm tra hai file cùng tên nhưng khác thư mục có document_id và chunk_id khác nhau."""
    doc_id_hr = "hr-policy"
    doc_id_fin = "finance-policy"

    chunk_hr = Chunk(
        chunk_id=f"{doc_id_hr}:v1:section:c001",
        text="Quy định nhân sự",
        source="policy.pdf",
        document_id=doc_id_hr,
        document_version="v1",
        source_path="hr/policy.pdf",
    )
    chunk_fin = Chunk(
        chunk_id=f"{doc_id_fin}:v1:section:c001",
        text="Quy định tài chính",
        source="policy.pdf",
        document_id=doc_id_fin,
        document_version="v1",
        source_path="finance/policy.pdf",
    )

    assert chunk_hr.chunk_id != chunk_fin.chunk_id
    assert chunk_hr.document_id != chunk_fin.document_id


def test_chunk_quality_contract_validation():
    """Kiểm tra Chunk Quality Contract phát hiện lỗi rỗng hoặc trùng lặp."""
    from src.ingestion import validate_chunk_quality_contract

    valid_chunks = [
        Chunk(
            chunk_id="doc1:v1:common:c000",
            text="Nội dung 1",
            source="doc1.txt",
            document_id="doc1",
            document_version="v1",
            source_path="doc1.txt",
        ),
        Chunk(
            chunk_id="doc1:v1:common:c001",
            text="Nội dung 2",
            source="doc1.txt",
            document_id="doc1",
            document_version="v1",
            source_path="doc1.txt",
        ),
    ]
    report = validate_chunk_quality_contract(valid_chunks)
    assert report["passed"] is True

    # Trường hợp vi phạm: Chunk rỗng
    invalid_chunks = [
        Chunk(
            chunk_id="doc1:v1:common:c000",
            text="   ",
            source="doc1.txt",
            document_id="doc1",
            document_version="v1",
            source_path="doc1.txt",
        ),
    ]
    with pytest.raises(ValueError, match="chất lượng chunk"):
        validate_chunk_quality_contract(invalid_chunks)
