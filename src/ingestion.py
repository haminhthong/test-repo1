"""Xử lý nạp và trích xuất tài liệu (Data Ingestion & Structure-Aware Chunking Pipeline).

Tệp này hỗ trợ đọc các định dạng tài liệu nội bộ phổ biến (TXT, Markdown, PDF, DOCX),
giữ lại thông tin nguồn (Source Metadata) và số trang (Page Number khi có).
Đồng thời cung cấp cả hai cơ chế phân tách:
1. Phân tách theo cấu trúc (Structure-Aware Chunking): Nhận diện tiêu đề/section, chia đoạn
   theo ranh giới câu (Sentence Boundaries), gán metadata phân tầng và định danh ổn định không xung đột.
2. Phân tách cửa sổ trượt (Sliding Window Chunking): Hỗ trợ so sánh thực nghiệm (Ablation).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import compute_file_checksum

LOGGER = logging.getLogger("rag_knowledge_assistant.ingestion")

# Regex nhận diện tiêu đề (Heading / Section Header) tiếng Việt và Markdown
# Lưu ý: Pattern heading CHỈ kích hoạt khi dòng đứng riêng lẻ (đầu/cuối là khoảng trắng hoặc
# xuống dòng). Điều này tránh match nhầm các cụm từ viết hoa ngắn nằm trong câu văn.
HEADING_REGEX = re.compile(
    r"(?:^|\n)\s*(?:"
    r"#{1,6}\s+.+|"  # Markdown headings (# Tiêu đề, ## Mục 1)
    r"(?:\d+\.|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b\.?)\s+[A-ZÀ-Ỹ].+|"  # 1. Mục, I. Phần (viết hoa đầu)
    r"Điều\s+\d+[:.]?\s*.+|"  # Điều 1: ..., Điều 2.
    r"CHƯƠNG\s+[IVXLCDM\d]+[:.]?\s*.+|"  # CHƯƠNG I, CHƯƠNG 2
    r"[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9 \-_:]{3,79}$"  # Dòng viết hoa ngắn (CHÍNH SÁCH NGHỈ PHÉP)
    r")\s*(?=\n|$)",
    flags=re.UNICODE,
)

# Regex tách câu dựa trên dấu kết thúc (. ? !) theo sau bởi khoảng trắng và ký tự viết hoa
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9])", flags=re.UNICODE)


def generate_document_id(relative_path: Path | str) -> str:
    """Tạo mã định danh duy nhất ổn định (Stable Document ID) từ đường dẫn tương đối.

    Sử dụng 12 ký tự đầu của mã băm SHA-256 để chống xung đột tên file giữa các thư mục khác nhau.

    Args:
        relative_path (Union[Path, str]): Đường dẫn tương đối của tệp tài liệu.

    Returns:
        str: Chuỗi 12 ký tự hex định danh tài liệu.
    """
    normalized_path = str(relative_path).replace("\\", "/").strip().lower()
    return hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:12]


from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Cấu trúc dữ liệu đại diện cho một đoạn văn bản (Chunk) sau khi phân tách.

    Attributes:
        chunk_id (str): Mã định danh ổn định không xung đột: {document_id}:p{page}:c{index}.
        text (str): Nội dung văn bản của chunk.
        source (str): Tên tệp tài liệu gốc.
        page (Optional[int]): Số trang tương ứng (có với PDF kỹ thuật số, None với DOCX/TXT/MD).
        checksum (Optional[str]): Mã băm MD5 của tệp nguồn phục vụ kiểm tra tính toàn vẹn (Provenance).
        word_count (int): Số lượng từ trong chunk.
        document_id (str): Mã hash định danh duy nhất của tài liệu.
        source_path (str): Đường dẫn tương đối của tệp nguồn trong kho lưu trữ.
        section (Optional[str]): Tiêu đề phần/chương/mục chứa chunk.
        chunk_index (int): Chỉ số thứ tự của chunk trong tài liệu (0-indexed).
        content_hash (str): Mã băm SHA-256 của nội dung chunk.
        security_scope (List[str]): Danh sách nhóm người dùng được phép truy cập chunk này (ACL).
    """

    chunk_id: str
    text: str
    source: str
    page: int | None = None
    checksum: str | None = None
    word_count: int = 0
    document_id: str = ""
    source_path: str = ""
    section: str | None = None
    chunk_index: int = 0
    content_hash: str = ""
    security_scope: list[str] = field(default_factory=lambda: ["public"])

    def __post_init__(self) -> None:
        """Tự động tính toán các trường metadata nếu chưa được cung cấp."""
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.content_hash and self.text:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        if not self.source_path and self.source:
            self.source_path = self.source
        if not self.document_id and self.source_path:
            self.document_id = generate_document_id(self.source_path)
        if not self.security_scope or self.security_scope == ["public"]:
            scopes = ["public", "employee"]
            lowered = (self.source_path or self.source).lower()
            if "hr" in lowered or "leave" in lowered:
                scopes.append("hr")
            if "finance" in lowered or "expense" in lowered or "reimbursement" in lowered or "procurement" in lowered:
                scopes.append("finance")
            if "security" in lowered or "password" in lowered or "incident" in lowered:
                scopes.append("security")
            self.security_scope = scopes


def check_document_quality(path: Path, raw_pages: list[tuple[str, int | None]]) -> tuple[bool, str]:
    """Kiểm tra chất lượng văn bản trích xuất (Document Quality Gate)."""
    if not raw_pages:
        return False, "EMPTY_EXTRACTION"
    total_chars = sum(len(p[0].strip()) for p in raw_pages)
    if total_chars < 10:
        return False, f"INSUFFICIENT_CHARACTERS_{total_chars}"
    empty_pages = sum(1 for p in raw_pages if not p[0].strip())
    empty_ratio = empty_pages / len(raw_pages)
    if len(raw_pages) > 2 and empty_ratio > 0.8:
        return False, f"HIGH_EMPTY_PAGE_RATIO_{empty_ratio:.2f}_POSSIBLE_SCANNED_PDF"
    return True, "QUALITY_PASSED"


def validate_chunk_quality_contract(chunks: list[Chunk]) -> dict[str, Any]:
    """Kiểm tra hợp đồng chất lượng của tập chunk trước khi đưa vào Indexing (Chunk Quality Contract)."""
    empty_chunks = [c for c in chunks if not c.text.strip()]
    chunk_ids = [c.chunk_id for c in chunks]
    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
    content_hashes = [c.content_hash for c in chunks]
    duplicate_content = len(content_hashes) - len(set(content_hashes))

    report = {
        "total_chunks": len(chunks),
        "empty_chunks": len(empty_chunks),
        "duplicate_chunk_ids": duplicate_ids,
        "duplicate_content": duplicate_content,
        "passed": len(empty_chunks) == 0 and duplicate_ids == 0,
    }
    if not report["passed"]:
        raise ValueError(f"Chunk Quality Contract vi phạm tiêu chuẩn: {report}")
    return report



def read_text(path: Path) -> list[tuple[str, int | None]]:
    """Đọc nội dung văn bản từ tệp tài liệu dựa trên phần mở rộng file.

    Hỗ trợ các định dạng: .txt, .md, .pdf, .docx.
    - PDF: Trích xuất trang và số trang từ PDF kỹ thuật số (không hỗ trợ OCR scanned PDF).
    - DOCX: Trích xuất toàn bộ đoạn văn bản; số trang là None vì DOCX là flow layout không có trang cố định.
    - TXT/MD: Đọc toàn văn UTF-8; số trang là None.

    Args:
        path (Path): Đường dẫn tới tệp tài liệu cần đọc.

    Returns:
        List[Tuple[str, Optional[int]]]: Danh sách các cặp (Nội dung văn bản, Số trang khi khả dụng).

    Raises:
        ValueError: Nếu định dạng tệp không nằm trong danh sách hỗ trợ.
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(text, None)]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages_content: list[tuple[str, int | None]] = []
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                pages_content.append((extracted, idx + 1))
            return pages_content
        except ImportError:
            LOGGER.warning(
                "Thư viện 'pypdf' chưa được cài đặt. Không thể xử lý file PDF: %s",
                path.name,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Lỗi khi đọc file PDF %s: %s", path.name, exc)
            return []

    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [(full_text, None)]
        except ImportError:
            LOGGER.warning(
                "Thư viện 'python-docx' chưa được cài đặt. Không thể xử lý file DOCX: %s",
                path.name,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Lỗi khi đọc file DOCX %s: %s", path.name, exc)
            return []

    raise ValueError(
        f"Định dạng tệp không được hỗ trợ: '{suffix}' (Chỉ hỗ trợ .txt, .md, .pdf, .docx)"
    )


def split_into_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách câu văn hoàn chỉnh dựa trên ranh giới ngữ pháp.

    Args:
        text (str): Đoạn văn bản đầu vào.

    Returns:
        List[str]: Danh sách các câu văn không rỗng.
    """
    raw_sentences = SENTENCE_SPLIT_REGEX.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def structure_aware_chunk(
    text: str,
    source: str,
    source_path: str = "",
    page: int | None = None,
    target_words: int = 250,
    overlap_words: int = 40,
    checksum: str | None = None,
    document_id: str | None = None,
    base_chunk_index: int = 0,
) -> list[Chunk]:
    """Phân tách văn bản dựa trên cấu trúc tài liệu (Tiêu đề, Đoạn văn, Ranh giới câu).

    Quy trình:
    1. Nhận diện các Section Header bằng Regular Expression.
    2. Tách từng Section thành các đoạn văn (Paragraphs) và câu (Sentences).
    3. Gom các câu văn vào Chunk cho đến khi đạt ngưỡng target_words mà không cắt đứt câu.
    4. Giữ lại câu gối đầu (Sentence-level Overlap) giữa hai chunk liên tiếp.
    5. Gán đầy đủ metadata (section, document_id, chunk_id, content_hash).

    Args:
        text (str): Nội dung văn bản cần phân tách.
        source (str): Tên file nguồn.
        source_path (str): Đường dẫn tương đối của file nguồn.
        page (Optional[int]): Số trang nếu có (PDF).
        target_words (int): Kích thước từ mục tiêu mỗi chunk (mặc định: 250).
        overlap_words (int): Số từ gối đầu mục tiêu (mặc định: 40).
        checksum (Optional[str]): Mã hash MD5 của file nguồn.
        document_id (Optional[str]): Mã định danh ổn định của tài liệu.
        base_chunk_index (int): Chỉ số bắt đầu cho chunk_index.

    Returns:
        List[Chunk]: Danh sách các chunk có cấu trúc ngữ nghĩa hoàn chỉnh.
    """
    if target_words <= 0:
        raise ValueError("target_words phải lớn hơn 0")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError(
            f"overlap_words={overlap_words} phải nằm trong khoảng [0, target_words={target_words})"
        )

    clean_text = text.strip()
    if not clean_text:
        return []

    doc_path = source_path or source
    doc_id = document_id or generate_document_id(doc_path)
    page_str = str(page) if page is not None else "0"

    # Phân rã văn bản theo dòng để nhận diện các section
    lines = clean_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_section = "Nội dung chung"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Dùng re.search vì HEADING_REGEX đã được neo bằng ^\s* hoặc \n\s*
        if HEADING_REGEX.search(stripped):
            if current_lines:
                sections.append((current_section, current_lines))
                current_lines = []
            current_section = stripped
        else:
            current_lines.append(stripped)

    if current_lines:
        sections.append((current_section, current_lines))

    chunks: list[Chunk] = []
    chunk_counter = base_chunk_index

    for sec_title, sec_lines in sections:
        sec_text = " ".join(sec_lines)
        sentences = split_into_sentences(sec_text)
        if not sentences:
            continue

        buffer_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            sent_words = len(sentence.split())
            if current_word_count + sent_words > target_words and buffer_sentences:
                # Đóng gói chunk hiện tại
                chunk_content = " ".join(buffer_sentences)
                # Thêm tiền tố tiêu đề mục để bảo toàn ngữ cảnh cục bộ
                if sec_title != "Nội dung chung" and not chunk_content.startswith(sec_title):
                    chunk_text_with_context = f"{sec_title}: {chunk_content}"
                else:
                    chunk_text_with_context = chunk_content

                chunk_id = f"{doc_id}:p{page_str}:c{chunk_counter:03d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text_with_context,
                        source=source,
                        page=page,
                        checksum=checksum,
                        word_count=len(chunk_text_with_context.split()),
                        document_id=doc_id,
                        source_path=doc_path,
                        section=sec_title,
                        chunk_index=chunk_counter,
                    )
                )
                chunk_counter += 1

                # Giữ lại overlap theo câu
                overlap_buffer: list[str] = []
                overlap_wc = 0
                for s in reversed(buffer_sentences):
                    swc = len(s.split())
                    if overlap_wc + swc <= overlap_words or not overlap_buffer:
                        overlap_buffer.insert(0, s)
                        overlap_wc += swc
                    else:
                        break
                buffer_sentences = overlap_buffer
                current_word_count = sum(len(s.split()) for s in buffer_sentences)

            buffer_sentences.append(sentence)
            current_word_count += sent_words

        # Đóng gói phần dư còn lại trong section
        if buffer_sentences:
            chunk_content = " ".join(buffer_sentences)
            if len(chunk_content.split()) >= 15 or not chunks:
                if sec_title != "Nội dung chung" and not chunk_content.startswith(sec_title):
                    chunk_text_with_context = f"{sec_title}: {chunk_content}"
                else:
                    chunk_text_with_context = chunk_content

                chunk_id = f"{doc_id}:p{page_str}:c{chunk_counter:03d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text_with_context,
                        source=source,
                        page=page,
                        checksum=checksum,
                        word_count=len(chunk_text_with_context.split()),
                        document_id=doc_id,
                        source_path=doc_path,
                        section=sec_title,
                        chunk_index=chunk_counter,
                    )
                )
                chunk_counter += 1

    return chunks


def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_words: int = 220,
    overlap_words: int = 30,
    checksum: str | None = None,
    strategy: str = "sliding_window",
    source_path: str = "",
) -> list[Chunk]:
    """Phân tách văn bản theo cấu hình được chỉ định (Sliding Window hoặc Structure-Aware).

    Args:
        text (str): Nội dung văn bản cần phân tách.
        source (str): Tên file nguồn.
        page (Optional[int]): Số trang (nếu có).
        chunk_words (int): Kích thước từ tối đa mỗi chunk.
        overlap_words (int): Số từ chồng lấp giữa các chunk liên tiếp.
        checksum (Optional[str]): Mã hash của tệp nguồn.
        strategy (str): Chiến lược chunking ("sliding_window" hoặc "structure_aware").
        source_path (str): Đường dẫn tương đối của tệp nguồn.

    Returns:
        List[Chunk]: Danh sách các đối tượng Chunk đã tạo.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words phải lớn hơn 0")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError(
            f"overlap_words={overlap_words} phải nằm trong khoảng [0, chunk_words={chunk_words})"
        )

    if strategy == "structure_aware":
        return structure_aware_chunk(
            text=text,
            source=source,
            source_path=source_path,
            page=page,
            target_words=chunk_words,
            overlap_words=overlap_words,
            checksum=checksum,
        )

    # Chiến lược Sliding Window cổ điển (dùng cho baseline / ablation)
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_words - overlap_words)
    doc_path = source_path or source
    doc_id = generate_document_id(doc_path)
    page_str = str(page) if page is not None else "0"

    for i, start_idx in enumerate(range(0, len(words), step)):
        part_words = words[start_idx : start_idx + chunk_words]
        if len(part_words) < 15 and len(words) > 15:
            continue

        chunk_text_str = " ".join(part_words)
        chunk_id = f"{doc_id}:p{page_str}:c{i:03d}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                source=source,
                page=page,
                checksum=checksum,
                word_count=len(part_words),
                document_id=doc_id,
                source_path=doc_path,
                chunk_index=i,
            )
        )

    return chunks


def create_document_manifest(
    chunks: list[Chunk],
    data_dir: str | Path,
) -> dict[str, Any]:
    """Tạo tài liệu kê khai chỉ mục (Document Manifest) phục vụ truy xuất nguồn gốc (Provenance).

    Args:
        chunks (List[Chunk]): Danh sách toàn bộ các chunk đã tạo.
        data_dir (Union[str, Path]): Thư mục dữ liệu gốc.

    Returns:
        Dict[str, Any]: Manifest theo dõi file hash, chunk counts và chunk IDs.
    """
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "data_dir": str(data_dir),
        "total_documents": 0,
        "total_chunks": len(chunks),
        "documents": {},
    }

    doc_map: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_key = chunk.source_path or chunk.source
        if doc_key not in doc_map:
            doc_map[doc_key] = {
                "document_id": chunk.document_id,
                "file_name": chunk.source,
                "checksum": chunk.checksum,
                "chunk_count": 0,
                "chunk_ids": [],
                "pages": set(),
                "status": "INDEXED",
            }
        doc_map[doc_key]["chunk_count"] += 1
        doc_map[doc_key]["chunk_ids"].append(chunk.chunk_id)
        if chunk.page is not None:
            doc_map[doc_key]["pages"].add(chunk.page)

    for item in doc_map.values():
        item["pages"] = sorted(list(item["pages"]))

    manifest["total_documents"] = len(doc_map)
    manifest["documents"] = doc_map
    return manifest


def ingest_folder(
    folder: str | Path,
    chunk_words: int = 250,
    overlap_words: int = 40,
    strategy: str = "structure_aware",
) -> list[Chunk]:
    """Quét toàn bộ thư mục dữ liệu và chuyển đổi tất cả tài liệu thành danh sách Chunks.

    Args:
        folder (Union[str, Path]): Thư mục chứa các tệp tài liệu nội bộ.
        chunk_words (int): Kích thước chunk mục tiêu tính theo số từ.
        overlap_words (int): Độ chồng lấp tính theo số từ.
        strategy (str): Chiến lược chunking ("structure_aware" hoặc "sliding_window").

    Returns:
        List[Chunk]: Danh sách toàn bộ các chunk thu thập được từ tất cả tài liệu.

    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
    """
    data_path = Path(folder)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: '{data_path.resolve()}'"
        )

    all_chunks: list[Chunk] = []
    file_paths = sorted([p for p in data_path.rglob("*") if p.is_file()])

    for path in file_paths:
        try:
            relative_path = path.relative_to(data_path).as_posix()
            file_hash = compute_file_checksum(path)
            doc_pages = read_text(path)

            doc_id = generate_document_id(relative_path)
            doc_chunk_counter = 0

            for content, page_num in doc_pages:
                if not content.strip():
                    continue
                if strategy == "structure_aware":
                    page_chunks = structure_aware_chunk(
                        text=content,
                        source=path.name,
                        source_path=relative_path,
                        page=page_num,
                        target_words=chunk_words,
                        overlap_words=overlap_words,
                        checksum=file_hash,
                        document_id=doc_id,
                        base_chunk_index=doc_chunk_counter,
                    )
                else:
                    page_chunks = chunk_text(
                        text=content,
                        source=path.name,
                        page=page_num,
                        chunk_words=chunk_words,
                        overlap_words=overlap_words,
                        checksum=file_hash,
                        strategy="sliding_window",
                        source_path=relative_path,
                    )

                doc_chunk_counter += len(page_chunks)
                all_chunks.extend(page_chunks)
        except ValueError as ve:
            LOGGER.debug("Bỏ qua tệp %s: %s", path.name, ve)
            continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Lỗi không xác định khi xử lý tệp %s: %s", path.name, exc)
            continue

    LOGGER.info(
        "Đã ingest thành công %d chunks từ %d tệp tài liệu (strategy='%s').",
        len(all_chunks),
        len(file_paths),
        strategy,
    )
    return all_chunks
