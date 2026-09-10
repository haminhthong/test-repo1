"""Parser cho TXT, Markdown, PDF số và DOCX."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .models import DocumentBlock

LOGGER = logging.getLogger("rag_knowledge_assistant.ingestion")

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
HEADING_REGEX = re.compile(
    r"(?:^|\n)\s*(?:"
    r"#{1,6}\s+.+|"
    r"(?:\d+\.|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b\.?)\s+[A-ZÀ-Ỹ].+|"
    r"Điều\s+\d+[:.]?\s*.+|"
    r"CHƯƠNG\s+[IVXLCDM\d]+[:.]?\s*.+|"
    r"[A-ZÀ-Ỹ][A-ZÀ-Ỹ0-9 \-_:]{3,79}$"
    r")\s*(?=\n|$)",
    flags=re.UNICODE,
)


def check_document_quality(path: Path, raw_pages: list[tuple[str, int | None]]) -> tuple[bool, str]:
    """Loại file không đọc được hoặc có dấu hiệu là PDF scan."""
    if not raw_pages:
        return False, "EMPTY_EXTRACTION"
    total_chars = sum(len(text.strip()) for text, _ in raw_pages)
    if total_chars < 10:
        return False, f"INSUFFICIENT_CHARACTERS_{total_chars}"
    empty_pages = sum(1 for text, _ in raw_pages if not text.strip())
    if len(raw_pages) > 2 and empty_pages / len(raw_pages) > 0.8:
        return False, "HIGH_EMPTY_PAGE_RATIO_POSSIBLE_SCANNED_PDF"
    return True, "QUALITY_PASSED"


def read_document_blocks(path: Path) -> list[DocumentBlock]:
    """Đọc tài liệu thành các block vẫn giữ section, page và table row."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [DocumentBlock(text=text)] if text.strip() else []

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return [
                DocumentBlock(text=page.extract_text() or "", page=index + 1)
                for index, page in enumerate(PdfReader(str(path)).pages)
            ]
        except ImportError:
            LOGGER.warning("Thiếu pypdf, bỏ qua PDF: %s", path.name)
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Không đọc được PDF %s: %s", path.name, exc)
            return []

    if suffix == ".docx":
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            document = Document(str(path))
            parent: Any = document.element.body
            blocks: list[DocumentBlock] = []
            for child in parent.iterchildren():
                if child.tag.endswith("}p"):
                    paragraph = Paragraph(child, parent)
                    text = paragraph.text.strip()
                    if not text:
                        continue
                    style_name = str(getattr(paragraph.style, "name", ""))
                    heading_level: int | None = None
                    if style_name.lower().startswith("heading"):
                        match = re.search(r"(\d+)", style_name)
                        heading_level = int(match.group(1)) if match else 1
                    blocks.append(
                        DocumentBlock(
                            text=text,
                            block_type="heading" if heading_level else "paragraph",
                            heading_level=heading_level,
                        )
                    )
                elif child.tag.endswith("}tbl"):
                    table = Table(child, parent)
                    for row in table.rows:
                        values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if values:
                            blocks.append(
                                DocumentBlock(text=" | ".join(values), block_type="table_row")
                            )
            return blocks
        except ImportError:
            LOGGER.warning("Thiếu python-docx, bỏ qua DOCX: %s", path.name)
            return []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Không đọc được DOCX %s: %s", path.name, exc)
            return []

    raise ValueError(f"Định dạng không hỗ trợ: {suffix!r}.")


def read_text(path: Path) -> list[tuple[str, int | None]]:
    """Trả về nội dung tài liệu cùng page metadata nếu parser có thể cung cấp."""
    blocks = read_document_blocks(path)
    if path.suffix.lower() == ".pdf":
        return [(block.text, block.page) for block in blocks]
    text = "\n".join(block.text for block in blocks if block.text.strip())
    return [(text, None)] if text else []
