"""Điểm vào CLI xây dựng immutable full index release.

Chạy: python -m src.train --data-dir data/raw --model-dir models/rag_index
"""

from __future__ import annotations

from .config import parse_args
from .index import build_index


def main() -> None:
    """Đọc tham số CLI và kích hoạt quá trình index tài liệu."""
    config = parse_args()
    build_index(config)


if __name__ == "__main__":
    main()
