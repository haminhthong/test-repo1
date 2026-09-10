"""Các tiện ích ghi log, đọc/lưu JSON và tính checksum cho dự án RAG."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Khởi tạo Logger chung cho dự án
LOGGER = logging.getLogger("rag_knowledge_assistant")


def setup_logging(level_name: str | None = None) -> None:
    """Cấu hình định dạng và cấp độ log, đồng thời bật UTF-8 trên Windows.

    Tham số:
        level_name (str | None): Cấp độ log (DEBUG, INFO, WARNING, ERROR).
            Nếu bỏ trống, lấy từ biến môi trường LOG_LEVEL (mặc định INFO).
    """
    # Tự động cấu hình sys.stdout và sys.stderr về UTF-8 trên Windows console
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError) as exc:
            LOGGER.debug("Không thể đổi encoding console sang UTF-8: %s", exc)

    if level_name is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_json(path: str | Path, payload: Any) -> None:
    """Lưu dữ liệu dưới dạng tệp JSON bằng Unicode UTF-8.

    Tham số:
        path (str hoặc Path): Đường dẫn tệp cần lưu.
        payload (Any): Dữ liệu Python cần ghi.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    """Đọc và giải mã dữ liệu từ tệp JSON.

    Tham số:
        path (str hoặc Path): Đường dẫn tệp JSON.

    Kết quả trả về:
        Any: Dữ liệu cấu trúc thu được từ tệp JSON.

    Ngoại lệ:
        FileNotFoundError: Nếu tệp không tồn tại.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp JSON tại: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def compute_file_checksum(file_path: str | Path) -> str:
    """Tính mã băm MD5 để nhận diện thay đổi, không dùng cho mục đích bảo mật.

    Tham số:
        file_path (str hoặc Path): Đường dẫn tới tệp tài liệu.

    Kết quả trả về:
        str: Chuỗi mã băm MD5 dạng thập lục phân của tệp.
    """
    hasher = hashlib.md5(usedforsecurity=False)
    path = Path(file_path)
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
