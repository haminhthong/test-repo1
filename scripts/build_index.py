"""CLI xây dựng artifact Dense/BM25 hiện hành."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import parse_args
from src.index import build_index


def _configure_console_utf8() -> None:
    """Bảo đảm trợ giúp CLI hiển thị đúng tiếng Việt trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    _configure_console_utf8()
    build_index(parse_args())
