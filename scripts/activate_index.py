"""Kiểm tra active pointer sau khi build; build_index đã activate nguyên tử."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect active index pointer")
    parser.add_argument("--model-dir", default="models/rag_index")
    args = parser.parse_args()
    pointer = Path(args.model_dir) / "active_index.json"
    if not pointer.exists():
        raise SystemExit(f"Chưa có active index: {pointer}")
    print(json.dumps(json.loads(pointer.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
