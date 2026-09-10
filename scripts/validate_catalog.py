"""Validate catalog và coverage của raw corpus trước khi build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog import load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra Knowledge Catalog")
    parser.add_argument("--catalog", default="configs/knowledge_catalog.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    entries = load_catalog(args.catalog, data_dir=args.data_dir)
    print(f"Catalog hợp lệ: {len(entries)} records.")


if __name__ == "__main__":
    main()
