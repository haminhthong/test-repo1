"""Đọc và kiểm tra final locked-test artifact đã sinh từ Dev policy."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    report_path = Path("reports/final_test_metrics.json")
    if not report_path.exists():
        raise SystemExit("Chưa có final_test_metrics.json; hãy chạy evaluate_dev.py trước.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("locked_test_tuned"):
        raise SystemExit("Artifact không hợp lệ: locked test đã bị dùng để tune.")
    print(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
