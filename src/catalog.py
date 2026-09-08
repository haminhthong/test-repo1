"""Knowledge Catalog: nguồn sự thật cho document governance.

ACL, policy version và thời gian hiệu lực phải đến từ catalog đã được kiểm
duyệt. Module này cố ý không suy luận quyền từ tên file.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
VALID_STATUSES = {"ACTIVE", "ARCHIVED", "RETIRED", "DRAFT"}
GROUP_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$", flags=re.IGNORECASE)


class CatalogError(ValueError):
    """Lỗi hợp đồng catalog hoặc governance metadata."""


@dataclass(frozen=True)
class CatalogEntry:
    """Metadata được phê duyệt cho một phiên bản tài liệu."""

    document_id: str
    policy_key: str
    file: str
    department: str
    version: str
    effective_from: date
    effective_to: date | None
    status: str
    allowed_groups: tuple[str, ...]

    @property
    def policy_status(self) -> str:
        """Trả về status dạng chữ thường dùng trong metadata của chunk."""
        return self.status.lower()

    def is_active(self, as_of: date) -> bool:
        """Kiểm tra phiên bản có hiệu lực tại một ngày cụ thể hay không."""
        return (
            self.status == "ACTIVE"
            and self.effective_from <= as_of
            and (self.effective_to is None or as_of < self.effective_to)
        )

    def to_dict(self) -> dict[str, Any]:
        """Chuyển metadata sang JSON-serializable dictionary."""
        data = asdict(self)
        data["effective_from"] = self.effective_from.isoformat()
        data["effective_to"] = (
            self.effective_to.isoformat() if self.effective_to is not None else None
        )
        data["allowed_groups"] = list(self.allowed_groups)
        data["policy_status"] = self.policy_status
        return data


def _parse_date(value: Any, field_name: str, index: int) -> date:
    if not isinstance(value, str):
        raise CatalogError(f"Catalog record #{index}: {field_name} phải là YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CatalogError(
            f"Catalog record #{index}: {field_name} không hợp lệ: {value!r}."
        ) from exc


def _normalise_entry(raw: dict[str, Any], index: int) -> CatalogEntry:
    if not isinstance(raw, dict):
        raise CatalogError(f"Catalog record #{index} phải là object.")

    required = (
        "document_id",
        "policy_key",
        "file",
        "department",
        "version",
        "effective_from",
        "status",
        "allowed_groups",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise CatalogError(f"Catalog record #{index} thiếu trường: {', '.join(missing)}.")

    groups = raw["allowed_groups"]
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, str) or not group.strip() for group in groups)
    ):
        raise CatalogError(f"Catalog record #{index}: allowed_groups phải là danh sách không rỗng.")
    normalized_groups = tuple(dict.fromkeys(group.strip().lower() for group in groups))
    if any(not GROUP_PATTERN.fullmatch(group) for group in normalized_groups):
        raise CatalogError(
            f"Catalog record #{index}: allowed_groups chỉ được chứa tên nhóm an toàn."
        )

    status = str(raw["status"]).strip().upper()
    if status not in VALID_STATUSES:
        raise CatalogError(
            f"Catalog record #{index}: status {status!r} không thuộc {sorted(VALID_STATUSES)}."
        )

    raw_file = raw["file"]
    if not isinstance(raw_file, str):
        raise CatalogError(f"Catalog record #{index}: file phải là chuỗi đường dẫn.")
    file_path = raw_file.replace("\\", "/").strip()
    normalized_file = Path(file_path)
    if (
        not file_path
        or file_path.startswith("/")
        or normalized_file.is_absolute()
        or ".." in normalized_file.parts
    ):
        raise CatalogError(f"Catalog record #{index}: file không được trỏ ra ngoài data_dir.")
    file_path = normalized_file.as_posix()
    if Path(file_path).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CatalogError(
            f"Catalog record #{index}: file phải là định dạng được hỗ trợ: {file_path!r}."
        )

    effective_to_raw = raw.get("effective_to")
    effective_to = (
        None
        if effective_to_raw in (None, "")
        else _parse_date(effective_to_raw, "effective_to", index)
    )
    effective_from = _parse_date(raw["effective_from"], "effective_from", index)
    if effective_to is not None and effective_to <= effective_from:
        raise CatalogError(f"Catalog record #{index}: effective_to phải sau effective_from.")

    entry = CatalogEntry(
        document_id=str(raw["document_id"]).strip(),
        policy_key=str(raw["policy_key"]).strip(),
        file=file_path,
        department=str(raw["department"]).strip().lower(),
        version=str(raw["version"]).strip(),
        effective_from=effective_from,
        effective_to=effective_to,
        status=status,
        allowed_groups=normalized_groups,
    )
    if not entry.document_id or not entry.policy_key or not entry.version or not entry.department:
        raise CatalogError(f"Catalog record #{index}: metadata định danh không được rỗng.")
    if not entry.allowed_groups:
        raise CatalogError(f"Catalog record #{index}: ACL rỗng là không hợp lệ.")
    return entry


def load_catalog(path: str | Path, data_dir: str | Path | None = None) -> list[CatalogEntry]:
    """Đọc, chuẩn hóa và kiểm tra catalog.

    Khi truyền ``data_dir``, module kiểm tra cả file tồn tại và đảm bảo mọi tài
    liệu trong thư mục dữ liệu đều có record catalog tương ứng.
    """
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Không tìm thấy knowledge catalog: {catalog_path.resolve()}")

    try:
        import yaml
    except ImportError as exc:
        raise CatalogError("Cần cài PyYAML để đọc knowledge_catalog.yaml.") from exc

    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    raw_documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(raw_documents, list) or not raw_documents:
        raise CatalogError("Catalog phải có danh sách documents không rỗng.")

    entries = [_normalise_entry(item, index) for index, item in enumerate(raw_documents, start=1)]
    _validate_catalog_invariants(entries, data_dir)
    return entries


def _validate_catalog_invariants(
    entries: list[CatalogEntry], data_dir: str | Path | None = None
) -> None:
    """Bảo vệ ID, version, overlap hiệu lực và coverage file của catalog."""
    ids = [entry.document_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise CatalogError("document_id bị trùng trong catalog.")

    seen_policy_versions: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.policy_key, entry.version)
        if key in seen_policy_versions:
            raise CatalogError(f"Trùng policy_key/version: {entry.policy_key}/{entry.version}.")
        seen_policy_versions.add(key)

    by_policy: dict[str, list[CatalogEntry]] = {}
    for entry in entries:
        by_policy.setdefault(entry.policy_key, []).append(entry)
    for policy_key, versions in by_policy.items():
        active_versions = [entry for entry in versions if entry.status == "ACTIVE"]
        if len(active_versions) > 1:
            raise CatalogError(
                f"Có nhiều active version cho policy_key={policy_key}; "
                "chỉ một version được phép ở trạng thái ACTIVE."
            )

    if data_dir is None:
        return

    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_path.resolve()}")
    for entry in entries:
        resolved_file = (data_path / entry.file).resolve()
        if not resolved_file.is_relative_to(data_path):
            raise CatalogError(f"Catalog file trỏ ra ngoài data_dir: {entry.file}")
    files = {
        file.relative_to(data_path).as_posix()
        for file in data_path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
    }
    catalog_files = {entry.file for entry in entries}
    missing_catalog = sorted(files - catalog_files)
    missing_files = sorted(catalog_files - files)
    if missing_catalog:
        raise CatalogError("Tài liệu không có catalog entry: " + ", ".join(missing_catalog))
    if missing_files:
        raise CatalogError("Catalog trỏ tới file không tồn tại: " + ", ".join(missing_files))


def active_catalog(entries: list[CatalogEntry], as_of: date | None = None) -> list[CatalogEntry]:
    """Lọc duy nhất các tài liệu ACTIVE và đang có hiệu lực."""
    effective_date = as_of or date.today()
    return [entry for entry in entries if entry.is_active(effective_date)]


def catalog_snapshot(entries: list[CatalogEntry]) -> list[dict[str, Any]]:
    """Tạo snapshot bất biến để audit cùng index release."""
    return [entry.to_dict() for entry in sorted(entries, key=lambda item: item.document_id)]
