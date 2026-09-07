"""Xác thực danh tính và tạo AccessContext tin cậy ở phía server."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class AuthenticationError(PermissionError):
    """API key thiếu hoặc không hợp lệ."""


@dataclass(frozen=True)
class AccessContext:
    """Ngữ cảnh quyền đã được server xác thực.

    ``groups`` là tuple bất biến để tầng retrieval không thể bị thay đổi sau
    khi xác thực. Không có group nghĩa là deny-by-default.
    """

    user_id: str
    groups: tuple[str, ...]
    auth_method: str = "api_key"

    def __post_init__(self) -> None:
        user_id = self.user_id.strip()
        groups = tuple(
            dict.fromkeys(group.strip().lower() for group in self.groups if group.strip())
        )
        if not user_id:
            raise ValueError("user_id không được rỗng.")
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "groups", groups)

    @classmethod
    def deny_all(cls, user_id: str = "anonymous", auth_method: str = "none") -> AccessContext:
        """Tạo context không có quyền để bảo đảm fail closed."""
        return cls(user_id=user_id, groups=(), auth_method=auth_method)


def _load_key_mapping(env: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Đọc map API key -> identity từ biến môi trường server-side."""
    env_map = env or os.environ
    raw = env_map.get("RAG_API_KEYS_JSON", "")
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuthenticationError("RAG_API_KEYS_JSON không phải JSON hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise AuthenticationError("RAG_API_KEYS_JSON phải là object.")
        return {
            str(api_key): value for api_key, value in payload.items() if isinstance(value, dict)
        }

    configured_key = env_map.get("RAG_API_KEY")
    if configured_key:
        return {
            configured_key: {
                "user_id": env_map.get("RAG_API_USER_ID", "configured-user"),
                "groups": _parse_groups(env_map.get("RAG_API_GROUPS", "employee")),
            }
        }

    # Key demo này vẫn là cấu hình server-side, không lấy từ request body.
    if env_map.get("ENV", "development").lower() != "production":
        return {"local-demo-employee": {"user_id": "u123", "groups": ["employee"]}}
    return {}


def _parse_groups(raw: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw, str):
        return [value.strip().lower() for value in raw.split(",") if value.strip()]
    return [str(value).strip().lower() for value in raw if str(value).strip()]


def access_context_from_api_key(
    api_key: str | None,
    env: Mapping[str, str] | None = None,
) -> AccessContext:
    """Đổi API key thành AccessContext; không bao giờ tin groups từ JSON request."""
    if not api_key:
        raise AuthenticationError("Thiếu X-API-Key.")

    record = _load_key_mapping(env).get(api_key)
    if not record:
        raise AuthenticationError("X-API-Key không hợp lệ.")

    user_id = str(record.get("user_id", "")).strip()
    groups = _parse_groups(record.get("groups", []))
    if not user_id or not groups:
        raise AuthenticationError("Identity mapping phải có user_id và groups.")
    return AccessContext(user_id=user_id, groups=tuple(groups))


def is_admin_api_key(api_key: str | None, env: Mapping[str, str] | None = None) -> bool:
    """Kiểm tra quyền admin cho endpoint debug bằng cấu hình server-side."""
    if not api_key:
        return False
    env_map = env or os.environ
    configured = env_map.get("RAG_ADMIN_API_KEY", "")
    return bool(configured and api_key == configured)
