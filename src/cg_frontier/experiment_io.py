"""Small, experiment-agnostic I/O contracts shared by reproducible workflows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_file_inventory(
    roots: Path | str | Mapping[str, Path | str],
) -> dict[str, object]:
    """Describe files below one root, or labelled roots with physical deduplication."""

    labelled = isinstance(roots, Mapping)
    source_roots = roots.items() if labelled else ((None, roots),)
    files: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root_name, root in source_roots:
        source_root = Path(root)
        for path in sorted(source_root.rglob("*")):
            resolved = path.resolve()
            if not path.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            entry: dict[str, object] = {
                "path": path.relative_to(source_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if root_name is not None:
                entry = {"root": root_name, **entry}
            files.append(entry)
    if not files:
        raise ValueError("file inventory would be empty")
    return {
        "files": files,
        "file_count": len(files),
        "payload_bytes": sum(int(item["bytes"]) for item in files),
    }


def is_finite_tree(value: Any) -> bool:
    """Return whether a JSON-like value contains only finite numeric leaves."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(is_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(is_finite_tree(item) for item in value)
    return False


def resolve_within(root: Path | str, path: Path | str) -> Path:
    """Resolve ``path`` and fail closed unless it remains inside ``root``."""

    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"path escapes root: {path}")
    return resolved_path


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a file using bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_with_sha256(path: Path | str, value: Any) -> str:
    """Write canonical JSON plus a sibling ``.sha256`` sidecar and return its digest."""

    destination = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii", newline="\n"
    )
    return digest
