from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PRIVATE_PREFIXES = (
    ".agents/",
    ".private/",
    "docs-agent/",
    "docs/archive/",
    "docs/course-delivery/",
    "docs/obsidian/",
    "docs/obsidian-portfolio-demo/",
    "outputs/",
    "tmp/",
    "transfers/",
)
PRIVATE_FILES = {"AGENTS.md"}
UE_BINARY_SUFFIXES = {".uasset", ".umap"}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".uproject",
    ".yaml",
    ".yml",
}
MAX_TRACKED_BYTES = 20 * 1024 * 1024


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def content_patterns() -> list[tuple[str, re.Pattern[str]]]:
    slash_drive = re.compile(r"\b[A-Za-z]:[/\\]")
    home_path = re.compile(r"[/\\]Users[/\\][^/\\\s]+", re.IGNORECASE)
    credential = re.compile(
        r"(?:api[_-]?key|client[_-]?secret|password|securitytoken)[ \t]*[:=][ \t]*[^\s\"']+",
        re.IGNORECASE,
    )
    student_id = re.compile(r"\bPB\d{8}\b", re.IGNORECASE)
    course_person = re.compile(r"(?:\u53f2\u777f\u94ed|\u6768\u514b\u5fae)")
    private_contact = re.compile(
        r"\b[A-Z0-9._%+-]+@(?:qq\.com|corp\.netease\.com)\b",
        re.IGNORECASE,
    )
    return [
        ("absolute_windows_path", slash_drive),
        ("user_home_path", home_path),
        ("credential_like_assignment", credential),
        ("student_id", student_id),
        ("course_person_name", course_person),
        ("private_contact", private_contact),
    ]


def audit(root: Path) -> dict[str, object]:
    violations: list[dict[str, object]] = []
    files = tracked_files(root)
    self_path = Path(__file__).resolve()
    for relative in files:
        normalized = relative.replace("\\", "/")
        path = root / relative
        if normalized in PRIVATE_FILES or normalized.startswith(PRIVATE_PREFIXES):
            violations.append({"path": normalized, "kind": "private_path"})
        if path.suffix.lower() in UE_BINARY_SUFFIXES:
            violations.append({"path": normalized, "kind": "ue_binary"})
        if path.exists() and path.stat().st_size > MAX_TRACKED_BYTES:
            violations.append(
                {"path": normalized, "kind": "large_file", "bytes": path.stat().st_size}
            )
        if path.suffix.lower() not in TEXT_SUFFIXES or path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append({"path": normalized, "kind": "non_utf8_text"})
            continue
        for name, pattern in content_patterns():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                violations.append({"path": normalized, "kind": name, "line": line})
    return {
        "schema_version": 1,
        "tracked_file_count": len(files),
        "violations": violations,
        "pass": not violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the Git-tracked public repository surface")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit(args.root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
