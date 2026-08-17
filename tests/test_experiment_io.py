from __future__ import annotations

import hashlib
import json
import math

import pytest

from cg_frontier.experiment_io import (
    build_file_inventory,
    is_finite_tree,
    resolve_within,
    sha256_file,
    write_json_with_sha256,
)


def test_build_file_inventory_describes_a_single_tree_deterministically(tmp_path):
    root = tmp_path / "result"
    (root / "nested").mkdir(parents=True)
    (root / "z.bin").write_bytes(b"zzz")
    (root / "nested" / "a.bin").write_bytes(b"a")

    inventory = build_file_inventory(root)

    assert inventory == {
        "files": [
            {
                "path": "nested/a.bin",
                "bytes": 1,
                "sha256": hashlib.sha256(b"a").hexdigest(),
            },
            {
                "path": "z.bin",
                "bytes": 3,
                "sha256": hashlib.sha256(b"zzz").hexdigest(),
            },
        ],
        "file_count": 2,
        "payload_bytes": 4,
    }


def test_build_file_inventory_labels_multiple_roots_and_deduplicates_files(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.bin").write_bytes(b"a")
    (second / "b.bin").write_bytes(b"bb")

    inventory = build_file_inventory(
        {"first": first, "duplicate": first, "second": second}
    )

    assert inventory["files"] == [
        {
            "root": "first",
            "path": "a.bin",
            "bytes": 1,
            "sha256": hashlib.sha256(b"a").hexdigest(),
        },
        {
            "root": "second",
            "path": "b.bin",
            "bytes": 2,
            "sha256": hashlib.sha256(b"bb").hexdigest(),
        },
    ]
    assert inventory["file_count"] == 2
    assert inventory["payload_bytes"] == 3


def test_build_file_inventory_rejects_an_empty_result_tree(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="file inventory would be empty"):
        build_file_inventory(root)


def test_write_json_with_sha256_is_deterministic_and_creates_parent(tmp_path):
    path = tmp_path / "nested" / "report.json"

    digest = write_json_with_sha256(path, {"z": 1, "a": [2, 3]})

    expected = (json.dumps({"z": 1, "a": [2, 3]}, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert path.with_suffix(".json.sha256").read_text(encoding="ascii") == digest + "\n"


def test_sha256_file_hashes_file_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    payload = (b"c4-render-ablation\0" * 100_000) + b"tail"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_resolve_within_accepts_descendants_and_rejects_escape(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()

    assert resolve_within(root, root / "outputs" / "result.json") == (
        root / "outputs" / "result.json"
    ).resolve()
    with pytest.raises(ValueError, match="escapes root"):
        resolve_within(root, root / ".." / "outside.json")


def test_is_finite_tree_accepts_metadata_and_rejects_nested_nonfinite_values():
    assert is_finite_tree(
        {"status": "complete", "enabled": True, "metric": 0.25, "rows": [1, None]}
    )
    assert not is_finite_tree({"metrics": [{"loss": math.nan}]})
    assert not is_finite_tree({"metrics": [{"loss": math.inf}]})
