from __future__ import annotations

from pathlib import Path


def test_default_pytest_temp_path_stays_in_private_scratch(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected_temp_root = (repository_root / ".scratch" / "pytest").resolve()

    assert tmp_path.resolve().is_relative_to(expected_temp_root)
