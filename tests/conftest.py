from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if config.option.basetemp is not None:
        return

    repository_root = Path(__file__).resolve().parents[1]
    private_temp_root = repository_root / ".scratch" / "pytest"
    private_temp_root.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(private_temp_root / f"run-{uuid4().hex}")
