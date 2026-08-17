from __future__ import annotations

import subprocess

from scripts.audit_public_repo import audit


def test_public_repo_audit_distinguishes_empty_and_nonempty_credentials(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    config = tmp_path / "settings.ini"
    credential_key = "Security" + "Token"
    config.write_text(f"{credential_key}=\nbEnabled=False\n", encoding="utf-8")
    subprocess.run(["git", "add", "settings.ini"], cwd=tmp_path, check=True)

    empty_report = audit(tmp_path)
    assert empty_report["pass"] is True

    fake_value = "not-a-real-" + "token"
    config.write_text(f"{credential_key}={fake_value}\n", encoding="utf-8")
    nonempty_report = audit(tmp_path)
    assert nonempty_report["pass"] is False
    assert nonempty_report["violations"] == [
        {"path": "settings.ini", "kind": "credential_like_assignment", "line": 1}
    ]
