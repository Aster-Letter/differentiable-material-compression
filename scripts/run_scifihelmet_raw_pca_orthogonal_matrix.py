"""Serial single-GPU supervisor for the four raw-PCA O1/O2 10k candidates."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_orthogonal_10k_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/scifihelmet_c4_affine_v1/raw_pca_orthogonal/bf215f-matrix-v1"
ORDER = ("O1-r025", "O1-r050", "O2-r025", "O2-r050")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_alive(pid: int) -> bool:
    synchronize = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
    if not handle:
        return False
    try:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _verified_report(config, candidate_id: str) -> dict:
    root = ROOT / config["candidates"][candidate_id]["output_root"]
    path = root / "training_report.json"
    if not path.is_file():
        raise RuntimeError(f"{candidate_id} did not produce training_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_10k" or report.get("steps") != 10000:
        raise RuntimeError(f"{candidate_id} report is not a complete 10k endpoint")
    checkpoint = root / "checkpoints/step_10000/checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{candidate_id} is missing its 10k checkpoint")
    return {
        "candidate_id": candidate_id,
        "report": path.relative_to(ROOT).as_posix(),
        "report_sha256": _sha256(path),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "wall_seconds": report["wall_seconds"],
    }


def run(config_path: Path, output: Path, *, wait_pid: int | None, completed_current: str | None) -> dict:
    config = yaml.safe_load(config_path.read_bytes())
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matrix root: {output}")
    output.mkdir(parents=True)
    rows = []
    if wait_pid is not None:
        if completed_current not in ORDER:
            raise ValueError("--completed-current must identify the waited candidate")
        while _process_alive(wait_pid):
            time.sleep(10.0)
        rows.append(_verified_report(config, completed_current))
    start_index = ORDER.index(completed_current) + 1 if completed_current else 0
    runner = ROOT / "scripts/train_scifihelmet_c4_affine_raw_pca_orthogonal_10k.py"
    for candidate_id in ORDER[start_index:]:
        stdout_path = output / f"{candidate_id.lower()}.log"
        stderr_path = output / f"{candidate_id.lower()}.err.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(runner),
                    "--config",
                    str(config_path),
                    "--candidate",
                    candidate_id,
                ],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{candidate_id} exited {result.returncode}; see {stderr_path}")
        rows.append(_verified_report(config, candidate_id))
    report = {
        "schema_version": 1,
        "status": "complete_four_candidate_10k_matrix",
        "order": list(ORDER),
        "candidates": rows,
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "matrix_report.json").write_bytes(payload)
    (output / "matrix_report.json.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--completed-current")
    args = parser.parse_args()
    report = run(
        args.config.resolve(),
        args.output_root.resolve(),
        wait_pid=args.wait_pid,
        completed_current=args.completed_current,
    )
    print(json.dumps({"status": report["status"], "candidates": len(report["candidates"])}))


if __name__ == "__main__":
    main()
