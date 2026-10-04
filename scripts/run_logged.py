#!/usr/bin/env python3
"""Run an argv command without a shell and retain a self-contained run record."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

WORKSPACE = Path(__file__).resolve().parents[1]
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".lock"}
EXCLUDED = {".git", ".venv", "__pycache__", "data", "assets", "cache", "tmp", "runs"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path, base: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(base)), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="lowercase run slug")
    parser.add_argument("--config", help="JSON object or path to a JSON object file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.name) or not command:
        parser.error("use a lowercase name slug and provide -- <command...>")
    try:
        config_text = args.config or "{}"
        config = json.loads(config_text if config_text.lstrip().startswith("{") else Path(config_text).read_text())
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")
        json.dumps(config, allow_nan=False)
    except (OSError, ValueError) as error:
        parser.error(f"invalid --config: {error}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.name}-{uuid.uuid4().hex[:8]}"
    run_dir = WORKSPACE / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [part.replace("{run_dir}", str(run_dir)) for part in command]
    environment = os.environ.copy()
    environment["POET_RUN_DIR"] = str(run_dir)
    manifest = {"schema_version": 1, "run_id": run_id, "state": "running", "started_at_utc": utc(),
                "ended_at_utc": None, "returncode": None, "command": command, "cwd": str(WORKSPACE),
                "location": "local_workspace",
                "wrapper_pid": os.getpid(), "child_pid": None, "signal": None,
                "environment": {key: environment[key] for key in ENV_KEYS if key in environment},
                "config": "config.json", "code_snapshot": "code", "log": "run.log", "metrics": "metrics.json"}
    write_json(run_dir / "manifest.json", manifest)
    print(f"RUN_DIR={run_dir}", flush=True)
    started, child, interrupted, returncode = time.monotonic(), None, None, None

    def forward_signal(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        manifest["signal"] = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)  # Only the new session created for this child.
            except ProcessLookupError:
                pass

    handlers = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            try:
                write_json(run_dir / "config.json", config)
                sources = []
                for directory in (WORKSPACE / "scripts", WORKSPACE / "vendor" / "poet"):
                    if not directory.is_dir():
                        if directory == WORKSPACE / "vendor" / "poet":
                            continue  # Optional for the classical-only quickstart.
                        raise FileNotFoundError(f"Code snapshot directory missing: {directory}")
                    for path in sorted(directory.rglob("*")):
                        relative = path.relative_to(WORKSPACE)
                        if path.is_symlink() or not path.is_file() or EXCLUDED.intersection(relative.parts):
                            continue
                        if path.suffix not in SOURCE_SUFFIXES and path.name not in {"LICENSE", "COPYING", "Dockerfile", "Makefile"}:
                            continue
                        destination = run_dir / "code" / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)
                        sources.append(fingerprint(destination, run_dir / "code"))
                write_json(run_dir / "code_manifest.json", {"files": sources})
                manifest["code_manifest_sha256"] = fingerprint(run_dir / "code_manifest.json", run_dir)["sha256"]
                manifest["config_sha256"] = fingerprint(run_dir / "config.json", run_dir)["sha256"]
                if not interrupted:
                    child = subprocess.Popen(command, cwd=WORKSPACE, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    manifest["child_pid"] = child.pid
                    write_json(run_dir / "manifest.json", manifest)
                    if interrupted:
                        forward_signal(interrupted, None)
                    returncode = child.wait()
            except Exception as error:
                manifest["error"] = {"type": type(error).__name__, "message": str(error)}
                log.write(f"\nWRAPPER ERROR: {type(error).__name__}: {error}\n")
                if child is not None and child.poll() is None:
                    forward_signal(signal.SIGTERM, None)
                    returncode = child.wait()
        manifest.update({"state": "interrupted" if interrupted else ("completed" if returncode == 0 else "failed"),
                         "ended_at_utc": utc(), "returncode": returncode, "signal": interrupted})
        write_json(run_dir / "metrics.json", {"purpose": "engineering_only", "wall_time_seconds": time.monotonic() - started, "returncode": returncode})
        (run_dir / "exit_code.txt").write_text(f"{returncode}\n" if returncode is not None else "not_started\n")
        artifacts = [fingerprint(path, run_dir) for path in sorted(run_dir.rglob("*"))
                     if path.is_file() and not path.is_symlink() and path.relative_to(run_dir).as_posix() not in {"manifest.json", "artifacts.json"}]
        write_json(run_dir / "artifacts.json", {"files": artifacts, "excludes": ["manifest.json", "artifacts.json"]})
        manifest["artifacts"] = "artifacts.json"
        write_json(run_dir / "manifest.json", manifest)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f"STATE={manifest['state']} RETURN_CODE={returncode}", flush=True)
    return 128 + interrupted if interrupted else (returncode if returncode is not None and returncode >= 0 else (128 - returncode if returncode is not None else 1))


if __name__ == "__main__":
    sys.exit(main())
