#!/usr/bin/env python3
"""Obtain the pinned PoET source; verify it before writing ignored vendor files.

No checkpoint, dataset, or Python package is downloaded. No source is executed.
Use --check PATH to verify an existing source directory without network access.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "upstream-manifest.json"
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024


def verify(directory: Path, manifest: dict) -> None:
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or nonregular source file: {filename}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {filename}")


def materialize(payload: bytes, directory: Path, manifest: dict) -> None:
    if hashlib.sha256(payload).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch; nothing was installed")
    expected = manifest["files"]
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or len(path.parts) != 2:
                raise ValueError("Unexpected archive layout or nonregular member")
            filename = path.parts[1]
            if filename not in expected or filename in found:
                raise ValueError(f"Unexpected or repeated file: {filename}")
            if member.size > MAX_ARCHIVE_BYTES:
                raise ValueError("Source member exceeds size limit")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable archive member")
            contents = stream.read(MAX_ARCHIVE_BYTES + 1)
            if hashlib.sha256(contents).hexdigest() != expected[filename]:
                raise ValueError(f"Source checksum mismatch: {filename}")
            found[filename] = contents
    if set(found) != set(expected):
        raise ValueError("Archive does not contain the complete expected source")
    for filename, contents in found.items():
        (directory / filename).write_bytes(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify existing source only; do not download")
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded official archive")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.check:
        if args.archive:
            parser.error("--check and --archive are mutually exclusive")
        verify(args.check, manifest)
        print(f"Verified {len(manifest['files'])} pinned PoET source files; no files changed.")
        return
    target = ROOT / "vendor" / "poet"
    if target.exists():
        verify(target, manifest)
        print("Existing vendor/poet source verified; nothing downloaded or overwritten.")
        return
    if args.archive:
        if args.archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds the size limit")
        payload = args.archive.read_bytes()
    else:
        commit = manifest["commit"]
        url = f"https://gitlab.lisn.upsaclay.fr/mtals/publications/poet/-/archive/{commit}/poet-{commit}.tar.gz"
        request = Request(url, headers={"User-Agent": "ASL-course-source-setup/1.0"})
        with urlopen(request, timeout=30) as response:
            payload = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds the size limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".poet-", dir=target.parent))
    try:
        materialize(payload, temporary, manifest)
        verify(temporary, manifest)
        if target.exists():
            raise FileExistsError("vendor/poet appeared during setup; it was not overwritten")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Verified {len(manifest['files'])} source files in Git-ignored vendor/poet/.")


if __name__ == "__main__":
    main()
