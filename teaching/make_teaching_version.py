#!/usr/bin/env python3
"""Generate the members' teaching repository from this full (answer-key) repository.

Python files mark answer code with

    # BEGIN SOLUTION <tag>
    ...answer lines...
    # BEGIN STUDENT            (optional)
    # ...commented lines the members should see instead...
    # END SOLUTION

Without a STUDENT part the block becomes a "raise NotImplementedError" stub.
Markdown files use <!-- BEGIN SOLUTION <tag> --> ... <!-- END SOLUTION --> and the
block is removed. The tag "instructor" is never released.

Releasing answers (--release <tag>, e.g. after a milestone):
  * Markdown: the tag's blocks are shown in place (concept answers, results).
  * Python: the exercise files keep their TODO stubs (or the member's own code);
    the answer version is written next to them under answers/<same path>, so a
    member's work is never replaced.
Released tags are remembered in the output's TEACHING_BUILD.json and stay
released on later builds unless --reset-releases is given.

Usage:
    python teaching/make_teaching_version.py --output ../asl-team-teaching
    python teaching/make_teaching_version.py --output ../asl-team-teaching --release member1

Only files tracked or not ignored by git are copied; teaching/ and the
generator's own test are excluded. An existing output must come from an earlier
build. Files that changed there since the last build (for example a member's
merged implementation) and files the build did not create are kept and listed,
never overwritten, unless --overwrite-changed is given. The output's .git and
ignored local data are never touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ("teaching/", "tests/test_teaching_version.py")
TAGS = ("member1", "member2", "member3", "results-2026-10", "instructor")
NEVER_RELEASED = {"instructor"}
ANSWERS_DIR = "answers"
BUILD_RECORD = "TEACHING_BUILD.json"
PY_BEGIN = re.compile(r"^(\s*)# BEGIN SOLUTION (\S+)\s*$")
PY_STUDENT = re.compile(r"^\s*# BEGIN STUDENT\s*$")
PY_END = re.compile(r"^\s*# END SOLUTION\s*$")
MD_BEGIN = re.compile(r"^\s*<!-- BEGIN SOLUTION (\S+) -->\s*$")
MD_END = re.compile(r"^\s*<!-- END SOLUTION -->\s*$")
LEFTOVER = re.compile(r"BEGIN SOLUTION|END SOLUTION|BEGIN STUDENT")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_files() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True).stdout.decode().split("\0")
    return sorted(name for name in listed if name and (ROOT / name).is_file()
                  and not name.startswith(EXCLUDED_PREFIXES))


def enclosing_function(lines: list[str], index: int) -> str:
    for line in reversed(lines[:index]):
        match = re.match(r"\s*def (\w+)\(", line)
        if match:
            return match.group(1)
    return "module"


def python_tags(text: str) -> set[str]:
    return {match.group(2) for match in map(PY_BEGIN.match, text.split("\n")) if match}


def process_python(text: str, path: str, released: set[str]) -> str:
    lines, output, i = text.split("\n"), [], 0
    while i < len(lines):
        begin = PY_BEGIN.match(lines[i])
        if not begin:
            if PY_STUDENT.match(lines[i]) or PY_END.match(lines[i]):
                raise ValueError(f"{path}:{i + 1}: marker outside a SOLUTION block")
            output.append(lines[i])
            i += 1
            continue
        indent, tag = begin.groups()
        if tag not in TAGS:
            raise ValueError(f"{path}:{i + 1}: unknown tag {tag!r}")
        answer, student, part, j = [], [], "answer", i + 1
        while j < len(lines) and not PY_END.match(lines[j]):
            if PY_BEGIN.match(lines[j]):
                raise ValueError(f"{path}:{j + 1}: nested SOLUTION block")
            if PY_STUDENT.match(lines[j]):
                part = "student"
            else:
                (answer if part == "answer" else student).append(lines[j])
            j += 1
        if j == len(lines):
            raise ValueError(f"{path}:{i + 1}: SOLUTION block is never closed")
        if tag in released:
            output.extend(answer)
        elif student:
            for line in student:
                stripped = line[len(indent):] if line.startswith(indent) else line.lstrip()
                if not stripped.startswith("#"):
                    raise ValueError(f"{path}: STUDENT lines must be commented out in the answer key")
                output.append(indent + (stripped[2:] if stripped.startswith("# ") else stripped[1:]))
        else:
            name = enclosing_function(lines, i)
            output.append(f"{indent}# Write your code here, then delete the raise line below.")
            output.append(f'{indent}raise NotImplementedError("{tag}: implement {name}() - see the TODO in its docstring")')
        i = j + 1
    return "\n".join(output)


def process_markdown(text: str, path: str, released: set[str]) -> str:
    lines, output, i = text.split("\n"), [], 0
    while i < len(lines):
        begin = MD_BEGIN.match(lines[i])
        if not begin:
            if MD_END.match(lines[i]):
                raise ValueError(f"{path}:{i + 1}: END marker without BEGIN")
            output.append(lines[i])
            i += 1
            continue
        tag, j = begin.group(1), i + 1
        if tag not in TAGS:
            raise ValueError(f"{path}:{i + 1}: unknown tag {tag!r}")
        while j < len(lines) and not MD_END.match(lines[j]):
            j += 1
        if j == len(lines):
            raise ValueError(f"{path}:{i + 1}: SOLUTION block is never closed")
        if tag in released:
            output.extend(lines[i + 1:j])
        i = j + 1
    return "\n".join(output)


def checked_python(text: str, name: str) -> str:
    if LEFTOVER.search(text):
        raise ValueError(f"{name}: solution marker left in the teaching output")
    compile(text, name, "exec")
    return text


def render(released: set[str]) -> dict[str, tuple[bytes, int]]:
    """All output files in memory: path -> (content, permission bits)."""
    outputs = {}
    for name in source_files():
        source = ROOT / name
        mode = source.stat().st_mode & 0o777
        if name.endswith(".py"):
            text = source.read_text(encoding="utf-8")
            outputs[name] = (checked_python(process_python(text, name, set()), name).encode(), mode)
            shown = python_tags(text) & released
            if shown:
                answer = checked_python(process_python(text, name, shown), name)
                outputs[f"{ANSWERS_DIR}/{name}"] = (answer.encode(), mode)
        elif name.endswith(".md"):
            text = process_markdown(source.read_text(encoding="utf-8"), name, released)
            if LEFTOVER.search(text):
                raise ValueError(f"{name}: solution marker left in the teaching output")
            outputs[name] = (text.encode(), mode)
        else:
            outputs[name] = (source.read_bytes(), mode)
    return outputs


def git_identity() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"source_commit": commit or None, "source_worktree_clean": not dirty}


def build(output: Path, released: set[str], *, reset_releases: bool = False, overwrite_changed: bool = False) -> dict:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("The output must be outside this repository")
    record_path = output / BUILD_RECORD
    released, previous = set(released), {}
    if output.exists() and any(output.iterdir()):
        if not record_path.is_file():
            raise ValueError(f"{output} is not empty and has no {BUILD_RECORD}; refusing to touch it")
        record = json.loads(record_path.read_text())
        previous = record["files"]
        if not reset_releases:
            released |= set(record.get("released_tags", []))
    outputs = render(released)
    kept, files = [], {}
    for name, generated in previous.items():
        path = output / name
        if not path.is_file():
            continue
        if sha256_bytes(path.read_bytes()) != generated and not overwrite_changed:
            kept.append(name)        # changed in the teaching repository: leave it alone
            files[name] = generated  # keep the old fingerprint so it stays protected
        elif name not in outputs:
            path.unlink()            # an unchanged file that the new build no longer produces
    for name, (data, mode) in outputs.items():
        path = output / name
        if name in kept:
            continue
        if path.exists() and name not in previous and not overwrite_changed:
            kept.append(name)        # a file the build did not create (for example a member's own file)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        files[name] = sha256_bytes(data)
    record = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "generator": "teaching/make_teaching_version.py", **git_identity(),
              "released_tags": sorted(released), "kept_unchanged_by_build": sorted(kept), "files": dict(sorted(files.items()))}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", action="append", default=[], choices=[t for t in TAGS if t not in NEVER_RELEASED],
                        help="Show this tag's answers (repeatable); remembered by later builds")
    parser.add_argument("--release-all", action="store_true", help="Show every releasable answer (for checking)")
    parser.add_argument("--reset-releases", action="store_true",
                        help="Forget the tags released by earlier builds of this output")
    parser.add_argument("--overwrite-changed", action="store_true",
                        help="Also overwrite files that changed in the output since the last build")
    args = parser.parse_args(argv)
    released = {t for t in TAGS if t not in NEVER_RELEASED} if args.release_all else set(args.release)
    record = build(args.output, released, reset_releases=args.reset_releases, overwrite_changed=args.overwrite_changed)
    print(f"Wrote {len(record['files']) - len(record['kept_unchanged_by_build'])} files to {args.output.resolve()} "
          f"(released: {', '.join(record['released_tags']) or 'none'}; source {record['source_commit'] or 'uncommitted'}"
          f"{'' if record['source_worktree_clean'] else ', with uncommitted changes'})")
    if record["kept_unchanged_by_build"]:
        print("Kept (changed in the teaching repository or not created by this build):")
        for name in record["kept_unchanged_by_build"]:
            print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
