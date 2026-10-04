#!/usr/bin/env python3
"""Install the verified official heavy pose model into this project's venv."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe'
SOURCE_URL = 'https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'assets/mediapipe/pose_landmark_heavy.tflite'
    if digest(source) != EXPECTED or source.read_bytes()[4:8] != b'TFL3':
        raise ValueError('Official MediaPipe asset integrity check failed')
    spec = importlib.util.find_spec('mediapipe')
    if spec is None or spec.origin is None:
        raise RuntimeError('Run using the project .venv Python')
    target = Path(spec.origin).parent / 'modules/pose_landmark/pose_landmark_heavy.tflite'
    if not target.resolve().is_relative_to((ROOT / '.venv').resolve()):
        raise RuntimeError('Refusing to modify a non-project MediaPipe installation')
    if target.exists() and digest(target) != EXPECTED:
        raise RuntimeError('Existing model differs; inspect it before replacing')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix('.tflite.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    assert digest(target) == EXPECTED
    print(json.dumps({'status': 'verified', 'target': str(target),
                      'sha256': EXPECTED, 'source': SOURCE_URL,
                      'python': sys.executable}, indent=2))


if __name__ == '__main__':
    main()
