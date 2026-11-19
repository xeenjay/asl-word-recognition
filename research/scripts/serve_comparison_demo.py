#!/usr/bin/env python3
"""Private loopback UI and serialized video inference, with persistent job evidence.

Run on the project server, then use an SSH local forward. Static files are served
only from docs/comparison-demo; raw model paths and uploaded files are not served.
No model training, dataset evaluation, or external service is triggered.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import traceback
from urllib.parse import unquote, urlsplit
import uuid

MAX_VIDEO_BYTES = 25 * 1024 * 1024
MAX_BODY_BYTES = MAX_VIDEO_BYTES * 4 // 3 + 65536


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def safe_public_result(value):
    """Expose predictions, not internal machine paths or pickle provenance."""
    result = {k: value[k] for k in ('status', 'task', 'coverage', 'timings_seconds',
                                  'warnings', 'interpretation', 'shared_raw_pose',
                                  'pose_extraction_count') if k in value}
    result['methods'] = {}
    for name, method in value.get('methods', {}).items():
        result['methods'][name] = {k: method[k] for k in
            ('prediction', 'topk', 'score_type', 'timings_seconds', 'seed', 'description')
            if k in method}
        if 'seed' in method.get('provenance', {}):
            result['methods'][name]['seed'] = method['provenance']['seed']
    return result


class DemoState:
    def __init__(self, root, static_dir, session_dir, device='cpu', engine_factory=None):
        self.root = Path(root).resolve()
        self.static_dir = Path(static_dir).resolve()
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.device = device
        self.engine_factory = engine_factory
        self.engine = None
        self.jobs = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='asl-video')
        gallery = json.loads((self.static_dir / 'analysis.json').read_text())
        self.cases = {}
        for case in gallery['cases']:
            candidate = (self.static_dir / case['video_file']).resolve()
            if not candidate.is_relative_to(self.static_dir) or not candidate.is_file():
                raise ValueError('Case video missing or outside static directory')
            self.cases[str(case['id'])] = candidate
        write_json(self.session_dir / 'session.json', {
            'status': 'serving', 'started_at_utc': utc(), 'pid': os.getpid(),
            'device': device, 'purpose': 'interactive_validation_demo',
            'test_used': False, 'max_video_bytes': MAX_VIDEO_BYTES,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), self.root / 'scripts/infer_comparison.py')
                              if p.is_file()},
        })

    def submit(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        allowed = {'case_id'} if 'case_id' in payload else {'filename', 'video_base64'}
        if set(payload) != allowed:
            raise ValueError('Choose one gallery case or one video upload')
        content = None
        if 'case_id' in payload:
            video = self.cases.get(str(payload['case_id']))
            if video is None:
                raise ValueError('Unknown gallery case')
            source = {'case_id': str(payload['case_id']), 'kind': 'validation_gallery'}
        else:
            if not isinstance(payload['filename'], str) or not isinstance(payload['video_base64'], str):
                raise ValueError('Invalid upload')
            extension = Path(payload['filename']).suffix.lower()
            if extension not in {'.mp4', '.mov', '.webm', '.avi', '.m4v'}:
                raise ValueError('Use an MP4, MOV, WebM, AVI or M4V video')
            try:
                content = base64.b64decode(payload['video_base64'], validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError('Invalid base64 video data') from error
            if not 0 < len(content) <= MAX_VIDEO_BYTES:
                raise ValueError('Video must be nonempty and at most 25 MiB')
            source = {'kind': 'user_upload', 'bytes': len(content),
                      'sha256': hashlib.sha256(content).hexdigest()}
        with self.lock:
            active = sum(x['status'] in {'queued', 'running'} for x in self.jobs.values())
            if active >= 3:
                raise RuntimeError('Three videos are already queued; wait for a result')
            job_id = uuid.uuid4().hex
            directory = self.session_dir / 'jobs' / job_id
            directory.mkdir(parents=True, exist_ok=False)
            if content is not None:
                video = directory / ('input' + extension)
                video.write_bytes(content)
            job = {'job_id': job_id, 'status': 'queued', 'created_at_utc': utc(),
                   'source': source, 'stage': 'queued'}
            self.jobs[job_id] = job
            write_json(directory / 'job.json', job)
        self.executor.submit(self._predict, job_id, video, directory)
        return {'job_id': job_id, 'status': 'queued'}

    def _update(self, job_id, directory, **kwargs):
        with self.lock:
            self.jobs[job_id].update(kwargs)
            write_json(directory / 'job.json', self.jobs[job_id])

    def _predict(self, job_id, video, directory):
        try:
            self._update(job_id, directory, status='running', stage='loading_models', started_at_utc=utc())
            if self.engine is None:
                if self.engine_factory is None:
                    from infer_comparison import ComparisonEngine
                    self.engine_factory = ComparisonEngine
                self.engine = self.engine_factory(self.root, device=self.device)
            self._update(job_id, directory, stage='extracting_and_predicting')
            result = self.engine.predict_video(video, directory / 'prediction')
            self._update(job_id, directory, status='completed', stage='complete',
                         result=safe_public_result(result), ended_at_utc=utc())
        except Exception as error:
            (directory / 'error.log').write_text(traceback.format_exc())
            self._update(job_id, directory, status='failed', stage='failed', ended_at_utc=utc(),
                         error={'type': type(error).__name__,
                                'message': 'Video processing failed. Use a valid isolated-word clip '
                                           '(up to 30 seconds, 900 frames, 1920×1080 and 25 MiB). '
                                           'Detailed diagnostics are saved in the server job log.'})

    def get_job(self, job_id):
        with self.lock:
            value = self.jobs.get(job_id)
            return json.loads(json.dumps(value)) if value is not None else None


class DemoHandler(SimpleHTTPRequestHandler):
    state: DemoState

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.state.static_dir), **kwargs)

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._allowed_host():
            return self._json(403, {'error': 'Use a localhost SSH tunnel'})
        route = urlsplit(self.path).path
        if route == '/api/status':
            return self._json(200, {'ready': True, 'service': 'asl-comparison-demo',
                'models_loaded': self.state.engine is not None,
                'limits': {'max_video_mb': 25, 'max_duration_seconds': 30},
                'test_used': False})
        if route.startswith('/api/jobs/'):
            job_id = route.removeprefix('/api/jobs/')
            if not re.fullmatch('[0-9a-f]{32}', job_id):
                return self._json(404, {'error': 'Unknown job'})
            job = self.state.get_job(job_id)
            return self._json(200 if job else 404, job or {'error': 'Unknown job'})
        if route.startswith('/api/'):
            return self._json(404, {'error': 'Unknown endpoint'})
        return super().do_GET()

    def do_POST(self):
        if not self._allowed_host():
            return self._json(403, {'error': 'Use a localhost SSH tunnel'})
        if urlsplit(self.path).path != '/api/predict':
            return self._json(404, {'error': 'Unknown endpoint'})
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self._json(403, {'error': 'Cross-origin requests are not allowed'})
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            return self._json(415, {'error': 'Expected application/json'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY_BYTES:
                return self._json(413, {'error': 'Request exceeds upload limit'})
            payload = json.loads(self.rfile.read(length))
            result = self.state.submit(payload)
            return self._json(202, result)
        except (ValueError, KeyError, UnicodeError) as error:
            return self._json(400, {'error': str(error)})
        except RuntimeError as error:
            return self._json(429, {'error': str(error)})

    def _allowed_host(self):
        try:
            return urlsplit('//' + self.headers.get('Host', '')).hostname in {'127.0.0.1', 'localhost'}
        except ValueError:
            return False

    def do_HEAD(self):
        if not self._allowed_host():
            self.send_error(403)
            return
        return super().do_HEAD()

    def send_head(self):
        route = unquote(urlsplit(self.path).path)
        relative = route.lstrip('/') or 'index.html'
        candidate = (self.state.static_dir / relative).resolve()
        if (not candidate.is_relative_to(self.state.static_dir) or
                candidate.suffix.lower() not in {'.html', '.js', '.css', '.json', '.md', '.jpg', '.jpeg', '.png', '.mp4'}):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--session-dir', type=Path, required=True, help='New directory only')
    args = parser.parse_args()
    root = args.root.resolve()
    state = DemoState(root, root / 'docs/comparison-demo', args.session_dir, args.device)
    DemoHandler.state = state
    server = ThreadingHTTPServer(('127.0.0.1', args.port), DemoHandler)
    print(json.dumps({'status': 'serving', 'address': f'http://127.0.0.1:{args.port}',
                      'session_dir': str(state.session_dir)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.executor.shutdown(wait=True)
        write_json(state.session_dir / 'stopped.json', {'status': 'stopped', 'at_utc': utc()})


if __name__ == '__main__':
    main()
