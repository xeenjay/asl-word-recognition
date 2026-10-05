"""Protocol and failure handling for the personal dataset preparation path."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import asl_citizen_source as source
import prepare_team_data as preparation


class SelectionTests(unittest.TestCase):
    def rows(self):
        return {split: [{'Gloss': f'WORD{i:02d}', 'Video file': f'{split}-{i}-{j}.mp4'}
                        for i in reversed(range(12)) for j in range(9)]
                for split in ('train', 'val')}

    def test_mini_is_deterministic_balanced_and_ignores_row_order(self):
        rows = self.rows()
        labels, selected, ids = preparation.select_samples(rows, 'mini')
        reordered = {key: list(reversed(value)) for key, value in rows.items()}
        self.assertEqual((labels, selected, ids), preparation.select_samples(reordered, 'mini'))
        self.assertEqual(labels, [f'WORD{i:02d}' for i in range(10)])
        self.assertEqual(ids, list(range(10)))
        self.assertEqual(len(selected['train']), 60)
        self.assertEqual(len(selected['val']), 20)
        self.assertFalse({r['Video file'] for r in selected['train']} &
                         {r['Video file'] for r in selected['val']})

    def test_full_preserves_every_selected_split_row(self):
        rows = self.rows()
        labels, selected, _ = preparation.select_samples(rows, 'full')
        self.assertEqual(len(labels), 12)
        for split in rows:
            self.assertEqual({r['Video file'] for r in rows[split]},
                             {r['Video file'] for r in selected[split]})

    def test_reject_test_profile_and_vocabulary_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Only mini and full'):
            preparation.select_samples(self.rows(), 'test')
        rows = self.rows()
        rows['val'][0]['Gloss'] = 'UNSEEN'
        with self.assertRaisesRegex(ValueError, 'vocabularies differ'):
            preparation.select_samples(rows, 'mini')

    def test_changed_official_csv_fails(self):
        with self.assertRaisesRegex(ValueError, 'differs from the pinned official split'):
            preparation.split_rows({'train': b'changed', 'val': b'changed'})


class CacheTests(unittest.TestCase):
    def test_cache_reuse_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            video_hash = 'a' * 64
            vector = np.arange(1020, dtype=np.float32)
            with patch.object(preparation, 'extract_video', return_value=(vector, {'decoded_frames': 3})) as extract:
                first = preparation.cached_features(Path('unused.mp4'), video_hash, cache)
                second = preparation.cached_features(Path('unused.mp4'), video_hash, cache)
                np.testing.assert_array_equal(first[0], second[0])
                extract.assert_called_once()
            (cache / (video_hash + '.npy')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'cache integrity'):
                preparation.cached_features(Path('unused.mp4'), video_hash, cache)


class DownloadTests(unittest.TestCase):
    class Response:
        def __init__(self, code, headers, body):
            self.status_code, self.headers, self.body = code, headers, body
            self.body_read = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def iter_content(self, chunk_size):
            self.body_read = True
            yield self.body

    class Session:
        def __init__(self, response):
            self.response = response
        def get(self, *args, **kwargs):
            return self.response

    def test_server_ignoring_range_never_reads_full_archive(self):
        response = self.Response(200, {}, b'huge archive')
        reader = source.RangeReader(self.Session(response))
        with self.assertRaisesRegex(source.AuditError, 'Expected HTTP 206'):
            reader.read(0, 9)
        self.assertFalse(response.body_read)

    def test_wrong_content_range_fails_before_body(self):
        response = self.Response(206, {'Content-Range': 'bytes 1-10/123'}, b'bad')
        reader = source.RangeReader(self.Session(response))
        with self.assertRaisesRegex(source.AuditError, 'Content-Range'):
            reader.read(0, 9)
        self.assertFalse(response.body_read)

    def test_range_byte_budget_and_truncation(self):
        response = self.Response(206, {'Content-Range': f'bytes 0-9/{source.ARCHIVE_BYTES}'}, b'abc')
        reader = source.RangeReader(self.Session(response), max_download_bytes=10)
        with self.assertRaisesRegex(source.AuditError, 'Truncated'):
            reader.read(0, 9)
        with self.assertRaisesRegex(source.AuditError, 'download budget'):
            reader.read(0, 1)


if __name__ == '__main__':
    unittest.main()
