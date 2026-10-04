"""The portable local runner must work without optional vendor assets."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PortableRunnerTests(unittest.TestCase):
    def test_classical_only_run_without_vendor_and_failure_is_preserved(self):
        source = Path(__file__).resolve().parents[1] / "scripts/run_logged.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            script = root / "scripts/run_logged.py"
            shutil.copyfile(source, script)
            for expected in (0, 7):
                result = subprocess.run([sys.executable, "-B", str(script), "--name", f"probe-{expected}",
                                         "--", sys.executable, "-c", f"raise SystemExit({expected})"],
                                        cwd=root, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)
            runs = list((root / "runs").iterdir())
            self.assertEqual(len(runs), 2)
            seen = set()
            for run in runs:
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertNotIn("host", manifest)
                self.assertEqual(manifest["location"], "local_workspace")
                seen.add((manifest["state"], manifest["returncode"]))
                snapshot = json.loads((run / "code_manifest.json").read_text())
                self.assertEqual([x["path"] for x in snapshot["files"]], ["scripts/run_logged.py"])
                self.assertTrue((run / "run.log").is_file())
            self.assertEqual(seen, {("completed", 0), ("failed", 7)})


if __name__ == "__main__":
    unittest.main()
