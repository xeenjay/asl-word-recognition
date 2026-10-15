"""Answer-key repository only: the generated teaching version must hide answers
yet keep the shared infrastructure working; released answers must be complete
and must never overwrite a member's own code. Excluded from the teaching repository."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "teaching"))
from make_teaching_version import LEFTOVER, build, process_markdown, process_python  # noqa: E402

MEMBER_TESTS = ("tests.test_member1_knn", "tests.test_member2_linear_svm",
                "tests.test_member2_handedness", "tests.test_member3_random_forest")
SHARED_TESTS = ("tests.test_team_runner", "tests.test_classical_models", "tests.test_classical_features")
ALL_RELEASABLE = {"member1", "member2", "member3", "results-2026-10"}
ANSWER_KEY = "Teaching answer key"


def run_tests(folder, modules):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "unittest", *modules], cwd=folder, env=environment,
                          capture_output=True, text=True)


class MarkerProcessing(unittest.TestCase):
    def test_python_stub_student_part_and_release(self):
        text = ("def f(x):\n    # BEGIN SOLUTION member1\n    return x + 1\n    # END SOLUTION\n"
                "# BEGIN SOLUTION member2\nNOTE = 'answer'\n# BEGIN STUDENT\n# NOTE = 'todo'\n# END SOLUTION\n")
        hidden = process_python(text, "x.py", set())
        self.assertIn('raise NotImplementedError("member1: implement f()', hidden)
        self.assertIn("NOTE = 'todo'", hidden)
        self.assertNotIn("x + 1", hidden)
        self.assertNotIn("'answer'", hidden)
        shown = process_python(text, "x.py", {"member1", "member2"})
        self.assertIn("    return x + 1", shown)
        self.assertIn("NOTE = 'answer'", shown)
        self.assertIsNone(LEFTOVER.search(hidden + shown))

    def test_markdown_blocks_and_errors(self):
        text = "a\n<!-- BEGIN SOLUTION instructor -->\nsecret\n<!-- END SOLUTION -->\nb"
        self.assertEqual(process_markdown(text, "x.md", set()), "a\nb")
        with self.assertRaises(ValueError):
            process_python("# BEGIN SOLUTION member1\nx = 1\n", "x.py", set())
        with self.assertRaises(ValueError):
            process_python("# BEGIN SOLUTION nobody\nx = 1\n# END SOLUTION\n", "x.py", set())


class GeneratedTree(unittest.TestCase):
    def test_hidden_build_keeps_infrastructure_and_hides_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            hidden = Path(folder) / "hidden"
            record = build(hidden, set())
            self.assertFalse(any(name.startswith(("teaching/", "answers/")) for name in record["files"]))
            self.assertNotIn("tests/test_teaching_version.py", record["files"])
            for name in record["files"]:
                if name.endswith((".py", ".md")):
                    self.assertIsNone(LEFTOVER.search((hidden / name).read_text()), name)
            self.assertNotIn(ANSWER_KEY, (hidden / "students/knn.py").read_text())
            self.assertNotIn("Answer notes", (hidden / "docs/roles/member1-knn.md").read_text())
            self.assertEqual(run_tests(hidden, SHARED_TESTS).returncode, 0)
            for module in MEMBER_TESTS:
                result = run_tests(hidden, (module,))
                self.assertNotEqual(result.returncode, 0, f"{module} passes without the answers")
                self.assertIn("NotImplementedError", result.stderr)

    def test_released_answers_are_complete_and_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "released"
            build(output, ALL_RELEASABLE)
            self.assertNotIn(ANSWER_KEY, (output / "students/knn.py").read_text())
            self.assertIn(ANSWER_KEY, (output / "answers/students/knn.py").read_text())
            self.assertIn("Answer notes", (output / "docs/roles/member1-knn.md").read_text())
            self.assertNotIn("Instructor notes", (output / "docs/roles/README.md").read_text())
            for answer in (output / "answers").rglob("*.py"):
                shutil.copyfile(answer, output / answer.relative_to(output / "answers"))
            result = run_tests(output, MEMBER_TESTS + SHARED_TESTS)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_rebuilds_remember_releases_and_protect_member_work(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "teaching"
            build(output, set())
            (output / "data").mkdir()
            (output / "data/keep.txt").write_text("local data")
            build(output, {"member1"})
            self.assertEqual((output / "data/keep.txt").read_text(), "local data")
            self.assertIn(ANSWER_KEY, (output / "answers/students/knn.py").read_text())
            build(output, {"member2"})
            self.assertTrue((output / "answers/students/knn.py").is_file(), "member1 release was forgotten")
            self.assertTrue((output / "answers/students/linear_svm.py").is_file())
            own = "# member 1's own implementation\n"
            (output / "students/knn.py").write_text(own)
            record = build(output, set())
            self.assertEqual((output / "students/knn.py").read_text(), own)
            self.assertIn("students/knn.py", record["kept_unchanged_by_build"])
            self.assertEqual((output / "students/knn.py").read_text(), own)
            build(output, set(), reset_releases=True)
            self.assertFalse((output / "answers/students/knn.py").exists())
            self.assertEqual((output / "students/knn.py").read_text(), own)
            build(output, set(), overwrite_changed=True)
            self.assertIn("raise NotImplementedError", (output / "students/knn.py").read_text())

    def test_refuses_unknown_nonempty_output(self):
        with tempfile.TemporaryDirectory() as folder:
            stranger = Path(folder) / "stranger"
            stranger.mkdir()
            (stranger / "file.txt").write_text("x")
            with self.assertRaises(ValueError):
                build(stranger, set())


if __name__ == "__main__":
    unittest.main()
