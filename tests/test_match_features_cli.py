import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MatchFeaturesCliTest(unittest.TestCase):
    def test_cli_accepts_explicit_feature_and_match_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pairs = tmpdir / "pairs.txt"
            features = tmpdir / "features.h5"
            matches = tmpdir / "matches.h5"
            pairs.write_text("", encoding="utf-8")
            features.touch()

            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hloc.match_features",
                    "--pairs",
                    str(pairs),
                    "--features",
                    str(features),
                    "--matches",
                    str(matches),
                    "--conf",
                    "NN-mutual",
                ],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(
                proc.returncode,
                0,
                "stdout:\n{0}\nstderr:\n{1}".format(proc.stdout, proc.stderr),
            )


if __name__ == "__main__":
    unittest.main()
