import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pycolmap


class ReachedModelLoader(Exception):
    pass


class TriangulationCLITest(unittest.TestCase):
    def test_reference_model_flag_reaches_model_loader(self):
        # Stop at the native model loader: this test covers CLI routing, not SfM.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            reference.mkdir()
            paths = {name: root / name for name in ("pairs", "features", "matches")}
            for path in paths.values():
                path.touch()
            argv = [
                "hloc.triangulation",
                "--sfm_dir",
                str(root / "output"),
                "--reference_sfm_model",
                str(reference),
                "--image_dir",
                str(root),
            ]
            for name, path in paths.items():
                argv.extend([f"--{name}", str(path)])
            with patch.object(sys, "argv", argv), patch.object(
                pycolmap, "Reconstruction", side_effect=ReachedModelLoader
            ) as load_model:
                with self.assertRaises(ReachedModelLoader):
                    runpy.run_module("hloc.triangulation", run_name="__main__")
            load_model.assert_called_once_with(reference)


if __name__ == "__main__":
    unittest.main()
