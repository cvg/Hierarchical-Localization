import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hloc import extract_features, pairs_from_covisibility


class ReachedPipeline(Exception):
    pass


class DatasetEntrypointTest(unittest.TestCase):
    def test_robotcar_starts_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intrinsics = root / "intrinsics"
            intrinsics.mkdir()
            for side in ("left", "right", "rear"):
                (intrinsics / f"{side}_intrinsics.txt").write_text(
                    "fx 800\nfy 800\ncx 512\ncy 512\n"
                )
            argv = [
                "hloc.pipelines.RobotCar.pipeline",
                "--dataset",
                str(root),
                "--outputs",
                str(root / "output"),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                extract_features, "main", side_effect=ReachedPipeline
            ) as extract:
                with self.assertRaises(ReachedPipeline):
                    runpy.run_module(argv[0], run_name="__main__")
            self.assertEqual(extract.call_args.args[1], root / "images")

    def test_aachen_loftr_starts_covisibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = [
                "hloc.pipelines.Aachen_v1_1.pipeline_loftr",
                "--dataset",
                str(root),
                "--outputs",
                str(root / "output"),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                pairs_from_covisibility, "main", side_effect=ReachedPipeline
            ) as covisibility:
                with self.assertRaises(ReachedPipeline):
                    runpy.run_module(argv[0], run_name="__main__")
            self.assertEqual(
                covisibility.call_args.args[0], root / "3D-models/aachen_v_1_1"
            )


if __name__ == "__main__":
    unittest.main()
