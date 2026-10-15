import importlib
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hloc import pairs_from_covisibility


class ReachedSlice(Exception):
    pass


class CMUPipelineTest(unittest.TestCase):
    def test_cli_slice_selection(self):
        for selection, expected in [
            (None, "slice2"),
            ("5-6", "slice5"),
            ("[3,4]", "slice3"),
        ]:
            with self.subTest(
                selection=selection
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv = [
                    "hloc.pipelines.CMU.pipeline",
                    "--dataset",
                    str(root),
                    "--outputs",
                    str(root / "output"),
                ]
                if selection is not None:
                    argv.extend(["--slices", selection])
                with patch.object(sys, "argv", argv), patch.object(
                    pairs_from_covisibility, "main", side_effect=ReachedSlice
                ) as covisibility:
                    with self.assertRaises(ReachedSlice):
                        runpy.run_module(argv[0], run_name="__main__")
                self.assertEqual(
                    covisibility.call_args.args[0], root / expected / "sparse"
                )

    def test_generated_query_list_is_passed_to_localizer(self):
        pipeline = importlib.import_module("hloc.pipelines.CMU.pipeline")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(pipeline.pairs_from_covisibility, "main"), patch.object(
                pipeline.extract_features, "main", return_value=root / "features.h5"
            ), patch.object(
                pipeline.match_features, "main", return_value=root / "matches.h5"
            ), patch.object(
                pipeline.triangulation, "main"
            ), patch.object(
                pipeline, "generate_query_list"
            ) as generate, patch.object(
                pipeline.pairs_from_retrieval, "main"
            ), patch.object(
                pipeline.localize_sfm, "main"
            ) as localize:
                pipeline.run_slice("slice2", root, root / "output", 20, 10)
            query_list = root / "slice2/queries_with_intrinsics.txt"
            self.assertEqual(generate.call_args.args[1], query_list)
            self.assertEqual(localize.call_args.args[1], query_list)


if __name__ == "__main__":
    unittest.main()
