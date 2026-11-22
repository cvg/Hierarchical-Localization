from pathlib import Path

import h5py
import numpy as np
import pycolmap
import pytest

from hloc.utils.io import get_keypoints, get_matches, list_h5_names, write_poses
from hloc.utils.parsers import (
    names_to_pair,
    names_to_pair_old,
    parse_image_list,
    parse_image_lists,
    parse_retrieval,
)


def test_parse_image_list_and_lists(tmp_path: Path) -> None:
    list_path = tmp_path / "images.txt"
    list_path.write_text("# comment\n\nimg0.jpg\ndir/img1.jpg\n")
    assert parse_image_list(list_path) == ["img0.jpg", "dir/img1.jpg"]
    assert parse_image_lists(list_path, with_intrinsics=False) == [
        "img0.jpg",
        "dir/img1.jpg",
    ]

    intrinsics_path = tmp_path / "queries_with_intrinsics.txt"
    intrinsics_path.write_text(
        "# comment\n"
        "q0.jpg SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n"
        "q1.jpg PINHOLE 800 600 600.0 610.0 400.0 300.0\n"
    )
    parsed = parse_image_lists(intrinsics_path, with_intrinsics=True)
    assert len(parsed) == 2
    name0, cam0 = parsed[0]
    assert name0 == "q0.jpg"
    assert cam0.model == pycolmap.CameraModelId.SIMPLE_PINHOLE
    assert (cam0.width, cam0.height) == (640, 480)
    assert np.allclose(cam0.params, [500.0, 320.0, 240.0])

    name1, cam1 = parsed[1]
    assert name1 == "q1.jpg"
    assert cam1.model == pycolmap.CameraModelId.PINHOLE
    assert np.allclose(cam1.params, [600.0, 610.0, 400.0, 300.0])


def test_parse_retrieval_and_names_to_pair(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("a/0.jpg b/1.jpg\na/0.jpg b/2.jpg\nc/3.jpg b/1.jpg\n")
    ret = parse_retrieval(pairs_path)
    assert ret == {"a/0.jpg": ["b/1.jpg", "b/2.jpg"], "c/3.jpg": ["b/1.jpg"]}

    assert names_to_pair("a/0.jpg", "b/1.jpg") == "a-0.jpg/b-1.jpg"
    assert names_to_pair_old("a/0.jpg", "b/1.jpg") == "a-0.jpg_b-1.jpg"
    assert names_to_pair("a/0.jpg", "b/1.jpg", separator="__") == "a-0.jpg__b-1.jpg"


def test_h5_io_roundtrip_and_reversed_matches(tmp_path: Path) -> None:
    feats_path = tmp_path / "feats.h5"
    kpts0 = np.array([[10.5, 20.5], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    with h5py.File(feats_path, "w") as f:
        grp = f.create_group("sub/img0.jpg")
        grp.create_dataset("keypoints", data=kpts0)
        grp["keypoints"].attrs["uncertainty"] = 2.0

    assert list_h5_names(feats_path) == ["sub/img0.jpg"]
    kpts_read = get_keypoints(feats_path, "sub/img0.jpg")
    assert np.allclose(kpts_read, kpts0)
    kpts_u, unc = get_keypoints(feats_path, "sub/img0.jpg", return_uncertainty=True)
    assert np.allclose(kpts_u, kpts0)
    assert unc == pytest.approx(2.0)

    matches_path = tmp_path / "matches.h5"
    with h5py.File(matches_path, "w") as f:
        grp = f.create_group(names_to_pair("a.jpg", "b.jpg"))
        grp.create_dataset("matches0", data=np.array([2, -1, 0], dtype=np.int32))
        grp.create_dataset(
            "matching_scores0", data=np.array([0.9, 0.1, 0.7], dtype=np.float32)
        )

    m_fwd, s_fwd = get_matches(matches_path, "a.jpg", "b.jpg")
    assert np.array_equal(m_fwd, np.array([[0, 2], [2, 0]]))
    assert np.allclose(s_fwd, [0.9, 0.7])

    m_rev, s_rev = get_matches(matches_path, "b.jpg", "a.jpg")
    assert np.array_equal(m_rev, np.array([[2, 0], [0, 2]]))
    assert np.allclose(s_rev, [0.9, 0.7])

    with pytest.raises(ValueError, match="Could not find pair"):
        get_matches(matches_path, "a.jpg", "missing.jpg")


def test_write_poses_format(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.txt"
    q_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(q_xyzw), np.array([1.0, -2.5, 3.25]))
    write_poses({"q0.jpg": pose}, poses_path, prepend_camera_name=False)

    parts = poses_path.read_text().strip().split()
    assert parts[0] == "q0.jpg"
    vals = [float(x) for x in parts[1:]]
    # write_poses emits qw qx qy qz tx ty tz
    assert np.allclose(vals[:4], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert np.allclose(vals[4:], [1.0, -2.5, 3.25])
