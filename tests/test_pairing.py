from pathlib import Path

import h5py
import numpy as np
import pycolmap
import pytest

from hloc import (
    pairs_from_covisibility,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_retrieval,
)


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    text = path.read_text().strip()
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        a, b = line.split()
        pairs.append((a, b))
    return pairs


def _make_pose_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        width=640,
        height=480,
        params=[500, 320, 240],
    )
    rec.add_camera_with_trivial_rig(cam)
    # Four cameras along X axis; camera 4 rotated by 90 deg around Y
    configs = [
        (1, "img0.jpg", np.eye(3), np.array([0.0, 0.0, 0.0])),
        (2, "img1.jpg", np.eye(3), np.array([1.0, 0.0, 0.0])),
        (3, "img2.jpg", np.eye(3), np.array([3.0, 0.0, 0.0])),
        (
            4,
            "img3.jpg",
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([0.5, 0.0, 0.0]),
        ),
    ]
    for img_id, name, R_wc, c_w in configs:
        R_cw = R_wc.T
        t_cw = -R_cw @ c_w
        img = pycolmap.Image(
            image_id=img_id,
            camera_id=1,
            name=name,
            keypoints=np.array([[100.0, 100.0], [200.0, 200.0]]),
        )
        rec.add_image_with_trivial_frame(
            img, pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw), t_cw)
        )
    # img0 shares 2 points with img1, 1 point with img2, 0 with img3
    rec.add_point3D(
        np.array([0.0, 0.0, 5.0]),
        pycolmap.Track(
            [
                pycolmap.TrackElement(1, 0),
                pycolmap.TrackElement(2, 0),
                pycolmap.TrackElement(3, 0),
            ]
        ),
    )
    rec.add_point3D(
        np.array([1.0, 0.0, 5.0]),
        pycolmap.Track([pycolmap.TrackElement(1, 1), pycolmap.TrackElement(2, 1)]),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    rec.write_binary(model_dir)
    return rec


def test_pairs_from_exhaustive(tmp_path: Path) -> None:
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    out = tmp_path / "pairs.txt"
    pairs_from_exhaustive.main(out, image_list=names)
    pairs = _read_pairs(out)
    assert len(pairs) == 6
    assert len({frozenset(p) for p in pairs}) == 6
    assert all(a != b for a, b in pairs)

    # Disjoint ref_list -> cross-product
    out_ref = tmp_path / "pairs_ref.txt"
    pairs_from_exhaustive.main(
        out_ref, image_list=["q0.jpg", "q1.jpg"], ref_list=["r0.jpg", "r1.jpg"]
    )
    assert _read_pairs(out_ref) == [
        ("q0.jpg", "r0.jpg"),
        ("q0.jpg", "r1.jpg"),
        ("q1.jpg", "r0.jpg"),
        ("q1.jpg", "r1.jpg"),
    ]

    # From features H5 + ref_list (also exercises features=Path with ref_list=list)
    feats = tmp_path / "feats.h5"
    with h5py.File(feats, "w") as f:
        for name in ("q0.jpg", "q1.jpg"):
            grp = f.create_group(name)
            grp.create_dataset("keypoints", data=np.zeros((1, 2), dtype=np.float32))
    out_feats = tmp_path / "pairs_feats.txt"
    pairs_from_exhaustive.main(out_feats, features=feats, ref_list=["r0.jpg"])
    assert sorted(_read_pairs(out_feats)) == [
        ("q0.jpg", "r0.jpg"),
        ("q1.jpg", "r0.jpg"),
    ]

    with pytest.raises(ValueError, match="Provide either"):
        pairs_from_exhaustive.main(out)


def test_pairs_from_covisibility_and_poses(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _make_pose_reconstruction(model_dir)

    covis_out = tmp_path / "covis.txt"
    pairs_from_covisibility.main(model_dir, covis_out, num_matched=1)
    covis_pairs = _read_pairs(covis_out)
    # img0 shares 2 points with img1 (vs 1 with img2), so top-1 for img0 is img1
    assert ("img0.jpg", "img1.jpg") in covis_pairs
    assert ("img1.jpg", "img0.jpg") in covis_pairs

    poses_out = tmp_path / "poses.txt"
    pairs_from_poses.main(model_dir, poses_out, num_matched=1, rotation_threshold=30.0)
    pose_pairs = _read_pairs(poses_out)
    # img3 is closest to img0 in translation (0.5m), but rotated 90 deg -> filtered out!
    assert ("img0.jpg", "img1.jpg") in pose_pairs
    assert ("img0.jpg", "img3.jpg") not in pose_pairs
    assert all(a != "img3.jpg" and b != "img3.jpg" for a, b in pose_pairs)


def test_pairs_from_retrieval(tmp_path: Path) -> None:
    desc_path = tmp_path / "global.h5"
    with h5py.File(desc_path, "w") as f:
        for name, vec in [
            ("q/0.jpg", [1.0, 0.0, 0.0]),
            ("db/0.jpg", [0.99, 0.1, 0.0]),
            ("db/1.jpg", [0.0, 1.0, 0.0]),
            ("db/2.jpg", [0.0, 0.0, 1.0]),
        ]:
            grp = f.create_group(name)
            grp.create_dataset(
                "global_descriptor", data=np.array(vec, dtype=np.float32)
            )

    out = tmp_path / "retrieval.txt"
    pairs_from_retrieval.main(
        desc_path, out, num_matched=1, query_prefix="q/", db_prefix="db/"
    )
    assert _read_pairs(out) == [("q/0.jpg", "db/0.jpg")]
