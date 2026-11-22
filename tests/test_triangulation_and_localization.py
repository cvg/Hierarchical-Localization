import pickle
from pathlib import Path

import h5py
import numpy as np
import pycolmap
import pytest

from hloc import localize_inloc, localize_sfm, triangulation
from hloc.utils.parsers import names_to_pair


def _build_synthetic_scene(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, pycolmap.Rigid3d, np.ndarray]:
    """Create a 3-view synthetic mapping scene + 1 query image with exact GT poses."""
    rng = np.random.default_rng(42)
    # 60 well-distributed 3D points in front of cameras (Z in [4, 8])
    pts3d = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, size=60),
            rng.uniform(-1.5, 1.5, size=60),
            rng.uniform(4.5, 8.0, size=60),
        ]
    )

    cam = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        width=800,
        height=600,
        params=[600.0, 400.0, 300.0],
    )

    def _look_at(cam_center: np.ndarray) -> pycolmap.Rigid3d:
        # Camera looks roughly toward (0, 0, 6)
        z = np.array([0.0, 0.0, 6.0]) - cam_center
        z /= np.linalg.norm(z)
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_cw = np.vstack([x, y, z])
        t_cw = -R_cw @ cam_center
        return pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw), t_cw)

    poses = {
        "db0.jpg": _look_at(np.array([-1.2, 0.0, 0.0])),
        "db1.jpg": _look_at(np.array([0.0, 0.2, -0.1])),
        "db2.jpg": _look_at(np.array([1.2, -0.1, 0.1])),
        "query.jpg": _look_at(np.array([0.5, 0.3, -0.2])),
    }

    # Project all 60 points into each camera; add 10 random outlier keypoints at the end
    kpts_hloc: dict[str, np.ndarray] = {}
    for name, pose in poses.items():
        uv_colmap = []
        for X in pts3d:
            X_c = pose * X
            uv = cam.img_from_cam(X_c)
            assert uv is not None
            uv_colmap.append(uv)
        outliers = np.column_stack(
            [rng.uniform(20, 780, size=10), rng.uniform(20, 580, size=10)]
        )
        # hloc stores keypoints in 0-centered pixel coords (COLMAP - 0.5)
        kpts_hloc[name] = (
            np.vstack([np.array(uv_colmap), outliers]).astype(np.float32) - 0.5
        )

    # Write reference SfM model (db0, db1, db2) with poses only (for triangulation)
    ref_dir = tmp_path / "ref_sfm"
    ref_dir.mkdir()
    ref = pycolmap.Reconstruction()
    ref.add_camera_with_trivial_rig(cam)
    for img_id, name in enumerate(["db0.jpg", "db1.jpg", "db2.jpg"], start=1):
        img = pycolmap.Image(image_id=img_id, camera_id=1, name=name)
        ref.add_image_with_trivial_frame(img, poses[name])
    ref.write_binary(ref_dir)

    # Write features H5
    feats_path = tmp_path / "feats.h5"
    with h5py.File(feats_path, "w") as f:
        for name, kps in kpts_hloc.items():
            grp = f.create_group(name)
            grp.create_dataset("keypoints", data=kps)

    # Write pairs & matches (60 true correspondences + 5 gross outlier matches)
    pairs_path = tmp_path / "pairs.txt"
    db_pairs = [("db0.jpg", "db1.jpg"), ("db1.jpg", "db2.jpg"), ("db0.jpg", "db2.jpg")]
    pairs_path.write_text("\n".join(f"{a} {b}" for a, b in db_pairs) + "\n")

    matches_path = tmp_path / "matches.h5"
    with h5py.File(matches_path, "w") as f:
        for a, b in db_pairs + [
            ("query.jpg", n) for n in ["db0.jpg", "db1.jpg", "db2.jpg"]
        ]:
            m0 = np.full(70, -1, dtype=np.int32)
            m0[:60] = np.arange(60, dtype=np.int32)
            # 5 gross outlier matches among the extra keypoints
            m0[60:65] = np.array([64, 63, 62, 61, 60], dtype=np.int32)
            s0 = np.ones(70, dtype=np.float32)
            grp = f.create_group(names_to_pair(a, b))
            grp.create_dataset("matches0", data=m0)
            grp.create_dataset("matching_scores0", data=s0)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return (
        ref_dir,
        images_dir,
        pairs_path,
        feats_path,
        matches_path,
        poses["query.jpg"],
        pts3d,
    )


def test_parse_option_args() -> None:
    opts = pycolmap.IncrementalPipelineOptions()
    parsed = triangulation.parse_option_args(
        [
            "mapper.init_min_num_inliers=50",
            "min_num_matches=12",
            "image_path=/tmp/foo",
            "ba_local_backend=CERES",
        ],
        opts,
    )
    assert parsed["mapper"]["init_min_num_inliers"] == 50
    assert parsed["min_num_matches"] == 12
    assert Path(parsed["image_path"]) == Path("/tmp/foo")
    assert parsed["ba_local_backend"] == pycolmap.BundleAdjustmentBackend.CERES

    with pytest.raises(ValueError, match="Unknown option"):
        triangulation.parse_option_args(["does_not_exist=1"], opts)
    with pytest.raises(ValueError, match="Could not parse value"):
        triangulation.parse_option_args(
            ["mapper.init_min_num_inliers=not_an_int"], opts
        )


@pytest.mark.parametrize("skip_gv", [False, True])
def test_triangulation_and_localize_sfm(tmp_path: Path, skip_gv: bool) -> None:
    (
        ref_dir,
        images_dir,
        pairs_path,
        feats_path,
        matches_path,
        gt_query_pose,
        gt_pts3d,
    ) = _build_synthetic_scene(tmp_path)

    sfm_dir = tmp_path / f"triangulated_{skip_gv}"
    rec = triangulation.main(
        sfm_dir,
        ref_dir,
        images_dir,
        pairs_path,
        feats_path,
        matches_path,
        skip_geometric_verification=skip_gv,
        verbose=False,
    )
    assert rec.num_reg_images() == 3
    assert rec.num_points3D() >= 55
    # Verify triangulated points accurately recover ground-truth 3D positions
    est_xyz = np.array([p.xyz for p in rec.points3D.values()])
    min_dists = np.min(
        np.linalg.norm(est_xyz[:, None, :] - gt_pts3d[None, :, :], axis=-1), axis=1
    )
    assert np.median(min_dists) < 1e-2

    # Verify QueryLocalizer config validation & non-mutating ransac_thresh
    with pytest.raises(ValueError, match="Unknown QueryLocalizer config keys"):
        localize_sfm.QueryLocalizer(rec, {"unknown": {}})

    # Localize query image with both covisibility_clustering=False and True
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("query.jpg SIMPLE_PINHOLE 800 600 600.0 400.0 300.0\n")
    loc_pairs = tmp_path / "loc_pairs.txt"
    loc_pairs.write_text("query.jpg db0.jpg\nquery.jpg db1.jpg\nquery.jpg db2.jpg\n")
    for clustering in (False, True):
        results_path = tmp_path / f"results_{skip_gv}_{clustering}.txt"
        localize_sfm.main(
            rec,
            queries_path,
            loc_pairs,
            feats_path,
            matches_path,
            results_path,
            ransac_thresh=8.0,
            covisibility_clustering=clustering,
        )
        line = results_path.read_text().strip().split()
        assert line[0] == "query.jpg"
        qw, qx, qy, qz, tx, ty, tz = [float(v) for v in line[1:]]
        est_pose = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.array([qx, qy, qz, qw])), np.array([tx, ty, tz])
        )
        assert np.linalg.norm(est_pose.translation - gt_query_pose.translation) < 1e-2
        R_err = est_pose.rotation.matrix().T @ gt_query_pose.rotation.matrix()
        angle_deg = np.rad2deg(
            np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        )
        assert angle_deg < 0.2

        # Pickle log round-trip
        with open(f"{results_path}_logs.pkl", "rb") as f:
            logs = pickle.load(f)
        loc_log = logs["loc"]["query.jpg"]
        pnp_ret = (
            loc_log["log_clusters"][loc_log["best_cluster"]]["PnP_ret"]
            if clustering
            else loc_log["PnP_ret"]
        )
        assert pnp_ret["num_inliers"] >= 50

    # Empty db_ids returns None cleanly
    localizer = localize_sfm.QueryLocalizer(rec)
    q_cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE", width=800, height=600, params=[600.0, 400.0, 300.0]
    )
    ret_empty, log_empty = localize_sfm.pose_from_cluster(
        localizer, "query.jpg", q_cam, [], feats_path, matches_path
    )
    assert ret_empty is None
    assert log_empty["num_matches"] == 0

    # Out-of-bounds match index fails loudly with IndexError
    bad_matches = tmp_path / "bad_matches.h5"
    with h5py.File(bad_matches, "w") as f:
        grp = f.create_group(names_to_pair("query.jpg", "db0.jpg"))
        grp.create_dataset("matches0", data=np.array([9999], dtype=np.int32))
        grp.create_dataset("matching_scores0", data=np.array([1.0], dtype=np.float32))
    with pytest.raises(IndexError):
        localize_sfm.pose_from_cluster(
            localizer, "query.jpg", q_cam, [1], feats_path, bad_matches
        )


def test_localize_inloc_empty_db_returns_none(tmp_path: Path) -> None:
    import cv2

    cv2.imwrite(str(tmp_path / "q.jpg"), np.zeros((480, 640, 3), dtype=np.uint8))
    feats_path = tmp_path / "feats.h5"
    with h5py.File(feats_path, "w") as f:
        grp = f.create_group("q.jpg")
        grp.create_dataset("keypoints", data=np.zeros((5, 2), dtype=np.float32))
        grp.create_dataset("image_size", data=np.array([640, 480], dtype=np.int32))
    matches_path = tmp_path / "matches.h5"
    with h5py.File(matches_path, "w"):
        pass

    with h5py.File(feats_path, "r") as f_feat, h5py.File(matches_path, "r") as f_match:
        ret, mkpq, mkpr, mkp3d, indices, num_matches = localize_inloc.pose_from_cluster(
            tmp_path,
            "q.jpg",
            [],
            f_feat,
            f_match,
            skip=None,
        )
    assert ret is None
    assert len(mkpq) == 0
    assert len(mkpr) == 0
    assert len(mkp3d) == 0
    assert len(indices) == 0
    assert num_matches == 0
