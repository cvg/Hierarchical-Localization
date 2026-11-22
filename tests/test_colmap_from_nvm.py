import importlib
from pathlib import Path

import numpy as np
import pycolmap

from hloc import colmap_from_nvm
from hloc.pipelines.Cambridge import utils as cambridge_utils
from hloc.pipelines.RobotCar import colmap_from_nvm as robotcar_colmap_from_nvm

seven_scenes_utils = importlib.import_module("hloc.pipelines.7Scenes.utils")
seven_scenes_gt = importlib.import_module("hloc.pipelines.7Scenes.create_gt_sfm")


def _write_synthetic_nvm_and_db(
    tmp_path: Path,
) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
    db_path = tmp_path / "db.db"
    with pycolmap.Database.open(db_path) as db:
        cam0_id = db.write_camera(
            pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=640,
                height=480,
                params=[500.0, 320.0, 240.0],
            )
        )
        cam1_id = db.write_camera(
            pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=640,
                height=480,
                params=[500.0, 320.0, 240.0],
            )
        )
        db.write_image(pycolmap.Image(name="img0.jpg", camera_id=cam0_id))
        db.write_image(pycolmap.Image(name="img1.jpg", camera_id=cam1_id))

    intrinsics_path = tmp_path / "intrinsics.txt"
    intrinsics_path.write_text(
        "img0.jpg SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n"
        "img1.jpg SIMPLE_PINHOLE 640 480 500.0 320.0 240.0\n"
    )

    # Deliberately unnormalized wxyz quaternion (2x unit) for img1 (90 deg around Z)
    c0 = np.array([0.0, 0.0, 0.0])
    c1 = np.array([2.0, -1.0, 0.5])
    q1_wxyz_unnorm = 2.0 * np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    q1_str = " ".join(map(str, q1_wxyz_unnorm))
    nvm_path = tmp_path / "model.nvm"
    nvm_path.write_text(
        "NVM_V3\n\n"
        "2\n"
        f"img0.jpg 500.0 1.0 0.0 0.0 0.0 {c0[0]} {c0[1]} {c0[2]} 0.0 0\n"
        f"img1.jpg 500.0 {q1_str} {c1[0]} {c1[1]} {c1[2]} 0.0 0\n\n"
        "1\n"
        "0.5 0.25 4.0 100 150 200 2 0 0 382.5 271.25 1 0 120.0 210.0\n"
    )
    return nvm_path, intrinsics_path, db_path, c0, c1


def test_colmap_from_nvm_and_robotcar_roundtrip(tmp_path: Path) -> None:
    nvm_path, intrinsics_path, db_path, c0, c1 = _write_synthetic_nvm_and_db(tmp_path)

    out_dir = tmp_path / "out_nvm"
    colmap_from_nvm.main(nvm_path, intrinsics_path, db_path, out_dir)
    rec = pycolmap.Reconstruction(out_dir)
    assert rec.num_reg_images() == 2
    assert rec.num_points3D() == 1

    by_name = {img.name: img for img in rec.images.values()}
    assert np.allclose(by_name["img0.jpg"].projection_center(), c0, atol=1e-6)
    assert np.allclose(by_name["img1.jpg"].projection_center(), c1, atol=1e-6)
    # Verify rotation matrix is orthonormal despite unnormalized input quaternion
    R1 = by_name["img1.jpg"].cam_from_world().rotation.matrix()
    assert np.allclose(R1 @ R1.T, np.eye(3), atol=1e-6)
    pt = next(iter(rec.points3D.values()))
    assert np.allclose(pt.xyz, [0.5, 0.25, 4.0])
    assert pt.track.length() == 2

    # RobotCar variant reads cameras from the database
    out_rc = tmp_path / "out_robotcar"
    robotcar_colmap_from_nvm.main(nvm_path, db_path, out_rc)
    rec_rc = pycolmap.Reconstruction(out_rc)
    assert rec_rc.num_reg_images() == 2
    assert rec_rc.num_points3D() == 1
    by_name_rc = {img.name: img for img in rec_rc.images.values()}
    assert np.allclose(by_name_rc["img1.jpg"].projection_center(), c1, atol=1e-6)


def test_cambridge_and_7scenes_model_helpers(tmp_path: Path) -> None:
    full_dir = tmp_path / "full_model"
    full_dir.mkdir()
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_RADIAL",
        width=640,
        height=480,
        params=[500.0, 320.0, 240.0, 0.0],
    )
    rec.add_camera_with_trivial_rig(cam)
    for img_id, name, tx in [
        (1, "train0.jpg", 0.0),
        (2, "train1.jpg", -1.0),
        (3, "test0.jpg", -2.0),
    ]:
        img = pycolmap.Image(
            image_id=img_id,
            camera_id=1,
            name=name,
            keypoints=np.array([[100.0, 100.0], [200.0, 200.0]]),
        )
        rec.add_image_with_trivial_frame(
            img, pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([tx, 0.0, 0.0]))
        )
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
        pycolmap.Track([pycolmap.TrackElement(3, 1)]),
    )
    full_txt_dir = tmp_path / "full_model_txt"
    full_txt_dir.mkdir()
    rec.write_binary(full_dir)
    rec.write_text(full_txt_dir)

    # Cambridge query list with intrinsics (auto-discovers both binary & text models)
    for model_dir in (full_dir, full_txt_dir):
        q_list = tmp_path / f"queries_{model_dir.name}.txt"
        cambridge_utils.create_query_list_with_intrinsics(model_dir, q_list)
        lines = q_list.read_text().strip().splitlines()
        assert len(lines) == 3
        assert any(line.startswith("test0.jpg SIMPLE_RADIAL 640 480") for line in lines)

    # 7Scenes reference model filtering via blacklist (auto-discovers bin & txt)
    blacklist = tmp_path / "blacklist.txt"
    blacklist.write_text("test0.jpg\n")
    for model_dir in (full_dir, full_txt_dir):
        ref_dir = tmp_path / f"ref_model_{model_dir.name}"
        seven_scenes_utils.create_reference_sfm(model_dir, ref_dir, blacklist=blacklist)
        ref = pycolmap.Reconstruction(ref_dir)
        assert {img.name for img in ref.images.values()} == {"train0.jpg", "train1.jpg"}
        # Only the point observed by training images survives, with track trimmed to 2
        assert ref.num_points3D() == 1
        assert next(iter(ref.points3D.values())).track.length() == 2


def test_7scenes_correct_sfm_with_gt_depth(tmp_path: Path) -> None:
    model_in = tmp_path / "sfm_in"
    model_out = tmp_path / "sfm_out"
    depth_dir = tmp_path / "depths"
    model_in.mkdir()
    depth_dir.mkdir()

    import PIL.Image

    # 2 cameras looking along +Z: valid 4.0m (4000mm) depth everywhere in img0;
    # img1 has valid depth at kp0 (50, 50) and invalid depth (0) at kp1 (80, 50).
    d0 = np.full((100, 100), 4000.0, dtype=np.float32)
    d1 = np.full((100, 100), 4000.0, dtype=np.float32)
    d1[:, 60:] = 0.0
    PIL.Image.fromarray(d0).save(depth_dir / "seq01_frame-000000.pose.depth.tiff")
    PIL.Image.fromarray(d1).save(depth_dir / "seq01_frame-000001.pose.depth.tiff")

    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        width=100,
        height=100,
        params=[100.0, 50.0, 50.0],
    )
    rec.add_camera_with_trivial_rig(cam)
    for img_id, name in [
        (1, "seq-01/frame-000000.color.png"),
        (2, "seq-01/frame-000001.color.png"),
    ]:
        img = pycolmap.Image(
            image_id=img_id,
            camera_id=1,
            name=name,
            keypoints=np.array([[50.0, 50.0], [80.0, 50.0]]),
        )
        rec.add_image_with_trivial_frame(img, pycolmap.Rigid3d())

    # Point 1 observed at (50, 50) in both images -> depth 4.0m -> xyz=(0, 0, 4).
    p1 = rec.add_point3D(
        np.array([0.0, 0.0, 10.0]),
        pycolmap.Track([pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0)]),
    )
    # Point 2 observed at (80, 50): valid in img0, invalid in img1 (pruned).
    p2 = rec.add_point3D(
        np.array([3.0, 0.0, 10.0]),
        pycolmap.Track([pycolmap.TrackElement(1, 1), pycolmap.TrackElement(2, 1)]),
    )
    rec.write_binary(model_in)

    seven_scenes_gt.correct_sfm_with_gt_depth(model_in, depth_dir, model_out)
    out = pycolmap.Reconstruction(model_out)
    assert p1 in out.points3D
    assert np.allclose(out.points3D[p1].xyz, [0.0, 0.0, 4.0], atol=1e-2)
    assert out.points3D[p1].track.length() == 2
    assert p2 in out.points3D
    assert out.points3D[p2].track.length() == 1
    assert not out.images[2].point2D(1).has_point3D()
