import glob
import logging
import os
from pathlib import Path

import numpy as np
import pycolmap

from ...utils.parsers import parse_retrieval

logger = logging.getLogger(__name__)


def get_timestamps(files, idx):
    """Extract timestamps from a pose or relocalization file."""
    lines = []
    for p in files.parent.glob(files.name):
        with open(p) as f:
            lines += f.readlines()
    timestamps = set()
    for line in lines:
        line = line.rstrip("\n")
        if line[0] == "#" or line == "":
            continue
        ts = line.replace(",", " ").split()[idx]
        timestamps.add(ts)
    return timestamps


def delete_unused_images(root, timestamps):
    """Delete all images in root if they are not contained in timestamps."""
    images = glob.glob((root / "**/*.png").as_posix(), recursive=True)
    deleted = 0
    for image in images:
        ts = Path(image).stem
        if ts not in timestamps:
            os.remove(image)
            deleted += 1
    logger.info(f"Deleted {deleted} images in {root}.")


def camera_from_calibration_file(id_, path):
    """Create a COLMAP camera from an MLAD calibration file."""
    with open(path, "r") as f:
        data = f.readlines()
    model, fx, fy, cx, cy = data[0].split()[:5]
    width, height = data[1].split()
    assert model == "Pinhole"
    model_name = "PINHOLE"
    params = [float(i) for i in [fx, fy, cx, cy]]
    camera = pycolmap.Camera(
        camera_id=id_,
        model=model_name,
        width=int(width),
        height=int(height),
        params=params,
    )
    return camera


def parse_poses(path, colmap=False):
    """Parse a list of poses in COLMAP or MLAD quaternion convention."""
    poses = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            if line[0] == "#" or line == "":
                continue
            data = line.replace(",", " ").split()
            ts, p = data[0], np.array(data[1:], float)
            if colmap:
                q, t = np.split(p, [4])
                q = q[[1, 2, 3, 0]]  # wxyz to xyzw
            else:
                t, q = np.split(p, [3])
            R = pycolmap.Rotation3d(q).matrix()
            poses.append((ts, R, t))
    return poses


def parse_relocalization(path, has_poses=False):
    """Parse a relocalization file, possibly with poses."""
    reloc = []
    with open(path) as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            if line[0] == "#" or line == "":
                continue
            data = line.replace(",", " ").split()
            out = data[:2]  # ref_ts, q_ts
            if has_poses:
                assert len(data) == 9
                t, q = np.split(np.array(data[2:], float), [3])
                R = pycolmap.Rotation3d(q).matrix()
                out += [R, t]
            reloc.append(out)
    return reloc


def build_empty_colmap_model(root, sfm_dir):
    """Build a COLMAP model with images and cameras only."""
    calibration = "Calibration/undistorted_calib_{}.txt"
    cam0 = camera_from_calibration_file(0, root / calibration.format(0))
    cam1 = camera_from_calibration_file(1, root / calibration.format(1))
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(cam0)
    reconstruction.add_camera_with_trivial_rig(cam1)

    T_0to1 = np.loadtxt(root / "Calibration/undistorted_calib_stereo.txt")
    poses = parse_poses(root / "poses.txt")
    id_ = 0
    for ts, R_cam0_to_w, t_cam0_to_w in poses:
        R_w_to_cam0 = R_cam0_to_w.T
        t_w_to_cam0 = -(R_w_to_cam0 @ t_cam0_to_w)

        R_w_to_cam1 = T_0to1[:3, :3] @ R_w_to_cam0
        t_w_to_cam1 = T_0to1[:3, :3] @ t_w_to_cam0 + T_0to1[:3, 3]

        for idx, (R_w_to_cam, t_w_to_cam) in enumerate(
            zip([R_w_to_cam0, R_w_to_cam1], [t_w_to_cam0, t_w_to_cam1])
        ):
            image = pycolmap.Image(
                image_id=id_,
                camera_id=idx,
                name=f"cam{idx}/{ts}.png",
            )
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(R_w_to_cam), t_w_to_cam
            )
            reconstruction.add_image_with_trivial_frame(image, cam_from_world)
            id_ += 1

    sfm_dir.mkdir(exist_ok=True, parents=True)
    reconstruction.write_binary(sfm_dir)


def generate_query_lists(timestamps, seq_dir, out_path):
    """Create a list of query images with intrinsics from timestamps."""
    cam0 = camera_from_calibration_file(
        0, seq_dir / "Calibration/undistorted_calib_0.txt"
    )
    intrinsics = [cam0.model.name, cam0.width, cam0.height] + cam0.params.tolist()
    intrinsics = [str(p) for p in intrinsics]
    data = map(lambda ts: " ".join([f"cam0/{ts}.png"] + intrinsics), timestamps)
    with open(out_path, "w") as f:
        f.write("\n".join(data))


def generate_localization_pairs(sequence, reloc, num, ref_pairs, out_path):
    """Create the matching pairs for the localization.
    We simply lookup the corresponding reference frame
    and extract its `num` closest frames from the existing pair list.
    """
    if "test" in sequence:
        # hard pairs will be overwritten by easy ones if available
        relocs = [str(reloc).replace("*", d) for d in ["hard", "moderate", "easy"]]
    else:
        relocs = [reloc]
    query_to_ref_ts = {}
    for reloc in relocs:
        with open(reloc, "r") as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                if line[0] == "#" or line == "":
                    continue
                ref_ts, q_ts = line.split()[:2]
                query_to_ref_ts[q_ts] = ref_ts

    ts_to_name = "cam0/{}.png".format
    ref_pairs = parse_retrieval(ref_pairs)
    loc_pairs = []
    for q_ts, ref_ts in query_to_ref_ts.items():
        ref_name = ts_to_name(ref_ts)
        selected = [ref_name] + ref_pairs[ref_name][: num - 1]
        loc_pairs.extend([" ".join((ts_to_name(q_ts), s)) for s in selected])
    with open(out_path, "w") as f:
        f.write("\n".join(loc_pairs))


def prepare_submission(results, relocs, poses_path, out_dir):
    """Obtain relative poses from estimated absolute and reference poses."""
    gt_poses = parse_poses(poses_path)
    all_T_ref0_to_w = {ts: (R, t) for ts, R, t in gt_poses}

    pred_poses = parse_poses(results, colmap=True)
    all_T_w_to_q0 = {Path(name).stem: (R, t) for name, R, t in pred_poses}

    for reloc in relocs.parent.glob(relocs.name):
        relative_poses = []
        reloc_ts = parse_relocalization(reloc)
        for ref_ts, q_ts in reloc_ts:
            R_w_to_q0, t_w_to_q0 = all_T_w_to_q0[q_ts]
            R_ref0_to_w, t_ref0_to_w = all_T_ref0_to_w[ref_ts]

            R_ref0_to_q0 = R_w_to_q0 @ R_ref0_to_w
            t_ref0_to_q0 = R_w_to_q0 @ t_ref0_to_w + t_w_to_q0

            tvec = t_ref0_to_q0.tolist()
            qvec = pycolmap.Rotation3d(R_ref0_to_q0).quat.tolist()  # xyzw

            out = [ref_ts, q_ts] + list(map(str, tvec)) + list(map(str, qvec))
            relative_poses.append(" ".join(out))

        out_path = out_dir / reloc.name
        with open(out_path, "w") as f:
            f.write("\n".join(relative_poses))
        logger.info(f"Submission file written to {out_path}.")


def evaluate_submission(submission_dir, relocs, ths=[0.1, 0.2, 0.5]):
    """Compute the relocalization recall from predicted and ground truth poses."""
    for reloc in relocs.parent.glob(relocs.name):
        poses_gt = parse_relocalization(reloc, has_poses=True)
        poses_pred = parse_relocalization(submission_dir / reloc.name, has_poses=True)
        poses_pred = {(ref_ts, q_ts): (R, t) for ref_ts, q_ts, R, t in poses_pred}

        error = []
        for ref_ts, q_ts, R_gt, t_gt in poses_gt:
            R, t = poses_pred[(ref_ts, q_ts)]
            e = np.linalg.norm(t - t_gt)
            error.append(e)

        error = np.array(error)
        recall = [np.mean(error <= th) for th in ths]
        s = f"Relocalization evaluation {submission_dir.name}/{reloc.name}\n"
        s += " / ".join([f"{th:>7}m" for th in ths]) + "\n"
        s += " / ".join([f"{100*r:>7.3f}%" for r in recall])
        logger.info(s)
