import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from scipy.io import loadmat
from tqdm import tqdm

from . import logger
from .utils.io import write_poses
from .utils.parsers import names_to_pair, parse_retrieval


def interpolate_scan(scan: np.ndarray, kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w, _c = scan.shape
    kp_norm = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp_norm > -1) and np.all(kp_norm < 1)
    scan_t = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp_t = torch.from_numpy(kp_norm)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(scan_t, kp_t, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan_t, kp_t, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    return kp3d, valid.numpy()


def get_scan_pose(dataset_dir: Path, rpath: str) -> np.ndarray:
    split_image_rpath = rpath.split("/")
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = Path(
        dataset_dir,
        "database/alignments",
        floor_name,
        f"transformations/{building_name}_trans_{scan_id}.txt",
    )
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array(
        [
            np.fromstring(raw_lines[7], sep=" "),
            np.fromstring(raw_lines[8], sep=" "),
            np.fromstring(raw_lines[9], sep=" "),
            np.fromstring(raw_lines[10], sep=" "),
        ]
    )

    return P_after_GICP


def pose_from_cluster(
    dataset_dir: Path,
    q: str,
    retrieved: Sequence[str],
    feature_file: h5py.File,
    match_file: h5py.File,
    skip: int | None = None,
) -> tuple[
    dict[str, Any] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    img = cv2.imread(str(dataset_dir / q))
    assert img is not None, dataset_dir / q
    height, width = img.shape[:2]
    cx = 0.5 * width
    cy = 0.5 * height
    focal_length = 4032.0 * 28.0 / 36.0

    all_mkpq_list: list[np.ndarray] = []
    all_mkpr_list: list[np.ndarray] = []
    all_mkp3d_list: list[np.ndarray] = []
    all_indices_list: list[np.ndarray] = []
    kpq = feature_file[q]["keypoints"].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        kpr = feature_file[r]["keypoints"].__array__()
        pair = names_to_pair(q, r)
        m = match_file[pair]["matches0"].__array__()
        v = m > -1

        if skip and (np.count_nonzero(v) < skip):
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        num_matches += len(mkpq)

        scan_r = loadmat(Path(dataset_dir, r + ".mat"))["XYZcut"]
        mkp3d, valid = interpolate_scan(scan_r, mkpr)
        Tr = get_scan_pose(dataset_dir, r)
        mkp3d = (Tr[:3, :3] @ mkp3d.T + Tr[:3, -1:]).T

        all_mkpq_list.append(mkpq[valid])
        all_mkpr_list.append(mkpr[valid])
        all_mkp3d_list.append(mkp3d[valid])
        all_indices_list.append(np.full(np.count_nonzero(valid), i))

    if len(all_mkpq_list) == 0:
        return (
            None,
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            np.zeros((0, 3)),
            np.zeros((0,), dtype=int),
            num_matches,
        )

    all_mkpq = np.concatenate(all_mkpq_list, 0)
    all_mkpr = np.concatenate(all_mkpr_list, 0)
    all_mkp3d = np.concatenate(all_mkp3d_list, 0)
    all_indices = np.concatenate(all_indices_list, 0)

    cfg: dict[str, Any] = {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [focal_length, cx, cy],
    }
    camera = pycolmap.Camera(**cfg)
    estimation_options = pycolmap.AbsolutePoseEstimationOptions()
    estimation_options.ransac.max_error = 48
    ret = pycolmap.estimate_and_refine_absolute_pose(
        all_mkpq, all_mkp3d, camera, estimation_options
    )
    if ret is not None:
        ret["cfg"] = cfg
        ret["camera"] = camera
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(
    dataset_dir: Path,
    retrieval: Path,
    features: Path,
    matches: Path,
    results: Path,
    skip_matches: int | None = None,
) -> None:
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())

    poses: dict[str, pycolmap.Rigid3d] = {}
    logs: dict[str, Any] = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    with (
        h5py.File(features, "r", libver="latest") as feature_file,
        h5py.File(matches, "r", libver="latest") as match_file,
    ):
        for q in tqdm(queries):
            db = retrieval_dict[q]
            ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
                dataset_dir, q, db, feature_file, match_file, skip_matches
            )

            if ret is not None:
                poses[q] = ret["cam_from_world"]
            logs["loc"][q] = {
                "db": db,
                "PnP_ret": ret,
                "keypoints_query": mkpq,
                "keypoints_db": mkpr,
                "3d_points": mkp3d,
                "indices_db": indices,
                "num_matches": num_matches,
            }

    logger.info(f"Writing poses to {results}...")
    write_poses(poses, results, prepend_camera_name=False)

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--skip_matches", type=int)
    args = parser.parse_args()
    main(**args.__dict__)
