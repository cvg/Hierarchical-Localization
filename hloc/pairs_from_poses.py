import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pycolmap
import scipy.spatial

from . import logger
from .pairs_from_retrieval import pairs_from_score_matrix

DEFAULT_ROT_THRESH = 30  # in degrees


def get_pairwise_distances(
    images: Mapping[int, pycolmap.Image] | pycolmap.ImageMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.array([id_ for id_, img in images.items() if img.has_pose])
    Rs = []
    ts = []
    for id_ in ids:
        world_from_cam = images[int(id_)].cam_from_world().inverse()
        Rs.append(world_from_cam.rotation.matrix())
        ts.append(world_from_cam.translation)
    Rs_arr = np.stack(Rs, 0)
    ts_arr = np.stack(ts, 0)

    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts_arr))

    # Instead of computing the angle between two camera orientations,
    # we compute the angle between the principal axes, as two images rotated
    # around their principal axis still observe the same scene.
    axes = Rs_arr[:, :, -1]
    dots = np.einsum("mi,ni->mn", axes, axes, optimize=True)
    dR = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    return ids, dist, dR


def main(
    model: Path,
    output: Path,
    num_matched: int,
    rotation_threshold: float = DEFAULT_ROT_THRESH,
) -> None:
    logger.info("Reading the COLMAP model...")
    images = pycolmap.Reconstruction(model).images

    logger.info(f"Obtaining pairwise distances between {len(images)} images...")
    ids, dist, dR = get_pairwise_distances(images)
    scores = -dist

    invalid = dR >= rotation_threshold
    np.fill_diagonal(invalid, True)
    pairs_idx = pairs_from_score_matrix(scores, invalid, num_matched)
    pairs = [(images[int(ids[i])].name, images[int(ids[j])].name) for i, j in pairs_idx]

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join(p) for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    parser.add_argument("--rotation_threshold", default=DEFAULT_ROT_THRESH, type=float)
    args = parser.parse_args()
    main(**args.__dict__)
