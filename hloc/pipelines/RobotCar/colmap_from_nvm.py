import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from ...colmap_from_nvm import (
    camera_center_to_translation,
    recover_database_images_and_ids,
)

logger = logging.getLogger(__name__)


def read_nvm_model(
    nvm_path: Path,
    database_path: Path,
    image_ids: dict[str, int],
    camera_ids: dict[str, int],
    skip_points: bool = False,
) -> pycolmap.Reconstruction:
    # Extract the intrinsics from the db file instead of the NVM model
    reconstruction = pycolmap.Reconstruction()
    with pycolmap.Database.open(database_path) as db:
        for camera in db.read_all_cameras():
            reconstruction.add_camera_with_trivial_rig(camera)

    with open(nvm_path) as nvm_f:
        line = nvm_f.readline()
        while line == "\n" or line.startswith("NVM_V3"):
            line = nvm_f.readline()
        num_images = int(line)

        logger.info(f"Reading {num_images} images...")
        image_idx_to_db_image_id = []
        image_data = []
        i = 0
        while i < num_images:
            line = nvm_f.readline()
            if line == "\n":
                continue
            data = line.strip("\n").lstrip("./").split(" ")
            image_data.append(data)
            image_idx_to_db_image_id.append(image_ids[data[0]])
            i += 1

        line = nvm_f.readline()
        while line == "\n":
            line = nvm_f.readline()
        num_points = int(line)

        if skip_points:
            logger.info(f"Skipping {num_points} points.")
            num_points = 0
        else:
            logger.info(f"Reading {num_points} points...")
        points_data = []
        image_idx_to_keypoints = defaultdict(list)
        i = 0
        pbar = tqdm(total=num_points, unit="pts")
        while i < num_points:
            line = nvm_f.readline()
            if line == "\n":
                continue

            data = line.strip("\n").split(" ")
            x, y, z, r, g, b, num_observations = data[:7]
            track_elements = []
            for j in range(int(num_observations)):
                s = 7 + 4 * j
                img_index, kp_index, kx, ky = data[s : s + 4]
                image_idx_to_keypoints[int(img_index)].append(
                    (int(kp_index), float(kx), float(ky))
                )
                db_image_id = image_idx_to_db_image_id[int(img_index)]
                track_elements.append(pycolmap.TrackElement(db_image_id, int(kp_index)))

            points_data.append(
                (
                    i,
                    np.array([x, y, z], float),
                    np.array([r, g, b], np.uint8),
                    track_elements,
                )
            )

            i += 1
            pbar.update(1)
        pbar.close()

    logger.info("Parsing image data...")
    for i, data in enumerate(image_data):
        # Skip the focal length. Skip the distortion and terminal 0.
        name, _, qw, qx, qy, qz, cx, cy, cz, _, _ = data
        qvec = np.array([qw, qx, qy, qz], float)
        qvec /= np.linalg.norm(qvec)
        c = np.array([cx, cy, cz], float)
        t = camera_center_to_translation(c, qvec)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(qvec[[1, 2, 3, 0]]), t)

        if i in image_idx_to_keypoints:
            # NVM only stores triangulated 2D keypoints: add dummy ones
            keypoints = image_idx_to_keypoints[i]
            point2D_idxs = np.array([d[0] for d in keypoints])
            tri_xys = np.array([[x, y] for _, x, y in keypoints])

            num_2Dpoints = int(max(point2D_idxs)) + 1
            xys = np.zeros((num_2Dpoints, 2), float)
            xys[point2D_idxs] = tri_xys
        else:
            xys = np.zeros((0, 2), float)

        image_id = image_ids[name]
        image = pycolmap.Image(
            image_id=image_id,
            camera_id=camera_ids[name],
            name=name.replace("png", "jpg"),  # some hack required for RobotCar
            keypoints=xys,
        )
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    for point_id, xyz, rgb, track_elements in points_data:
        point3D = pycolmap.Point3D(
            xyz=xyz,
            color=rgb,
            error=1.0,  # fake
            track=pycolmap.Track(track_elements),
        )
        reconstruction.add_point3D_with_id(point_id, point3D)

    return reconstruction


def main(nvm: Path, database: Path, output: Path, skip_points: bool = False) -> None:
    assert nvm.exists(), nvm
    assert database.exists(), database

    image_ids, camera_ids = recover_database_images_and_ids(database)

    logger.info("Reading the NVM model...")
    model = read_nvm_model(
        nvm, database, image_ids, camera_ids, skip_points=skip_points
    )

    logger.info("Writing the COLMAP model...")
    output.mkdir(exist_ok=True, parents=True)
    model.write_binary(output)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvm", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip_points", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
