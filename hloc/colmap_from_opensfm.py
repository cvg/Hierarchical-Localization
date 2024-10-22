import argparse
import json
import math
import typing

from pathlib import Path
from tqdm import tqdm

import numpy as np

from hloc import logger
from hloc.utils.read_write_model import (
    CAMERA_MODEL_NAMES,
    Camera,
    Image,
    Point3D,
    write_model
)


def angle_axis_to_quaternion(angle_axis: np.ndarray) -> typing.List[float]:
    angle = np.linalg.norm(angle_axis)
    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle
    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)
    return np.array([qw, float(qx), float(qy), float(qz)])


def read_opensfm_model(opensfm_path):
    logger.info("Reading OpenSfM reconstruction...")
    with open(opensfm_path / "reconstruction.json", "r") as fin:
        reconstructions_json = json.load(fin)
    assert len(reconstructions_json) == 1
    reconstruction_json = reconstructions_json[0]
    logger.info("Reading cameras...")
    camera_ids_map = {}
    cameras = {}
    for idx, (key, value) in enumerate(reconstruction_json["cameras"].items()):
        projection_type = value["projection_type"]
        assert projection_type == "spherical"
        camera_model = CAMERA_MODEL_NAMES["SPHERICAL"]
        camera_id = idx
        camera_ids_map[key] = camera_id
        width = int(value.get("width", 0))
        height = int(value.get("height", 0))
        camera = Camera(
            id = camera_id,
            model = camera_model.model_name,
            width = width,
            height = height,
            params = []
        )
        cameras[camera_id] = camera
    logger.info("Reading images...")
    images = {}
    for idx, (key, value) in enumerate(reconstruction_json["shots"].items()):
        camera_id = camera_ids_map[value["camera"]]
        image_id = idx
        image_name = key
        rvec = value["rotation"]
        tvec = value["translation"]
        qvec = angle_axis_to_quaternion(rvec)
        tvec_arr = np.array([tvec[0], tvec[1], tvec[2]])
        xys = np.zeros((0, 2), float)
        point3d_ids = np.full(0, -1, int)
        image = Image(
            id = image_id,
            qvec = qvec,
            tvec = tvec_arr,
            camera_id = camera_id,
            name = image_name,
            xys = xys,
            point3D_ids = point3d_ids,
        )
        images[image_id] = image
    logger.info("Reading points...")
    point3d_ids_map = {}
    points3d = {}
    num_points = len(reconstruction_json["points"])
    pbar = tqdm(total=num_points, unit="pts")
    for idx, (key, value) in enumerate(reconstruction_json["points"].items()):
        point3d_id = key
        point3d_ids_map[point3d_id] = idx
        coordinates = value["coordinates"]
        color = value["color"]
        xyz = np.array([coordinates[0], coordinates[1], coordinates[2]], float)
        color_arr = np.array([color[0], color[1], color[2]], int)
        point = Point3D(
            id = idx,
            xyz = xyz,
            rgb = color_arr,
            error = 1.0,  # fake
            image_ids = np.array([]),
            point2D_idxs = np.array([]),
        )
        points3d[idx] = point
        pbar.update(1)
    pbar.close()
    return cameras, images, points3d


def main(opensfm_path, output):
    assert opensfm_path.exists(), opensfm_path
    logger.info("Reading the OpenSfM model...")
    model = read_opensfm_model(opensfm_path)
    
    logger.info("Writing the COLMAP model...")
    output.mkdir(exist_ok=True, parents=True)
    write_model(*model, path=str(output), ext=".bin")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensfm-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.opensfm_path, args.output)
