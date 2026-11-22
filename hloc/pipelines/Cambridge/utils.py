import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


def scale_sfm_images(full_model: Path, scaled_model: Path, image_dir: Path) -> None:
    """Duplicate the provided model and scale the camera intrinsics so that
    they match the original image resolution - makes everything easier.
    """
    logger.info("Scaling the COLMAP model to the original image size.")
    scaled_model.mkdir(exist_ok=True, parents=True)
    reconstruction = pycolmap.Reconstruction(full_model)

    scaled_camera_ids: set[int] = set()
    for image in reconstruction.images.values():
        name = image.name
        img = cv2.imread(str(image_dir / name))
        assert img is not None, image_dir / name
        h, w = img.shape[:2]

        cam_id = image.camera_id
        camera = reconstruction.cameras[cam_id]
        if cam_id in scaled_camera_ids:
            assert camera.width == w
            assert camera.height == h
            continue

        assert camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL
        sx = w / camera.width
        sy = h / camera.height
        assert sx == sy, (sx, sy)
        camera.rescale(w, h)
        scaled_camera_ids.add(cam_id)

    reconstruction.write(scaled_model)


def create_query_list_with_intrinsics(
    model: Path,
    out: Path,
    list_file: Path | None = None,
    image_dir: Path | None = None,
) -> None:
    """Create a list of query images with intrinsics from the colmap model."""
    reconstruction = pycolmap.Reconstruction(model)
    images = reconstruction.images
    cameras = reconstruction.cameras

    name2id = {image.name: i for i, image in images.items()}
    if list_file is None:
        names = list(name2id)
    else:
        with open(list_file) as f:
            names = f.read().rstrip().split("\n")
    data = []
    for name in names:
        image = images[name2id[name]]
        camera = cameras[image.camera_id]
        w, h, params = camera.width, camera.height, camera.params.copy()

        if image_dir is not None:
            # Check the original image size and rescale the camera intrinsics
            img = cv2.imread(str(image_dir / name))
            assert img is not None, image_dir / name
            h_orig, w_orig = img.shape[:2]
            assert camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL
            sx = w_orig / w
            sy = h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig
            params = params * np.array([sx, sx, sy, 1.0])

        p: list[Any] = [name, camera.model.name, w, h] + params.tolist()
        data.append(" ".join(map(str, p)))
    with open(out, "w") as f:
        f.write("\n".join(data))


def evaluate(
    model: Path,
    results: Path,
    list_file: Path | None = None,
    only_localized: bool = False,
) -> None:
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(results) as f:
        for line in f.read().rstrip().split("\n"):
            data = line.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            predictions[name] = (pycolmap.Rotation3d(q[[1, 2, 3, 0]]).matrix(), t)
    images = pycolmap.Reconstruction(model).images
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file) as f:
            test_names = f.read().rstrip().split("\n")

    errors_t_list: list[float] = []
    errors_R_list: list[float] = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            pose = image.cam_from_world()
            R_gt, t_gt = pose.rotation.matrix(), pose.translation
            R, t = predictions[name]
            e_t = float(np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0))
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = float(np.rad2deg(np.abs(np.arccos(cos))))
        errors_t_list.append(e_t)
        errors_R_list.append(e_R)

    errors_t = np.array(errors_t_list)
    errors_R = np.array(errors_R_list)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {results.name}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R, strict=True):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t * 100:.0f}cm, {th_R:.0f}deg : {ratio * 100:.2f}%"
    logger.info(out)
