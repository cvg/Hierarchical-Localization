from pathlib import Path

import numpy as np
import PIL.Image
import pycolmap
import torch
from tqdm import tqdm


def scene_coordinates(
    p2D: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    depth: np.ndarray,
    camera: pycolmap.Camera,
) -> np.ndarray:
    assert len(depth) == len(p2D)
    if len(p2D) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    p2D_norm = np.asarray(camera.cam_from_img(p2D), dtype=np.float64).reshape(-1, 2)
    p2D_h = np.concatenate([p2D_norm, np.ones_like(p2D_norm[:, :1])], 1)
    p3D_c = p2D_h * depth[:, None]
    p3D_w: np.ndarray = (p3D_c - t_w2c) @ R_w2c
    return p3D_w


def interpolate_depth(
    depth: np.ndarray, kp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    kp_norm = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp_norm > -1) and np.all(kp_norm < 1)
    depth_t = torch.from_numpy(depth)[None, None]
    kp_t = torch.from_numpy(kp_norm)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(depth_t, kp_t, align_corners=True, mode="bilinear")[
        0, :, 0
    ]
    interp_nn = torch.nn.functional.grid_sample(
        depth_t, kp_t, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid_t = ~torch.any(torch.isnan(interp), 0)

    interp_depth = interp.T.numpy().flatten()
    valid = valid_t.numpy()
    return interp_depth, valid


def image_path_to_rendered_depth_path(image_name: str) -> str:
    parts = image_name.split("/")
    name = "_".join(["".join(parts[0].split("-")), parts[1]])
    name = name.replace("color", "pose")
    name = name.replace("png", "depth.tiff")
    return name


def project_to_image(
    p3D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    camera: pycolmap.Camera,
    eps: float = 1e-4,
    pad: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if len(p3D) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    p3D = (p3D @ R.T) + t
    visible = p3D[:, -1] >= eps  # keep points in front of the camera
    p2D = np.asarray(camera.img_from_cam(p3D), dtype=np.float64).reshape(-1, 2)
    size = np.array([camera.width - pad - 1, camera.height - pad - 1])
    valid = ~np.any(np.isnan(p2D), -1) & np.all((p2D >= pad) & (p2D <= size), -1)
    valid &= visible
    return p2D[valid], valid


def correct_sfm_with_gt_depth(
    sfm_path: Path, depth_folder_path: Path, output_path: Path
) -> None:
    reconstruction = pycolmap.Reconstruction(sfm_path)
    for imgid, img in tqdm(reconstruction.images.items()):
        if not img.has_pose:
            continue
        image_name = img.name
        depth_name = image_path_to_rendered_depth_path(image_name)

        depth_img = PIL.Image.open(Path(depth_folder_path) / depth_name)
        depth = np.array(depth_img).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.nan

        pose = img.cam_from_world()
        R_w2c, t_w2c = pose.rotation.matrix(), pose.translation
        camera = reconstruction.cameras[img.camera_id]
        obs = [
            (p2d_idx, p2d.point3D_id)
            for p2d_idx, p2d in enumerate(img.points2D)
            if p2d.has_point3D()
        ]
        if len(obs) == 0:
            continue
        obs_p2d_idxs = np.array([idx for idx, _ in obs], dtype=int)
        obs_p3d_ids = np.array([pid for _, pid in obs], dtype=int)
        p3Ds = np.stack([reconstruction.points3D[pid].xyz for pid in obs_p3d_ids], 0)

        p2Ds, valids_projected = project_to_image(p3Ds, R_w2c, t_w2c, camera)
        interp_depth, valids_backprojected = interpolate_depth(depth, p2Ds)
        scs = scene_coordinates(
            p2Ds[valids_backprojected],
            R_w2c,
            t_w2c,
            interp_depth[valids_backprojected],
            camera,
        )

        valids = np.zeros(len(obs), dtype=bool)
        valids[valids_projected] = valids_backprojected

        for p2d_idx, p3did in zip(
            obs_p2d_idxs[~valids], obs_p3d_ids[~valids], strict=True
        ):
            p3d_id = int(p3did)
            img.reset_point3D_for_point2D(int(p2d_idx))
            point3D = reconstruction.points3D[p3d_id]
            point3D.track.delete_element(imgid, int(p2d_idx))
            if point3D.track.length() == 0:
                reconstruction.delete_point3D(p3d_id)

        valid_p3d_ids = obs_p3d_ids[valids]
        assert len(valid_p3d_ids) == len(scs), f"{len(scs)}, {len(valid_p3d_ids)}"
        for p3did, sc in zip(valid_p3d_ids, scs, strict=True):
            reconstruction.points3D[int(p3did)].xyz = sc

    output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(output_path)


if __name__ == "__main__":
    dataset = Path("datasets/7scenes")
    outputs = Path("outputs/7Scenes")

    SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    for scene in SCENES:
        sfm_path = outputs / scene / "sfm_superpoint+superglue"
        depth_path = dataset / f"depth/7scenes_{scene}/train/depth"
        output_path = outputs / scene / "sfm_superpoint+superglue+depth"
        correct_sfm_with_gt_depth(sfm_path, depth_path, output_path)
