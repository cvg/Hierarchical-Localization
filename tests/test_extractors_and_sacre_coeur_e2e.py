import pickle
from pathlib import Path

import h5py
import numpy as np
import pycolmap
import torch

from hloc import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
    triangulation,
)
from hloc.extractors.dog import DoG
from hloc.matchers.nearest_neighbor import NearestNeighbor

SACRE_COEUR_DIR = (
    Path(__file__).resolve().parents[1] / "datasets" / "sacre_coeur" / "mapping"
)


def test_dog_extractor_blank_image_and_nn_matcher() -> None:
    dog = DoG({"options": {"max_num_features": 256}}).eval()
    blank = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    out = dog({"image": blank})
    assert out["keypoints"].shape == (1, 0, 2)
    assert out["descriptors"].shape == (1, 128, 0)
    assert out["scores"].shape == (1, 0)

    rng = np.random.default_rng(0)
    d0 = torch.from_numpy(rng.normal(size=(1, 128, 16)).astype(np.float32))
    d0 = torch.nn.functional.normalize(d0, dim=1)
    d1 = d0.clone()
    matcher = NearestNeighbor({"do_mutual_check": True, "ratio_threshold": 0.8}).eval()
    pred = matcher({"descriptors0": d0, "descriptors1": d1})
    assert torch.equal(pred["matches0"][0], torch.arange(16))


def test_sacre_coeur_end_to_end_sfm_triangulation_and_localization(
    tmp_path: Path,
) -> None:
    pycolmap.set_random_seed(42)
    all_images = sorted(p.name for p in SACRE_COEUR_DIR.glob("*.jpg"))
    assert len(all_images) == 10
    query_image = all_images[1]
    mapping_images = [im for im in all_images if im != query_image]

    # 1. Extract DoG SIFT features (no external weight downloads)
    feat_conf = extract_features.confs["sift"]
    features_path = extract_features.main(
        feat_conf,
        SACRE_COEUR_DIR,
        export_dir=tmp_path,
        image_list=mapping_images + [query_image],
    )
    with h5py.File(features_path, "r") as f:
        for name in mapping_images + [query_image]:
            assert f[name]["keypoints"].shape[0] > 50

    # 2. Generate exhaustive mapping pairs
    sfm_pairs = tmp_path / "pairs-sfm.txt"
    pairs_from_exhaustive.main(sfm_pairs, image_list=mapping_images)

    # 3. Match mapping pairs with NN-ratio
    match_conf = match_features.confs["NN-ratio"]
    sfm_matches = match_features.main(
        match_conf,
        sfm_pairs,
        features=features_path,
        matches=tmp_path / "matches-sfm.h5",
    )

    # 4. Reconstruct sparse 3D SfM model deterministically
    sfm_dir = tmp_path / "sfm"
    model = reconstruction.main(
        sfm_dir,
        SACRE_COEUR_DIR,
        sfm_pairs,
        features_path,
        sfm_matches,
        image_list=mapping_images,
        mapper_options={
            "random_seed": 42,
            "num_threads": 1,
            "mapper": {"init_min_num_inliers": 50},
        },
        verbose=False,
    )
    assert model is not None
    assert model.num_reg_images() >= 6
    assert model.num_points3D() >= 100

    # 5. Re-triangulate against the reconstructed reference poses using
    #    pycolmap.guided_geometric_verification
    reg_names = sorted(img.name for img in model.images.values())
    tri_pairs = tmp_path / "pairs-tri.txt"
    pairs_from_exhaustive.main(tri_pairs, image_list=reg_names)
    tri_dir = tmp_path / "sfm_triangulated"
    tri_model = triangulation.main(
        tri_dir,
        sfm_dir,
        SACRE_COEUR_DIR,
        tri_pairs,
        features_path,
        sfm_matches,
        verbose=False,
    )
    assert tri_model.num_reg_images() == model.num_reg_images()
    assert tri_model.num_points3D() >= 100

    # 6. Localize the held-out query image against the reconstructed model
    loc_pairs = tmp_path / "pairs-loc.txt"
    pairs_from_exhaustive.main(loc_pairs, image_list=[query_image], ref_list=reg_names)
    loc_matches = match_features.main(
        match_conf,
        loc_pairs,
        features=features_path,
        matches=tmp_path / "matches-loc.h5",
    )

    # Build query intrinsics using COLMAP's EXIF/default camera inference
    q_cam = pycolmap.infer_camera_from_image(SACRE_COEUR_DIR / query_image)
    queries_txt = tmp_path / "queries.txt"
    params_str = " ".join(map(str, q_cam.params.tolist()))
    queries_txt.write_text(
        f"{query_image} {q_cam.model.name} {q_cam.width} {q_cam.height} {params_str}\n"
    )

    estimated_poses: dict[bool, pycolmap.Rigid3d] = {}
    for clustering in (False, True):
        results_txt = tmp_path / f"loc_results_{clustering}.txt"
        localize_sfm.main(
            model,
            queries_txt,
            loc_pairs,
            features_path,
            loc_matches,
            results_txt,
            covisibility_clustering=clustering,
        )
        lines = results_txt.read_text().strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == query_image
        pose_vals = np.array([float(v) for v in parts[1:]])
        assert pose_vals.shape == (7,)
        assert np.all(np.isfinite(pose_vals))
        assert np.isclose(np.linalg.norm(pose_vals[:4]), 1.0, atol=1e-4)

        with open(f"{results_txt}_logs.pkl", "rb") as f:
            logs = pickle.load(f)
        loc_log = logs["loc"][query_image]
        pnp_ret = (
            loc_log["log_clusters"][loc_log["best_cluster"]]["PnP_ret"]
            if clustering
            else loc_log["PnP_ret"]
        )
        assert pnp_ret is not None
        assert pnp_ret["num_inliers"] >= 15

        qw, qx, qy, qz, tx, ty, tz = pose_vals
        estimated_poses[clustering] = pycolmap.Rigid3d(
            pycolmap.Rotation3d(np.array([qx, qy, qz, qw])), np.array([tx, ty, tz])
        )

    pose_no_cluster = estimated_poses[False]
    pose_cluster = estimated_poses[True]
    R_diff = pose_no_cluster.rotation.matrix().T @ pose_cluster.rotation.matrix()
    angle_diff_deg = np.rad2deg(
        np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    )
    assert angle_diff_deg < 5.0
    rel_t_diff = np.linalg.norm(
        pose_no_cluster.translation - pose_cluster.translation
    ) / max(float(np.linalg.norm(pose_no_cluster.translation)), 1.0)
    assert rel_t_diff < 0.1
