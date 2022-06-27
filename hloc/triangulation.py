import argparse
import contextlib
import io
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pycolmap

from . import logger
from .utils.database import COLMAPDatabase
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval
from .utils.geometry import compute_epipolar_errors


class OutputCapture:
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                logger.error('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()


def create_db_from_model(reconstruction, database_path):
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(
            camera.model_id, camera.width, camera.height, camera.params,
            camera_id=i, prior_focal_length=True)

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in reconstruction.images.items()}


def import_features(image_ids, database_path, features_path):
    logger.info('Importing features into the database...')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path,
                   min_match_score=None, skip_geometric_verification=False):
    logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()


def estimation_and_geometric_verification(database_path, pairs_path,
                                          verbose=False):
    logger.info('Performing geometric verification of the matches...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(
                database_path, pairs_path,
                max_num_trials=20000, min_inlier_ratio=0.1)


def geometric_verification(image_ids, reference, database_path, features_path,
                           pairs_path, matches_path, max_error=4.0):
    logger.info('Performing geometric verification of the matches...')

    pairs = parse_retrieval(pairs_path)
    db = COLMAPDatabase.connect(database_path)

    inlier_ratios = []
    matched = set()
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        image0 = reference.images[id0]
        cam0 = reference.cameras[image0.camera_id]
        kps0, noise0 = get_keypoints(
            features_path, name0, return_uncertainty=True)
        kps0 = np.array([cam0.image_to_world(kp) for kp in kps0])

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            kps1, noise1 = get_keypoints(
                features_path, name1, return_uncertainty=True)
            kps1 = np.array([cam1.image_to_world(kp) for kp in kps1])

            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.add_two_view_geometry(id0, id1, matches)
                continue

            qvec_01, tvec_01 = pycolmap.relative_pose(
                image0.qvec, image0.tvec, image1.qvec, image1.tvec)
            _, errors0, errors1 = compute_epipolar_errors(
                qvec_01, tvec_01, kps0[matches[:, 0]], kps1[matches[:, 1]])
            valid_matches = np.logical_and(
                errors0 <= max_error * noise0 / cam0.mean_focal_length(),
                errors1 <= max_error * noise1 / cam1.mean_focal_length())
            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
            inlier_ratios.append(np.mean(valid_matches))
    logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                np.mean(inlier_ratios) * 100, np.median(inlier_ratios) * 100,
                np.min(inlier_ratios) * 100, np.max(inlier_ratios) * 100)

    db.commit()
    db.close()


def run_triangulation(model_path, database_path, image_dir, reference_model,
                      verbose=False):
    model_path.mkdir(parents=True, exist_ok=True)
    logger.info('Running 3D triangulation...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, image_dir, model_path)
    return reconstruction


def main(sfm_dir, reference_model, image_dir, pairs, features, matches,
         skip_geometric_verification=False, estimate_two_view_geometries=False,
         min_match_score=None, verbose=False):

    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    reference = pycolmap.Reconstruction(reference_model)

    image_ids = create_db_from_model(reference, database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            geometric_verification(
                image_ids, reference, database, features, pairs, matches)
    reconstruction = run_triangulation(sfm_dir, database, image_dir, reference,
                                       verbose)
    logger.info('Finished the triangulation with statistics:\n%s',
                reconstruction.summary())
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--reference_sfm_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(**args.__dict__)
