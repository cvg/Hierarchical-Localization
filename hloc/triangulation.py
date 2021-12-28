import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import pycolmap

from .utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair


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
                logging.error('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()


def create_db_from_model(reconstruction, database_path):
    if database_path.exists():
        logging.warning('The database already exists, deleting it.')
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
    logging.info('Importing features into the database...')
    hfile = h5py.File(str(features_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = hfile[image_name]['keypoints'].__array__()
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    hfile.close()
    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path,
                   min_match_score=None, skip_geometric_verification=False):
    logging.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    hfile = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        pair = names_to_pair(name0, name1)
        if pair not in hfile:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in hfile}.')

        matches = hfile[pair]['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = hfile[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)
        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    hfile.close()
    db.commit()
    db.close()


def geometric_verification(database_path, pairs_path, verbose=False):
    logging.info('Performing geometric verification of the matches...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(
                database_path, pairs_path,
                max_num_trials=20000, min_inlier_ratio=0.1)


def run_triangulation(model_path, database_path, image_dir, reference_model,
                      verbose=False):
    model_path.mkdir(parents=True, exist_ok=True)
    logging.info('Running 3D triangulation...')
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, image_dir, model_path)
    return reconstruction


def main(sfm_dir, reference_model, image_dir, pairs, features, matches,
         skip_geometric_verification=False, min_match_score=None,
         verbose=False):

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
        geometric_verification(database, pairs, verbose)
    reconstruction = run_triangulation(sfm_dir, database, image_dir, reference,
                                       verbose)
    logging.info('Finished the triangulation with statistics:\n%s',
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
