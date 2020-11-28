import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import subprocess
import pprint
import platform

from .utils.read_write_model import (
        read_cameras_binary, read_images_binary, CAMERA_MODEL_NAMES)
from .utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair


def create_db_from_model(empty_model, database_path):
    if database_path.exists():
        logging.warning('Database already exists.')

    cameras = read_cameras_binary(str(empty_model / 'cameras.bin'))
    images = read_images_binary(str(empty_model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(
            model_id, camera.width, camera.height, camera.params, camera_id=i,
            prior_focal_length=True)

    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, features_path):
    logging.info('Importing features into the database...')
    hfile = h5py.File(str(features_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        if platform.system() == 'Windows':
            image_name = image_name.replace('/', '\\')
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
        pairs = [p.split(' ') for p in f.read().split('\n')]

    hfile = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        pair0 = names_to_pair(name0, name1)
        pair1 = names_to_pair(name0, name1)
        if pair0 in hfile:
            pair = pair0
        elif pair1 in hfile:
            pair = pair1
        else:
            raise ValueError(f'Could not find pair {(name0, name1)}')

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


def geometric_verification(colmap_path, database_path, pairs_path):
    logging.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs']
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with matches_importer, exiting.')
        exit(ret)


def run_triangulation(colmap_path, model_path, database_path, image_dir,
                      empty_model):
    logging.info('Running the triangulation...')
    assert model_path.exists()

    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(empty_model),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    logging.info(' '.join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with point_triangulator, exiting.')
        exit(ret)

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', str(model_path)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    return stats


def main(sfm_dir, empty_sfm_model, image_dir, pairs, features, matches,
         colmap_path='colmap', skip_geometric_verification=False,
         min_match_score=None):

    assert empty_sfm_model.exists(), empty_sfm_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    model = sfm_dir / 'model'
    model.mkdir(exist_ok=True)

    image_ids = create_db_from_model(empty_sfm_model, database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)
    stats = run_triangulation(
        colmap_path, model, database, image_dir, empty_sfm_model)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--empty_sfm_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    args = parser.parse_args()

    main(**args.__dict__)
