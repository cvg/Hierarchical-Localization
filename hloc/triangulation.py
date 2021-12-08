import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import subprocess
import pprint
import shutil

from .utils.read_write_model import (
        read_cameras_binary, read_images_binary, CAMERA_MODEL_NAMES,
        write_points3D_binary, write_images_binary)
from .utils.database import COLMAPDatabase
from .utils.parsers import names_to_pair


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        message = "Command '%s' returned non-zero exit status %d." % (
                ' '.join(self.cmd), self.returncode)
        if self.output is not None:
            message += ' Last outputs:\n%s' % (
                '\n'.join(self.output.decode('utf-8').split('\n')[-10:]))
        return message


# TODO: consider creating a Colmap object that holds the path and verbose flag
def run_command(cmd, verbose=False):
    stdout = None if verbose else subprocess.PIPE
    ret = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=stdout)
    if not ret.returncode == 0:
        raise CalledProcessError(
                returncode=ret.returncode, cmd=cmd, output=ret.stdout)


def create_empty_model(reference_model, empty_model):
    logging.info('Creating an empty model.')
    empty_model.mkdir(exist_ok=True)
    shutil.copyfile(reference_model/'cameras.bin', empty_model/'cameras.bin')
    write_points3D_binary(dict(), empty_model / 'points3D.bin')
    images = read_images_binary(str(reference_model / 'images.bin'))
    images_empty = dict()
    for id_, image in images.items():
        images_empty[id_] = image._replace(
            xys=np.zeros((0, 2), float), point3D_ids=np.full(0, -1, int))
    write_images_binary(images_empty, empty_model / 'images.bin')


def create_db_from_model(empty_model, database_path):
    if database_path.exists():
        logging.warning('The database already exists, deleting it.')
        database_path.unlink()

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


def geometric_verification(colmap_path, database_path, pairs_path, verbose):
    logging.info('Performing geometric verification of the matches...')
    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs',
        '--SiftMatching.use_gpu', '0',
        '--SiftMatching.max_num_trials', str(20000),
        '--SiftMatching.min_inlier_ratio', str(0.1)]
    run_command(cmd, verbose)


def run_triangulation(colmap_path, model_path, database_path, image_dir,
                      empty_model, verbose):
    model_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(empty_model),
        '--output_path', str(model_path),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0']
    logging.info('Running the triangulation with command:\n%s', ' '.join(cmd))
    run_command(cmd, verbose)

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


def main(sfm_dir, reference_sfm_model, image_dir, pairs, features, matches,
         colmap_path='colmap', skip_geometric_verification=False,
         min_match_score=None, verbose=False):

    assert reference_sfm_model.exists(), reference_sfm_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    empty_model = sfm_dir / 'empty'

    create_empty_model(reference_sfm_model, empty_model)
    image_ids = create_db_from_model(empty_model, database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs, verbose)
    stats = run_triangulation(
        colmap_path, sfm_dir, database, image_dir, empty_model, verbose)

    logging.info('Finished the triangulation with statistics:\n%s',
                 pprint.pformat(stats))
    shutil.rmtree(empty_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--reference_sfm_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(**args.__dict__)
