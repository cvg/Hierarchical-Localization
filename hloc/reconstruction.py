import argparse
import logging
from pathlib import Path
import shutil
import multiprocessing
import subprocess
import pprint

from .utils.read_write_model import read_cameras_binary
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, geometric_verification)


def create_empty_db(database_path):
    logging.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(colmap_path, sfm_dir, image_dir, database_path,
                  single_camera=False):
    logging.info('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')

    # We need to create dummy features for COLMAP to import images with EXIF
    dummy_dir = sfm_dir / 'dummy_features'
    dummy_dir.mkdir()
    for i in images:
        with open(str(dummy_dir / (i.name + '.txt')), 'w') as f:
            f.write('0 128')

    cmd = [
        str(colmap_path), 'feature_importer',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--import_path', str(dummy_dir),
        '--ImageReader.single_camera',
        str(int(single_camera))]
    subprocess.run(cmd, check=True)

    db = COLMAPDatabase.connect(database_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.commit()
    db.close()
    shutil.rmtree(str(dummy_dir))


def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(colmap_path, sfm_dir, database_path, image_dir,
                       min_num_matches=None):
    logging.info('Running the 3D reconstruction...')
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(models_path),
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 16))]
    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    logging.info(' '.join(cmd))
    subprocess.run(cmd, check=True)

    models = list(models_path.iterdir())
    if len(models) == 0:
        logging.error('Could not reconstruct any model!')
        return None
    logging.info(f'Reconstructed {len(models)} models.')

    largest_model = None
    largest_model_num_images = 0
    for model in models:
        num_images = len(read_cameras_binary(str(model / 'cameras.bin')))
        if num_images > largest_model_num_images:
            largest_model = model
            largest_model_num_images = num_images
    assert largest_model_num_images > 0
    logging.info(f'Largest model is #{largest_model.name} '
                 f'with {largest_model_num_images} images.')

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer',
         '--path', str(largest_model)])
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

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        shutil.move(str(largest_model / filename), str(sfm_dir))

    return stats


def main(sfm_dir, image_dir, pairs, features, matches,
         colmap_path='colmap', single_camera=False,
         skip_geometric_verification=False,
         min_match_score=None, min_num_matches=None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(
        colmap_path, sfm_dir, image_dir, database, single_camera)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)
    stats = run_reconstruction(
        colmap_path, sfm_dir, database, image_dir, min_num_matches)
    if stats is not None:
        stats['num_input_images'] = len(image_ids)
        logging.info(f'Statistics:\n{pprint.pformat(stats)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--single_camera', action='store_true')
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--min_num_matches', type=int)
    args = parser.parse_args()

    main(**args.__dict__)
