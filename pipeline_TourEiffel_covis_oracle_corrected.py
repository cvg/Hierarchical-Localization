from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hloc.pipelines.Cambridge.utils import evaluate
from hloc.utils.read_write_model import read_model
from hloc import match_features, localize_sfm, logger


def pairs_from_covisibilty(model, output, num_matched, db_images, query_images):
    logger.info('Reading the COLMAP model...')
    cameras, images, points3D = read_model(model)

    image_name_to_id = {image.name:image_id for image_id, image in images.items()}
    db_ids = [image_name_to_id[db_name] for db_name in db_images]

    logger.info('Extracting image pairs from covisibility info...')
    pairs = []
    for image_id, image in tqdm(images.items()):
        if image.name in query_images:
            matched = image.point3D_ids != -1
            points3D_covis = image.point3D_ids[matched]

            covis = defaultdict(int)
            for point_id in points3D_covis:
                for image_covis_id in points3D[point_id].image_ids:
                    if image_covis_id != image_id:
                        covis[image_covis_id] += 1

            if len(covis) == 0:
                logger.info(f'Image {image_id} does not have any covisibility.')
                continue

            covis_ids = np.array([covis_id for covis_id in covis.keys() if covis_id in db_ids])
            covis_num = np.array([covis[i] for i in covis_ids])

            if len(covis_ids) <= num_matched:
                top_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                # get covisible image ids with top k number of common matches
                ind_top = np.argpartition(covis_num, -num_matched)
                ind_top = ind_top[-num_matched:]  # unsorted top k
                ind_top = ind_top[np.argsort(-covis_num[ind_top])]
                top_covis_ids = [covis_ids[i] for i in ind_top]
                assert covis_num[ind_top[0]] == np.max(covis_num)

            for i in top_covis_ids:
                pair = (image.name, images[i].name)
                pairs.append(pair)

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == '__main__':

    num_loc = 10

    suffix = '2015_2016_2018'

    full_sfm = Path('/home/data/TourEiffel/allyears')
    outputs = Path('outputs/TourEiffelCorrected') / suffix
    results = outputs / 'results_covis_oracle.txt'
    gt_dir = Path('datasets/toureiffel') / suffix
    ref_sfm = outputs / 'sfm_superpoint+superglue'
    query_list = outputs / 'query_list_with_intrinsics.txt'
    loc_pairs = outputs / f'pairs-query-covis{num_loc}.txt'
    features = outputs / 'feats-superpoint-n4096-r1024.h5'

    matcher_conf = match_features.confs['superglue']

    db_images = (gt_dir / 'list_db.txt').read_text().splitlines()
    query_images = (gt_dir / 'list_query.txt').read_text().splitlines()

    pairs_from_covisibilty(full_sfm, loc_pairs, num_matched=num_loc, db_images=db_images, query_images=query_images)

    loc_matches = match_features.main(matcher_conf, loc_pairs, 'feats-superpoint-n4096-r1024', outputs)

    localize_sfm.main(
        ref_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=False
    )

    evaluate(
        gt_dir / 'empty_all', results,
        gt_dir / 'list_query.txt', ext='.txt'
    )
