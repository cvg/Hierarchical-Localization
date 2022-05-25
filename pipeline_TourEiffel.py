from pathlib import Path
import argparse

from hloc.pipelines.Cambridge.utils import create_query_list_with_intrinsics, evaluate
from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import triangulation, localize_sfm, pairs_from_retrieval, logger


if __name__ == '__main__':

    num_covis = 20
    num_loc = 10

    suffix = '2015_2016_2018'

    outputs = Path('outputs/TourEiffel') / suffix
    results = outputs / 'results.txt'
    gt_dir = Path('datasets/toureiffel') / suffix
    images = gt_dir / 'images'

    ref_sfm_sift = gt_dir / 'model_train'
    test_list = gt_dir / 'list_query.txt'

    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm = outputs / 'sfm_superpoint+superglue'
    ref_sfm_scaled = outputs / 'sfm_sift_scaled'
    query_list = outputs / 'query_list_with_intrinsics.txt'
    sfm_pairs = outputs / f'pairs-db-covis{num_covis}.txt'
    loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'

    feature_conf = {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    }
    matcher_conf = match_features.confs['superglue']
    retrieval_conf = extract_features.confs['netvlad']

    create_query_list_with_intrinsics(gt_dir / 'empty_all', query_list, test_list, ext='.txt')

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    with open(test_list, 'r') as f:
        query_images = {q for q in f.read().rstrip().split('\n')}
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_loc, db_model=ref_sfm_sift, query_list=query_images)

    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)

    sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    triangulation.main(
        ref_sfm, ref_sfm_sift,
        images,
        sfm_pairs,
        features,
        sfm_matches
    )

    loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)

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
