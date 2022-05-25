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
    results = outputs / 'results_covis_clustering.txt'
    gt_dir = Path('datasets/toureiffel') / suffix
    ref_sfm = outputs / 'sfm_superpoint+superglue'
    query_list = outputs / 'query_list_with_intrinsics.txt'
    loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'
    features = outputs / 'feats-superpoint-n4096-r1024.h5'
    loc_matches = outputs / 'feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad10.h5'

    localize_sfm.main(
        ref_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=True,
        prepend_camera_name=False
    )

    evaluate(
        gt_dir / 'empty_all', results,
        gt_dir / 'list_query.txt', ext='.txt'
    )
