from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization


dataset = Path('datasets/aachen/')  # change this if your dataset is somewhere else
images = dataset / 'images/images_upright/'

pairs = Path('pairs/aachen/')
sfm_pairs = pairs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
loc_pairs = pairs / 'pairs-query-netvlad50.txt'  # top 50 retrieved by NetVLAD

outputs = Path('outputs/aachen/')  # where everything will be saved
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad50.txt'  # the result file

print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

extract_features.main(feature_conf, images, outputs)

colmap_from_nvm.main(
    dataset / '3D-models/aachen_cvpr2018_db.nvm',
    dataset / '3D-models/database_intrinsics.txt',
    dataset / 'aachen.db',
    outputs / 'sfm_sift')

pairs_from_covisibility.main(
    outputs / 'sfm_sift', sfm_pairs, num_matched=20)

match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

colmap_from_nvm.main(
    dataset / '3D-models/aachen_cvpr2018_db.nvm',
    dataset / '3D-models/database_intrinsics.txt',
    dataset / 'aachen.db',
    outputs / 'sfm_empty',
    skip_points=True)

triangulation.main(
    reference_sfm,
    outputs / 'sfm_empty',
    images,
    sfm_pairs,
    outputs / f"{feature_conf['output']}.h5",
    outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{sfm_pairs.stem}.h5",
    colmap_path='colmap')  # change if COLMAP is not in your PATH

match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)

localize_sfm.main(
    reference_sfm / 'model',
    dataset / 'queries/*_time_queries_with_intrinsics.txt',
    loc_pairs,
    outputs / f"{feature_conf['output']}.h5",
    outputs / f"{feature_conf['output']}_{matcher_conf['output']}_{loc_pairs.stem}.h5",
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue

visualization.visualize_sfm_2d(reference_sfm / 'model', images, n=1, color_by='track_length')

visualization.visualize_sfm_2d(reference_sfm / 'model', images, n=1, color_by='visibility')

visualization.visualize_sfm_2d(reference_sfm / 'model', images, n=1, color_by='depth')

visualization.visualize_loc(
    results, images, reference_sfm / 'model', n=1, top_k_db=1, prefix='query/night', seed=2)
