from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
# from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import subprocess

images = Path('datasets/sacre_coeur')
outputs = Path('outputs/demo/')
subprocess.run(['rm', '-rf', outputs])
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
print(len(references), "mapping images")
# plot_images([read_image(images / r) for r in references[:4]], dpi=50)

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

# model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
# fig = viz_3d.init_figure()
# viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
# fig.show()