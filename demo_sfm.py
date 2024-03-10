from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)

# define path
images = Path("datasets/South-Building/images/")

outputs = Path("outputs/sfm_south/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]

# find image pairs via image retrieval
# We extract global descriptors with NetVLAD and find for each image the most similar ones.

# netvlad process each image and output a 4096 dim global descriptor vector
retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

# extract and match local features
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

# run colmap on the features and matches
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

# model.write_text("./")  # text format
model.export_PLY(outputs / "scene.ply")  # PLY format

# visualize
visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)