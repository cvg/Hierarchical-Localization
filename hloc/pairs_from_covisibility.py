import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger


def main(model: Path, output: Path, num_matched: int) -> None:
    logger.info("Reading the COLMAP model...")
    reconstruction = pycolmap.Reconstruction(model)
    images = reconstruction.images
    points3D = reconstruction.points3D

    logger.info("Extracting image pairs from covisibility info...")
    track_image_ids = {
        pid: [el.image_id for el in point3D.track.elements]
        for pid, point3D in points3D.items()
    }
    pairs = []
    for image_id, image in tqdm(images.items()):
        covis: dict[int, int] = defaultdict(int)
        for p2D in image.get_observation_points2D():
            for image_covis_id in track_image_ids[p2D.point3D_id]:
                if image_covis_id != image_id:
                    covis[image_covis_id] += 1

        if len(covis) == 0:
            logger.info(f"Image {image_id} does not have any covisibility.")
            continue

        covis_ids = np.array(list(covis.keys()))
        covis_num = np.array([covis[i] for i in covis_ids])

        if len(covis_ids) <= num_matched:
            top_covis_ids = covis_ids[np.argsort(-covis_num)]
        else:
            # get covisible image ids with top k number of common matches
            ind_top = np.argpartition(covis_num, -num_matched)
            ind_top = ind_top[-num_matched:]  # unsorted top k
            ind_top = ind_top[np.argsort(-covis_num[ind_top])]
            top_covis_ids = covis_ids[ind_top]
            assert covis_num[ind_top[0]] == np.max(covis_num)

        for i in top_covis_ids:
            pair = (image.name, images[i].name)
            pairs.append(pair)

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_matched", required=True, type=int)
    args = parser.parse_args()
    main(**args.__dict__)
