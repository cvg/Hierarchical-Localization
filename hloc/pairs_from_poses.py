import argparse
import logging
from pathlib import Path
import numpy as np
import scipy.spatial

from .utils.read_write_model import read_images_binary

DEFAULT_ROT_THRESH = 30  # in degrees


def get_pairwise_distances(images):
    ids = np.array(list(images.keys()))
    Rs = []
    ts = []
    for id_ in ids:
        image = images[id_]
        R = image.qvec2rotmat()
        t = image.tvec
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ts))
    trace = np.einsum('nji,mji->mn', Rs, Rs, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))
    return ids, dist, dR


def main(model, output, num_matched, rotation_threshold=DEFAULT_ROT_THRESH):
    logging.info('Reading the COLMAP model...')
    images = read_images_binary(model / 'images.bin')

    logging.info(
        f'Obtaining pairwise distances between {len(images)} images...')
    ids, dist, dR = get_pairwise_distances(images)

    valid = (dR < rotation_threshold)
    np.fill_diagonal(valid, False)
    dist = np.where(valid, dist, np.inf)

    logging.info('Extracting pairs...')
    k = num_matched
    pairs = []
    for i, id_ in enumerate(ids):
        dist_i = dist[i]
        idx = np.argpartition(dist_i, k)[:k]  # not sorted
        idx = idx[np.argsort(dist_i[idx])]  # sorted
        idx = idx[valid[i][idx]]

        for j in idx:
            name0 = images[id_].name
            name1 = images[ids[j]].name
            pairs.append((name0, name1))

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--num_matched', required=True, type=int)
    parser.add_argument('--rotation_threshold',
                        default=DEFAULT_ROT_THRESH, type=float)
    args = parser.parse_args()
    main(**args.__dict__)
