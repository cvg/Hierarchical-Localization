import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm

from hloc.utils.parsers import parse_image_lists_with_intrinsics
from hloc.utils.read_write_model import read_images_binary, read_images_text


def main(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None,
         per_camera=False, location_dir=None):
    logging.info('Extracting image pairs from a retrieval database.')
    hfile = h5py.File(str(descriptors), 'r')

    h5_names = []
    hfile.visititems(
        lambda _, obj: h5_names.append(obj.parent.name.strip('/'))
        if isinstance(obj, h5py.Dataset) else None)
    h5_names = list(set(h5_names))

    if db_prefix:
        if not isinstance(db_prefix, str):
            db_prefix = tuple(db_prefix)
        db_names = [n for n in h5_names if n.startswith(db_prefix)]
        assert len(db_names)
    elif db_list:
        db_names = [
            n for n, _ in parse_image_lists_with_intrinsics(db_list)]
    elif db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        raise ValueError('Provide either prefixes of DB names, or path to '
                         'lists of DB images, or path to a COLMAP model.')

    if query_prefix:
        if not isinstance(query_prefix, str):
            query_prefix = tuple(query_prefix)
        query_names = [n for n in h5_names if n.startswith(query_prefix)]
        assert len(query_names)
    elif query_list:
        query_names = [
            n for n, _ in parse_image_lists_with_intrinsics(query_list)]
    else:
        raise ValueError('Provide either prefixes of query names, or path to '
                         'lists of query images.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensor_from_names(names):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
        return desc

    mask = np.full((len(query_names), len(db_names)), True, np.bool)
    if per_camera:
        logging.info('Extracting camera information.')

        def get_cameras(names):
            cameras = []
            for name in names:
                if 'rear' in name:
                    n = 0
                elif 'left' in name:
                    n = 1
                elif 'right' in name:
                    n = 2
                else:
                    raise ValueError(name)
                cameras.append(n)
            return np.array(cameras)

        query_cameras = get_cameras(query_names)
        db_cameras = get_cameras(db_names)
        same_cameras = query_cameras[:, None] == db_cameras[None, :]
        mask &= same_cameras
    if location_dir is not None:
        logging.info('Extracting location information.')
        db_locations = {n: -1 for n in db_names}
        query_locations = {n: -1 for n in query_names}
        for i in tqdm(range(1, 50)):
            images = read_images_text(
                location_dir
                / f'colmap_reconstructions/{i:0>3}_aligned/images.txt')
            for im in images.values():
                name = im.name.replace('png', 'jpg')
                assert name in db_names, (name, db_names[:10])
                db_locations[name] = i
            qlist_path = (location_dir /
                          f'queries_per_location/queries_location_{i:0>3}.txt')
            with open(qlist_path, 'r') as f:
                for name in f.readlines():
                    name = name.rstrip()
                    assert name in query_names, (name, query_names[:10])
                    query_locations[name] = i
        db_locations = np.array([db_locations[n] for n in db_names])
        query_locations = np.array([query_locations[n] for n in query_names])
        assert np.all(query_locations != -1)
        assert np.all(db_locations != -1)
        same_locations = query_locations[:, None] == db_locations[None, :]
        mask &= same_locations
    mask = torch.from_numpy(mask).to(device)

    db_desc = tensor_from_names(db_names)
    query_desc = tensor_from_names(query_names)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    sim = torch.where(mask, sim, sim.new_tensor(float('-inf')))
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    pairs = []
    for query, indices in zip(query_names, topk):
        for i in indices:
            pair = (query, db_names[i])
            pairs.append(pair)

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--per_camera', action='store_true')
    parser.add_argument('--location_dir', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
