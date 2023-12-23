import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections

from . import logger
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary
from .utils.io import list_h5_names


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
        if len(names) == 0:
            raise ValueError(
                f'Could not find any image with the prefix `{prefix}`.')
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f'Unknown type of image list: {names}.'
                             'Provide either a list or a path to a list file.')
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r', libver='latest') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r', libver='latest') as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def main(descriptors: Path, output: Path, num_matched: int,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None,
         chunk_size: int = -1):
    logger.info('Extracting image pairs from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    db_desc = db_desc.to(device)
    query_desc = get_descriptors(query_names, descriptors)

    num_pairs = 0
    chunk_size = chunk_size if chunk_size > 0 else len(query_names)
    with output.open('w') as f:
        query_name_splits = [query_names[i:i + chunk_size] for i in range(0, len(query_names), chunk_size)]
        query_splits = torch.split(query_desc, chunk_size)
        for i, (names, query_desc_split) in enumerate(zip(query_name_splits, query_splits)):
            if i != 0:
                f.write('\n')
            sim = torch.einsum('id,jd->ij', query_desc_split.to(device), db_desc)

            # Avoid self-matching
            self = np.array(names)[:, None] == np.array(db_names)[None]
            pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
            pairs = [(names[i], db_names[j]) for i, j in pairs]
            num_pairs += len(pairs)
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    logger.info(f'Found {num_pairs} pairs.')


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
    parser.add_argument('--db_descriptors', type=Path)
    parser.add_argument('--chunk_size', type=int, default=-1)
    args = parser.parse_args()
    main(**args.__dict__)
