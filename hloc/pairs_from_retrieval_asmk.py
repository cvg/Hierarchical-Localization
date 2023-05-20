import argparse
from ast import arg
import imp
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

# import method to compute asmk
import sys
fire_path = Path(__file__).parent / '../third_party/fire'
sys.path.append(str(fire_path))
import pickle
from lib.asmk.asmk import io_helpers, asmk_method

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
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


def get_descriptors(names, path, name2idx=None, key='local_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r') as fd:
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


def pairs_from_rank_matrix(query_names,db_names,rank,topK = 25):
    pairs = []
    counter = 0
    for q in query_names:
        image_name = q
        # TODO: topK should < len(rank[counter,:]) 
        top_similarity_id = rank[counter,0:topK]

        for id in top_similarity_id:
            pair = (image_name, db_names[id])
            pairs.append(pair)
        counter += 1
    return pairs

def init_asmk(config_path,asmk_codebook_path):
    params = io_helpers.load_params(config_path)
    asmk_params = params['evaluation']['local_descriptor']['asmk']
    asmk_bin_path = asmk_codebook_path 
    asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)
    asmk = asmk.train_codebook(None, cache_path=asmk_bin_path)
    return asmk

def asmk_index_database(vecs, imids, asmk, logger, cache_path =None, distractors_path=None):
    """Asmk evaluation step 'aggregate_database' and 'build_ivf'"""
    asmk_dataset = asmk.build_ivf(vecs, imids, distractors_path=distractors_path, cache_path = cache_path)
    return asmk_dataset

def asmk_query_ivf(qvecs, qimids, logger, asmk_dataset, cache_path, imid_offset=0):
    """Asmk evaluation step 'query_ivf'"""
    qimids += imid_offset
    metadata, query_ids, ranks, scores = asmk_dataset.query_ivf(qvecs, qimids)
    logger.debug(f"Average query time (quant+aggr+search) is {metadata['query_avg_time']:.3f}s")
    # dumping to disk
    with cache_path.open("wb") as handle:
        pickle.dump({"metadata": metadata, "query_ids": query_ids, "ranks": ranks, "scores": scores}, handle)

    return ranks, query_ids, scores

def build_asmk_inputs():

    return 1

# make desc data as input to asmk
def convert_data(input_desc, names = None):
    vecs, imids = [], []
    size = input_desc.shape[0]
    for imid in range(size):
        vecs.append(input_desc[imid].cpu().numpy())               # 256 x 128
        imids.append(np.full((input_desc[imid].shape[0],), imid)) # 
    return np.vstack(vecs), np.hstack(imids)


def main(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None,
         topk = 25,
         asmk_config_path = None, 
         asmk_codebook_path = None,
         asmk_cache_db_path = None,
         asmk_cache_query_path = None):
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
    query_desc = get_descriptors(query_names, descriptors)

    logger.info(f'db_desc.shape  = {db_desc.shape} pairs.')
    logger.info(f'query_desc.shape = {query_desc.shape} pairs.')
    
    
    # TODO: asmk, load from file or train your self
    logger.info("initialize asmk begin ...")
    asmk = init_asmk(asmk_config_path, asmk_codebook_path)
    logger.info("initialize asmk done")

    # TODO: db_desc -> vecs, imids and query_desc -> qvecs, qimids
    logger.info("convert database data to asmk input ...")
    vecs, imids = convert_data(db_desc, names = db_names)
    logger.info("asmk index database ...")
    asmk_dataset = asmk_index_database(vecs, imids, asmk, logger, asmk_cache_db_path)
    logger.info("convert query data to asmk input ...")
    qvecs, qimids = convert_data(query_desc, names = query_names)
    logger.info("asmk index query ...")
    ranks, query_ids, scores = asmk_query_ivf(qvecs, qimids, logger, asmk_dataset, asmk_cache_query_path)
    
    # Avoid self-matching
    logger.info("make db-query pairs ...")
    pairs = pairs_from_rank_matrix(query_names,db_names,ranks,topK = topk)

    logger.info(f'Found {len(pairs)} pairs.')
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
    parser.add_argument('--db_descriptors', type=Path)

    parser.add_argument('--topk', type=int, required=False, default= 25)
    parser.add_argument('--asmk_config_path', type=Path)
    parser.add_argument('--asmk_codebook_path', type=Path)
    parser.add_argument('--asmk_cache_db_path', type=Path)
    parser.add_argument('--asmk_cache_query_path', type=Path)

    args = parser.parse_args()
    main(**args.__dict__)
