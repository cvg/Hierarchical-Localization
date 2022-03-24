import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import h5py
from typing import Optional, Dict, Union
import argparse

from . import logger
from .utils.parsers import names_to_pair, parse_retrieval
from .match_features import find_unique_new_pairs
from .match_dense import confs as match_confs

# Reimplementation of https://github.com/GrumpyZhou/image-matching-toolbox/blob/main/immatch/utils/localize_sfm_helper.py

confs = {
    'loftr': {
        'psize': 8,
        'max_dist': 2,
        'score_threshold': 0.2
    }
}


def get_grouped_ids(array):
    # Group array indices based on its values
    # all duplicates are grouped as a set
    idx_sort = np.argsort(array)
    sorted_array = array[idx_sort]
    _, ids, _ = np.unique(sorted_array, return_counts=True,
                          return_index=True)
    res = np.split(idx_sort, ids[1:])
    return res


def get_unique_matches_ids(match_ids, scores):
    if len(match_ids.shape) == 1:
        return [0]

    isets1 = get_grouped_ids(match_ids[:, 0])
    isets2 = get_grouped_ids(match_ids[:, 0])
    uid1s= [ids[scores[ids].argmax()] for ids in isets1]
    uid2s= [ids[scores[ids].argmax()] for ids in isets2]
    uids = list(set(uid1s).intersection(uid2s))
    return uids


class KeypointAggregator(object):
    default_conf = {
        'psize': 8,
        'max_dist': 2,
        'score_threshold': 0.2,
        'adjust_by_uncertainty': True
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def run(self,
            pairs_path: Path,
            dense_match_path: Path,
            feature_path: Path,
            match_path: Path,
            overwrite: Optional[bool] = False):
        def entries():
            return {'centers': [], 'kids': [], 'n_obs': []}
        all_kp_data = defaultdict(lambda: {'kps': [],
                                           'cells': defaultdict(entries)})
        logger.info('Start parsing dense matches and assigning keypoint ids ...')

        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
        with h5py.File(dense_match_path, 'r') as dmatch_file:
            with h5py.File(match_path, 'a') as match_file:
                for (name0, name1) in tqdm(pairs, smoothing=.1):
                    pair_name = names_to_pair(name0, name1)
                    kpts0_dset = dmatch_file[pair_name]['keypoints0']
                    kpts1_dset = dmatch_file[pair_name]['keypoints1']

                    scores = dmatch_file[pair_name]['scores'].__array__()
                    valid = np.where(scores >= self.conf.score_threshold)[0]
                    kpts0 = kpts0_dset.__array__()[valid]
                    kpts1 = kpts1_dset.__array__()[valid]

                    if self.conf.adjust_by_uncertainty:
                        uncertainty0 = kpts0_dset.attrs["uncertainty"]
                        uncertainty1 = kpts1_dset.attrs["uncertainty"]
                    else:
                        uncertainty0 = uncertainty1 = 1.0
                    # Compute match ids and quantize keypoints
                    matches, uids = self.get_keypoint_ids(
                        kpts0, kpts1, scores,
                        all_kp_data[name0], all_kp_data[name1],
                        uncertainty0=uncertainty0, uncertainty1=uncertainty1
                    )

                    # Convert matches to matches0
                    n_kps0 = np.max(matches[:, 0]) + 1
                    matches0 = -np.ones((n_kps0,))
                    scores0 = -np.ones((n_kps0,))
                    matches0[matches[:, 0]] = matches[:, 1]
                    scores0[matches[:, 0]] = scores[uids]

                    # Save matches
                    grp = match_file.create_group(pair_name)
                    grp.create_dataset('matches0', data=matches0)
                    grp.create_dataset('matching_scores0', data=scores0)
        # Save keypoints
        with h5py.File(feature_path, 'w') as fd:
            logger.info(f'Save keypoints from {len(all_kp_data)} images...')
            n_kps = 0
            for name in tqdm(all_kp_data, smoothing=.1):
                kps = np.array(all_kp_data[name]['kps'], dtype=np.float32)
                kgrp = fd.create_group(name)
                kgrp.create_dataset('keypoints', data=kps)
                n_kps += kps.shape[0]
        avg_kp_per_image = round(n_kps / len(all_kp_data), 1)
        logger.info(f'Finished assignment, found {avg_kp_per_image} '
                    f'keypoints/image (avg.), total {n_kps}.')

    def get_keypoint_ids(self, kpts0, kpts1, scores, kpd0, kpd1,
                         uncertainty0=1.0, uncertainty1=1.0):
        id1s = self.assign_keypoints(kpts0, kpd0, uncertainty=uncertainty0)
        id2s = self.assign_keypoints(kpts1, kpd1, uncertainty=uncertainty1)
        match_ids = np.dstack([id1s, id2s]).reshape(-1, 2)

        # Remove n-to-1 matches
        uids = get_unique_matches_ids(match_ids, scores)
        match_ids = match_ids[uids]
        return match_ids, uids

    def assign_keypoints(self, kpts, kpdata, uncertainty=1.0):
        kpt_ids = []
        cpts = (kpts / uncertainty) // self.conf.psize * self.conf.psize
        cpts = cpts.astype(np.int32)
        cells = kpdata['cells']
        for cpt, kpt in zip(cpts, kpts):
            cpt = tuple(cpt)
            cell = cells[cpt]
            centers = cell['centers']
            found_cluster = False
            if len(centers) > 0:
                dist = np.linalg.norm(kpt - np.array(centers), axis=1)
                cid = np.argmin(dist)
                if dist[cid] < self.conf.max_dist * uncertainty:
                    kid = cell['kids'][cid]
                    n_obs = cell["n_obs"]
                    centers[cid] = \
                        (centers[cid] * n_obs[cid] + kpt) / (n_obs[cid]+1)
                    kpdata['kps'][kid] = centers[cid]
                    n_obs[cid] += 1
                    found_cluster = True
            if not found_cluster:
                kid = len(kpdata['kps'])
                centers.append(kpt)
                kpdata['kps'].append(kpt)
                cell['kids'].append(kid)
                cell["n_obs"].append(1)
            kpt_ids.append(kid)
        return kpt_ids


def main(conf: Dict,
         pairs_path: Path,
         dense_matches: Union[Path, str],
         export_dir: Optional[Path] = None,
         features: Optional[Path] = None,
         matches: Optional[Path] = None,
         overwrite: bool = False):

    if isinstance(dense_matches, Path) or Path(dense_matches).exists():
        if matches is None or features is None:
            raise ValueError('Either provide all dense_matches, features and '
                              'matches as Path or all as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if dense_matches is not'
                             f' a file path: {dense_matches}.')
        if matches is None:
            matches = Path(
                export_dir, f'{dense_matches}_{pairs.stem}_sparse.h5')
        if features is None:
            features = Path(
                export_dir, f'{dense_matches}_feats.h5')
        dense_matches = Path(export_dir, dense_matches+'.h5')

    if not features.exists() and not matches.exists():
        aggregator = KeypointAggregator(conf)
        aggregator.run(pairs_path, dense_matches, features, matches,
                       overwrite)
    else:
        logger.info("Found existing local features and matches. Skipping.")

    return features, matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--dense_matches', type=str,
                        default=match_confs['loftr']['output'])
    parser.add_argument('--features', type=str)
    parser.add_argument('--matches', type=str)
    parser.add_argument('--conf', type=str, default='loftr',
                        choices=list(confs.keys()))
    args = parser.parse_args()

    main(confs[args.conf], args.pairs, args.dense_matches, args.export_dir,
         args.features, args.matches)
