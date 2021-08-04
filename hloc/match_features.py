import argparse
from typing import Union, Optional, Dict
import logging
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, parse_retrieval
from .utils.io import list_h5_names


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    }
}


def main(conf: Dict, pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None, matches: Optional[Path] = None,
         features_ref: Optional[Path] = None, exhaustive: bool = False):

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features+'.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]

    match_from_paths(
        conf, pairs, matches, features_q, features_ref, exhaustive)

    return matches


@torch.no_grad()
def match_from_paths(conf: Dict, pairs_path: Path, match_path: Path,
                     feature_path_q: Path, feature_paths_refs: Path,
                     exhaustive: bool = False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')
    name2ref = {n: i for i, p in enumerate(feature_paths_refs)
                for n in list_h5_names(p)}

    if not exhaustive:
        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    else:
        logging.info(f'Writing exhaustive match pairs to {pairs_path}.')
        assert not pairs_path.exists(), pairs_path
        names_q = list_h5_names(feature_path_q)
        # TODO: move exhqustive pair generation to a standalone script
        # detect self-matching
        if (len(feature_paths_refs) == 1
                and feature_path_q == feature_paths_refs[0]):
            pairs = [(n1, n2) for i, n1 in enumerate(names_q)
                     for n2 in names_q[:i]]
        else:
            pairs = [(n1, n2) for n1 in names_q for n2 in name2ref.keys()]
        with open(pairs_path, 'w') as f:
            f.write('\n'.join(' '.join((n1, n2)) for n1, n2 in pairs))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_path.parent.mkdir(exist_ok=True, parents=True)
    skip_pairs = set(list_h5_names(match_path) if match_path.exists() else ())

    for (name0, name1) in tqdm(pairs, smoothing=.1):
        pair = names_to_pair(name0, name1)
        # Avoid to recompute duplicates to save time
        if pair in skip_pairs or names_to_pair(name0, name1) in skip_pairs:
            continue

        data = {}
        with h5py.File(str(feature_path_q), 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(str(feature_paths_refs[name2ref[name1]]), 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        data = {k: v[None] for k, v in data.items()}

        pred = model(data)
        with h5py.File(str(match_path), 'a') as fd:
            grp = fd.create_group(pair)
            matches = pred['matches0'][0].cpu().short().numpy()
            grp.create_dataset('matches0', data=matches)

            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)

        skip_pairs.add(pair)

    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir,
         exhaustive=args.exhaustive)
