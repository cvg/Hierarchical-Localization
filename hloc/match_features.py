import argparse
from functools import partial
import os
from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch

from hloc.utils.tools import map_tensor
from hloc.utils.work_queue import WorkQueue

from . import matchers, logger
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
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
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


def main(conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False,
         num_workers=os.cpu_count()) -> Path:

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

    match_from_paths(conf, pairs, matches, features_q, features_ref, 
      overwrite, num_workers=num_workers)

    return matches

class FeaturesPairs():
  def __init__(self, 
    pairs : List[Tuple[str, str]],
    feature_path_q: Path,
    feature_paths_refs: Path):

      if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
      for path in feature_paths_refs:
          if not path.exists():
              raise FileNotFoundError(f'Reference feature file {path}.')

      self.name2ref = {n: i for i, p in enumerate(feature_paths_refs)
                  for n in list_h5_names(p)}


      self.feature_path_q = feature_path_q
      self.feature_paths_refs = feature_paths_refs
      self.pairs = pairs

      self.fd_q = h5py.File(str(self.feature_path_q), 'r')
      self.fd_refs = [h5py.File(ref, 'r') for ref in self.feature_paths_refs]

  def close(self):
    self.fd_q.close()
    for fd in self.fd_refs:
      fd.close()

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
      name0, name1 = self.pairs[idx]

      data = {}
      grp = self.fd_q[name0]
      for k, v in grp.items():
          data[k+'0'] = torch.from_numpy(v.__array__()).float()
      # some matchers might expect an image but only use its size
      data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])

      fd = self.fd_refs[self.name2ref[name1]]
      grp = fd[name1]
      for k, v in grp.items():
          data[k+'1'] = torch.from_numpy(v.__array__()).float()
      data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])

      return (names_to_pair(name0, name1), data)



def write_matches(item, fd):
  pairs, preds = item
  matches = preds['matches0'].cpu().short().numpy()

  scores = None
  if 'matching_scores0' in preds:
    scores = preds['matching_scores0'].cpu().half().numpy()

  for pair, match, score in zip(pairs, matches, scores):
    if pair in fd:
        del fd[pair]
    grp = fd.create_group(pair)
    grp.create_dataset('matches0', data=match)

    if scores is not None:
      grp.create_dataset('matching_scores0', data=score)



@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_paths_refs: Path,
                     overwrite: bool = False,
                     device=None,
                     num_workers=1,
                     autocast=True) -> Path:

    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]

    if device is None:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device=device)

    match_path.parent.mkdir(exist_ok=True, parents=True)
    skip_pairs = set(list_h5_names(match_path)
                     if match_path.exists() and not overwrite else ())

    pairs = [pair for pair in set(pairs) 
      if names_to_pair(*pair) not in skip_pairs]

    feature_pairs = FeaturesPairs(pairs, feature_path_q, feature_paths_refs)
    loader = torch.utils.data.DataLoader(feature_pairs, num_workers=num_workers, batch_size=1, shuffle=False, pin_memory=True)


    with h5py.File(str(match_path), 'a') as fd:

      writer = WorkQueue(
        process_item = partial(write_matches, fd=fd),
        num_threads=num_workers)

      with tqdm(smoothing=.1, total=len(feature_pairs)) as pbar:
        for pairs, data in loader:


            with torch.inference_mode():
              with torch.cuda.amp.autocast(autocast):

                data = map_tensor(data, partial(torch.Tensor.to, device=device))
                preds = model(data)

            writer.put( (pairs, preds) )

            pbar.update(1)
    
        writer.join()
        feature_pairs.close()

    logger.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--matches', type=Path)

    args = parser.parse_args()

    main(confs[args.conf], args.pairs, args.features, args.export_dir)
