import argparse
import torch
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint
import platform
from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair


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
    'NN': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        },
    }
}


@torch.no_grad()
def main(conf, pairs, features, export_dir, exhaustive=False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    feature_path = Path(export_dir, features+'.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')

    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    elif exhaustive:
        logging.info(f'Writing exhaustive match pairs to {pairs}.')
        assert not pairs.exists(), pairs

        # get the list of images from the feature file
        images = []
        feature_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = list(set(images))

        pair_list = [' '.join((images[i], images[j]))
                     for i in range(len(images)) for j in range(i)]
        with open(str(pairs), 'w') as f:
            f.write('\n'.join(pair_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_name = f'{features}_{conf["output"]}_{pairs_name}'
    match_path = Path(export_dir, match_name+'.h5')
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        data = {}
        if platform.system() == 'Windows':
            name0, name1 = name0.replace('/', '\\'), name1.replace('/', '\\')

        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            data[k+'0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k+'1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

        pred = model(data)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    main(
        confs[args.conf], args.pairs, args.features, args.export_dir,
        exhaustive=args.exhaustive)
