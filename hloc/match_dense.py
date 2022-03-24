from tqdm import tqdm
import numpy as np
import h5py
import torch
from pathlib import Path
from typing import Dict, Optional
import pprint
import argparse
import torchvision.transforms.functional as F
from omegaconf import OmegaConf

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_retrieval, names_to_pair
from .match_features import find_unique_new_pairs
from .extract_features import read_image, resize_image


confs = {
    'loftr': {
        'output': 'matches-loftr',
        'model': {
            'name': 'loftr',
            'weights': 'outdoor'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
            'dfactor': 8
        },
    },
}


class ImagePairDataset(torch.utils.data.Dataset):
    default_conf = {
        'grayscale': True,
        'resize_max': 1024,
        'dfactor': 8,
        'cache_images': False,
    }

    def __init__(self, image_dir, conf, pairs):
        self.image_dir = image_dir
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.pairs = pairs
        if self.conf.cache_images:
            image_names = set(sum(pairs, ()))  # unique image names in pairs
            logger.info(
                f'Loading and caching {len(image_names)} unique images.')
            self.images = {}
            self.scales={}
            for name in tqdm(image_names):
                image = read_image(self.image_dir / name, self.conf.grayscale)
                self.images[name], self.scales[name] = self.preprocess(image)

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x*scale)) for x in size)
                image = resize_image(image, size_new, 'cv2_area')
                scale = np.array(size) / np.array(size_new)

        if self.conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(map(
                lambda x: int(x // self.conf.dfactor * self.conf.dfactor),
                image.shape[-2:]))
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        if self.conf.cache_images:
            image0, scale0 = self.images[name0], self.scales[name0]
            image1, scale1 = self.images[name1], self.scales[name1]
        else:
            image0 = read_image(self.image_dir / name0, self.conf.grayscale)
            image1 = read_image(self.image_dir / name1, self.conf.grayscale)
            image0, scale0 = self.preprocess(image0)
            image1, scale1 = self.preprocess(image1)
        return image0, image1, scale0, scale1, names_to_pair(name0, name1)

def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts


@torch.no_grad()
def match_dense_from_paths(conf: Dict,
                           pairs_path: Path,
                           image_dir: Path,
                           match_path: Path,  # out
                           overwrite: bool = False) -> Path:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)

    if len(pairs) == 0:
        logger.info("All pairs exist. Skipping dense matching.")
        return

    dataset = ImagePairDataset(image_dir, conf["preprocessing"], pairs)

    loader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=1, shuffle=False)
    logger.info("Performing dense matching...")
    with h5py.File(str(match_path), 'a') as fd:
        for data in tqdm(loader):
            image0, image1, scale0, scale1, (pair,) = data
            scale0, scale1 = scale0[0].numpy(), scale1[0].numpy()
            data = {'image0': image0.to(device), 'image1': image1.to(device)}
            pred = model(data)
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            kpts0 += 0.5  # to COLMAP coordinates
            kpts1 += 0.5
            kpts0 = scale_keypoints(kpts0, scale0)
            kpts1 = scale_keypoints(kpts1, scale1)
            kpts0, kpts1 = kpts0.cpu().numpy() - 0.5, kpts1.cpu().numpy() - 0.5

            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)
            dset = grp.create_dataset('keypoints0', data=kpts0)
            dset.attrs['uncertainty'] = scale0.mean()
            dset = grp.create_dataset('keypoints1', data=kpts1)
            dset.attrs['uncertainty'] = scale1.mean()
            scores = pred['scores'].cpu().numpy()
            grp.create_dataset('scores', data=scores)


@torch.no_grad()
def main(conf: Dict,
         pairs: Path, image_dir: Path,
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         overwrite: bool = False) -> Path:

    logger.info('Matching dense features with configuration:'
                f'\n{pprint.pformat(conf)}')
    if matches is None:
        matches = Path(export_dir, conf['output']+'.h5')

    match_dense_from_paths(conf, pairs, image_dir, matches,
                           overwrite)

    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--matches', type=Path,
                        default=confs['loftr']['output'])
    parser.add_argument('--conf', type=str, default='loftr',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.image_dir, args.export_dir,
         args.matches)
