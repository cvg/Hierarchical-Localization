import copy
from extract_patches.core import extract_patches
import kornia
import numpy as np
import torch
import torchvision.transforms as transforms
import pycolmap

from ..utils.base_model import BaseModel


EPS = 1e-6
PATCH_SIZE = 32
MR_SIZE = 12
BATCH_SIZE = 512


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class DoG(BaseModel):
    default_conf = {
        'num_octaves': 4,
        'octave_resolution': 3,
        'first_octave': 0,
        'edge_thresh': 10,
        'peak_thresh': 0.01,
        'upright': False,
        'descriptor': 'rootsift',
        'max_keypoints': -1
    }
    required_inputs = ['image']
    detection_noise = 1.0

    def _init(self, conf):
        self.descriptor = conf['descriptor']
        self.max_keypoints = conf['max_keypoints']
        assert self.descriptor in ['sift', 'rootsift', 'sosnet'], f'Unknown descriptor {self.descriptor}.'

        if self.descriptor == 'sosnet':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.describe = kornia.feature.SOSNet(pretrained=True).to(self.device)
            self.transform = transforms.ToTensor()

        vlfeat_conf = copy.deepcopy(conf)
        vlfeat_conf.pop('name', None)
        vlfeat_conf.pop('descriptor', None)
        vlfeat_conf.pop('device', None)
        vlfeat_conf.pop('max_keypoints', None)
        self.extract = lambda image: pycolmap.extract_sift(
            image, **vlfeat_conf
        )

    def _forward(self, data):
        image = data['image'].cpu().numpy()
        assert image.shape[1] == 1
        assert image.min() >= -EPS and image.max() <= 1 + EPS

        keypoints, scores, descriptors = self.extract(image[0, 0])

        if self.descriptor == 'rootsift':
            descriptors = sift_to_rootsift(descriptors)
        elif self.descriptor == 'sosnet':
            # VLFeat -> xyA conversion.
            # Based on https://github.com/colmap/colmap/blob/dev/src/feature/types.cc#L43-L53.
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            scale = keypoints[:, 2]
            ori = keypoints[:, 3]
            a11 = scale * np.cos(ori)
            a12 = -scale * np.sin(ori)
            a21 = scale * np.sin(ori)
            a22 = scale * np.cos(ori)
            keypoints = np.stack([x, y, a11, a12, a21, a22], axis=-1)
            # Extract patches.
            patches = extract_patches(keypoints, image[0, 0], PATCH_SIZE, MR_SIZE, 'xyA')
            # Extract descriptors.
            descriptors = np.zeros((len(patches), 128))
            for i in range(0, len(patches), BATCH_SIZE):
                data_a = patches[i : i + BATCH_SIZE]
                data_a = torch.stack(
                    [self.transform(patch) for patch in data_a]
                ).to(self.device)
                out_a = self.describe(data_a)
                descriptors[i : i + BATCH_SIZE] = out_a.cpu().detach().numpy()

        keypoints = keypoints[:, : 2]  # Keep only x, y.

        if self.max_keypoints != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = np.argsort(scores)[:: -1][: self.max_keypoints]
            keypoints = keypoints[indices, :]
            scores = scores[indices]
            descriptors = descriptors[indices, :]

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
