from extract_patches.core import extract_patches
import kornia
import numpy as np
import torch
import torchvision.transforms as transforms
import pycolmap

from ..utils.base_model import BaseModel


EPS = 1e-6


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class DoG(BaseModel):
    default_conf = {
        'vlfeat': {
            'num_octaves': 4,
            'octave_resolution': 3,
            'first_octave': 0,
            'edge_thresh': 10,
            'peak_thresh': 0.01,
            'upright': False,
        },
        'descriptor': 'rootsift',
        'max_keypoints': -1,
        'patch_size': 32,
        'mr_size': 12,
        'batch_size': 1024,
    }
    required_inputs = ['image']
    detection_noise = 1.0

    def _init(self, conf):
        if conf['descriptor'] == 'sosnet':
            self.describe = kornia.feature.SOSNet(pretrained=True)
            self.transform = transforms.ToTensor()
        elif conf['descriptor'] not in ['sift', 'rootsift']:
            raise ValueError(f'Unknown descriptor: {conf["descriptor"]}')

    def _forward(self, data):
        image = data['image'].cpu().numpy()
        assert image.shape[1] == 1
        assert image.min() >= -EPS and image.max() <= 1 + EPS
        image = image[0, 0]
        device = data['image'].device

        keypoints, scores, descriptors = pycolmap.extract_sift(
            image, **self.conf['vlfeat']
        )

        if self.conf['descriptor'] in ['sift', 'rootsift']:
            if self.conf['descriptor'] == 'rootsift':
                descriptors = sift_to_rootsift(descriptors)
            descriptors = torch.from_numpy(descriptors)
        elif self.conf['descriptor'] == 'sosnet':
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
            patches = extract_patches(
                keypoints, image, self.conf['patch_size'],
                self.conf['mr_size'], 'xyA')
            # Extract descriptors.
            batch_size = self.conf['batch_size']
            descriptors = torch.zeros((len(patches), 128), device=device)
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch = torch.stack([self.transform(p) for p in batch])
                descs = self.describe(batch.to(device))
                descriptors[i:i+batch_size] = descs
        else:
            raise ValueError(f'Unknown descriptor: {self.conf["descriptor"]}')

        keypoints = torch.from_numpy(keypoints[:, :2])  # keep only x, y
        scores = torch.from_numpy(scores)

        if self.conf['max_keypoints'] != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = torch.topk(scores, self.conf['max_keypoints'])
            keypoints = keypoints[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]

        return {
            'keypoints': keypoints[None],
            'scores': scores[None],
            'descriptors': descriptors.T[None],
        }
