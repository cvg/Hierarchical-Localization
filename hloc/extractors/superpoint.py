import sys
from pathlib import Path
import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SuperGluePretrainedNetwork.models import superpoint  # noqa E402


# torch>=1.10 breaks the original version check.
# We monkeypatch the function until this PR is merged:
# https://github.com/magicleap/SuperGluePretrainedNetwork/pull/104
def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


superpoint.sample_descriptors = sample_descriptors


class SuperPoint(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = superpoint.SuperPoint(conf)

    def _forward(self, data):
        return self.net(data)
