import sys
from pathlib import Path

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint as SP


class SuperPoint(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.net = SP(conf)

    def _forward(self, data):
        return self.net(data)
