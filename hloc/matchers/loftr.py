import torch
import warnings
from kornia.feature.loftr.loftr import default_cfg
from kornia.feature import LoFTR as LoFTR_

from ..utils.base_model import BaseModel


class LoFTR(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'max_num_matches': None,
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        cfg = default_cfg
        cfg['match_coarse']['thr'] = conf['match_threshold']
        self.net = LoFTR_(pretrained=conf['weights'], config=cfg)

    def _forward(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data)

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] =\
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        pred['scores'] = scores
        del pred['confidence']
        return pred
