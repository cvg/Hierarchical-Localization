import sys
from pathlib import Path
import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel

r2d2_path = Path(__file__).parent / "../../third_party/r2d2"
sys.path.append(str(r2d2_path))
from extract import load_network, NonMaxSuppression, extract_multiscale


class R2D2(BaseModel):
    default_conf = {
        'checkpoint_name': 'r2d2_WASF_N16.pt',
        'top-k': 5000,

        'scale-f': 2**0.25,
        'min-size': 256,
        'max-size': 1024,
        'min-scale': 0,
        'max-scale': 1,

        'reliability-thr': 0.7,
        'repetability-thr': 0.7,
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_fn = r2d2_path / "models" / conf['checkpoint_name']
        self.norm_rgb = tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.net = load_network(model_fn)
        self.detector = NonMaxSuppression(rel_thr=conf['reliability-thr'], 
            rep_thr=conf['repetability-thr'])
        
    def _forward(self, data):
        img = data['image']
        img = self.norm_rgb(img[0])[None]

        xys, desc, scores = extract_multiscale(self.net, img, self.detector,
                scale_f = self.conf['scale-f'],
                min_size = self.conf['min-size'],
                max_size = self.conf['max-size'],
                min_scale = self.conf['min-scale'],
                max_scale = self.conf['max-scale'],
        )
        idxs = scores.argsort()[-self.conf['top-k'] or None:]
        xy = xys[idxs, :2]
        desc = desc[idxs].t()
        scores = scores[idxs]

        pred = {'keypoints': xy[None], 'descriptors': desc[None], 'scores': scores[None]}
        return pred