import sys
from pathlib import Path
from functools import partial
import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel

disk_path = Path(__file__).parent / "../../third_party/disk"
sys.path.append(str(disk_path))
from disk import DISK as _DISK  # noqa E402


class DISK(BaseModel):
    default_conf = {
        'model_name': 'depth-save.pth',
        'max_keypoints': None,
        'desc_dim': 128,
        'mode': 'nms',
        'nms_window_size': 5,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.model = _DISK(window=8, desc_dim=conf['desc_dim'])

        state_dict = torch.load(
            disk_path / conf['model_name'], map_location='cpu')
        if 'extractor' in state_dict:
            weights = state_dict['extractor']
        elif 'disk' in state_dict:
            weights = state_dict['disk']
        else:
            raise KeyError('Incompatible weight file!')
        self.model.load_state_dict(weights)

        if conf['mode'] == 'nms':
            self.extract = partial(
                self.model.features,
                kind='nms',
                window_size=conf['nms_window_size'],
                cutoff=0.,
                n=conf['max_keypoints']
            )
        elif conf['mode'] == 'rng':
            self.extract = partial(self.model.features, kind='rng')
        else:
            raise KeyError(
                f'mode must be `nms` or `rng`, got `{conf["mode"]}`')

    def _forward(self, data):
        image = data['image']
        # make sure that the dimensions of the image are multiple of 16
        orig_h, orig_w = image.shape[-2:]
        new_h = round(orig_h / 16) * 16
        new_w = round(orig_w / 16) * 16
        image = F.pad(image, (0, new_w - orig_w, 0, new_h - orig_h))

        batched_features = self.extract(image)

        assert(len(batched_features) == 1)
        features = batched_features[0]

        # filter points detected in the padded areas
        kpts = features.kp
        valid = torch.all(kpts <= kpts.new_tensor([orig_w, orig_h]) - 1, 1)
        kpts = kpts[valid]
        descriptors = features.desc[valid]
        scores = features.kp_logp[valid]

        # order the keypoints
        indices = torch.argsort(scores, descending=True)
        kpts = kpts[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

        return {
            'keypoints': kpts[None],
            'descriptors': descriptors.t()[None],
            'scores': scores[None],
        }
