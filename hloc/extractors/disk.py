import sys
from pathlib import Path
from functools import partial
import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel

disk_path = Path(__file__).parent / "../../third_party/disk"
sys.path.append(str(disk_path))
from disk import DISK as _DISK

class DISK(BaseModel):
    default_conf = {
        'model_name': 'depth-save.pth', # Name of the model's .pth save file
        # we rescale image only in the preprocessing step.
        # we only pad the image size to be the multiple of 16 in this model
        # 'height': None, # rescaled height (px). If unspecified, image is not resized in height dimension
        # 'width': None, # rescaled width (px). If unspecified, image is not resized in width dimension
        'window':  5, # NMS window size
        'n': None, # Maximum number of features to extract. If unspecified, the number is not limited
        'desc-dim': 128, # descriptor dimension. Needs to match the checkpoint value
        'mode': 'nms', # Whether to extract features using the non-maxima suppresion mode or through training-time grid sampling technique
    }
    required_inputs = ['image']

    def _init(self, conf):
        state_dict = torch.load(disk_path / conf['model_name'], map_location='cpu')

        if 'extractor' in state_dict:
            weights = state_dict['extractor']
        elif 'disk' in state_dict:
            weights = state_dict['disk']
        else:
            raise KeyError('Incompatible weight file!')
        self.model = _DISK(window=8, desc_dim=conf['desc-dim'])
        self.model.load_state_dict(weights)
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if conf['mode'] == 'nms':
            self.extract = partial(
                self.model.features,
                kind='nms',
                window_size=conf['window'],
                cutoff=0.,
                n=conf['n']
            )
        elif conf['mode'] == 'rng':
            self.extract = partial(self.model.features, kind='rng')
        else:
            raise KeyError('mode must be either nms or rng!')

    def _forward(self, data):
        img = data['image'][0]
        assert len(img.shape) == 3 and img.shape[0] == 3
        # pad img so that its height and width be the multiple of 16 as required by the original repo
        orig_h, orig_w = img.shape[1:]
        new_h = ((orig_h-1)//16 + 1) * 16
        new_w = ((orig_w-1)//16 + 1) * 16
        y_pad = new_h - orig_h
        x_pad = new_w - orig_w

        img = F.pad(img, (0, x_pad, 0, y_pad))
        assert img.shape[1] == new_h and img.shape[2] == new_w, "Wrong Padding!"

        with torch.no_grad():
            try:
                batched_features = self.extract(img[None])  # add batch dimension
            except RuntimeError as e:
                if 'U-Net failed' in str(e):
                    msg = ('Please use input size which is multiple of 16 (or '
                           'adjust the --height and --width flags to let this '
                           'script rescale it automatically). This is because '
                           'we internally use a U-Net with 4 downsampling '
                           'steps, each by a factor of 2, therefore 2^4=16.')
                    raise RuntimeError(msg) from e
                else:
                    raise
            
            assert(len(batched_features) == 1)
            features = batched_features[0]
            for features in batched_features.flat:
                features = features.to(torch.device('cpu'))

            kps_crop_space = features.kp.t()

            # we didn't rescale so no need for this
            # kps_img_space, mask = image.to_image_coord(kps_crop_space)
            kps_img_space = kps_crop_space # (2, N)
            x = kps_crop_space[0, :]
            y = kps_crop_space[1, :]
            mask = (0 <= x) & (x < orig_w) & (0 <= y) & (y < orig_h)
            

            keypoints   = kps_img_space.t()[mask]
            descriptors = features.desc[mask]
            scores      = features.kp_logp[mask]

            order = torch.argsort(-scores)

            keypoints   = keypoints[order]
            descriptors = descriptors[order]
            scores      = scores[order]

            assert descriptors.shape[1] == self.conf['desc-dim']
            assert keypoints.shape[1] == 2

            pred = {'keypoints': keypoints[None], 'descriptors':descriptors.t()[None], 'scores':scores[None]}
            return pred

