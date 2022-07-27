import kornia
from kornia.feature.laf import (
    laf_from_center_scale_ori, raise_error_if_laf_is_not_valid,
    normalize_laf, denormalize_laf, get_laf_scale,
    generate_patch_grid_from_normalized_LAF, pyrdown)
import numpy as np
import torch
import torch.nn.functional as F
import pycolmap

from ..utils.base_model import BaseModel


EPS = 1e-6


def extract_patches_from_pyramid(
    img: torch.Tensor, laf: torch.Tensor, PS: int = 32,
    normalize_lafs_before_extraction: bool = True
) -> torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    Copied from kornia.feature.laf.extract_patches_from_pyramid with one minor
    difference - highlighted below.
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    _, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    pyr_idx = scale.log2().relu().long()  # diff: floor instead of round
    cur_img = img
    cur_pyr_level = 0
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        _, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_patch_grid_from_normalized_LAF(
                    cur_img[i: i + 1], nlaf[i: i + 1, scale_mask, :, :], PS)
            patches = F.grid_sample(
                cur_img[i: i + 1].expand(grid.size(0), ch, h, w),
                grid,  # type: ignore
                padding_mode="border",
                align_corners=False,
            )
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = pyrdown(cur_img)
        cur_pyr_level += 1
    return out


def sift_to_rootsift(x):
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class DoG(BaseModel):
    default_conf = {
        'options': {
            'first_octave': 0,
            'peak_threshold': 0.01,
        },
        'descriptor': 'rootsift',
        'max_keypoints': -1,
        'patch_size': 32,
        'mr_size': 12,
    }
    required_inputs = ['image']
    detection_noise = 1.0

    def _init(self, conf):
        if conf['descriptor'] == 'sosnet':
            self.describe = kornia.feature.SOSNet(pretrained=True)
        elif conf['descriptor'] not in ['sift', 'rootsift']:
            raise ValueError(f'Unknown descriptor: {conf["descriptor"]}')

        self.sift = None  # lazily instantiated on the first image
        self.device = torch.device('cpu')

    def to(self, *args, **kwargs):
        device = kwargs.get('device')
        if device is None:
            match = [a for a in args if isinstance(a, (torch.device, str))]
            if len(match) > 0:
                device = match[0]
        if device is not None:
            self.device = torch.device(device)
        return super().to(*args, **kwargs)

    def _forward(self, data):
        image = data['image']
        image_np = image.cpu().numpy()[0, 0]
        assert image.shape[1] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS

        if self.sift is None:
            use_gpu = pycolmap.has_cuda and self.device.type == 'cuda'
            options = {**self.conf['options']}
            if self.conf['descriptor'] == 'rootsift':
                options['normalization'] = pycolmap.Normalization.L1_ROOT
            else:
                options['normalization'] = pycolmap.Normalization.L2
            self.sift = pycolmap.Sift(
                options=pycolmap.SiftExtractionOptions(options),
                device=getattr(pycolmap.Device, 'cuda' if use_gpu else 'cpu'))

        keypoints, scores, descriptors = self.sift.extract(image_np)

        if self.conf['descriptor'] in ['sift', 'rootsift']:
            # We still renormalize because COLMAP does not normalize well,
            # maybe due to numerical errors
            if self.conf['descriptor'] == 'rootsift':
                descriptors = sift_to_rootsift(descriptors)
            descriptors = torch.from_numpy(descriptors)
        elif self.conf['descriptor'] == 'sosnet':
            center = keypoints[:, :2] + 0.5
            scale = keypoints[:, 2] * self.conf['mr_size'] / 2
            ori = -np.rad2deg(keypoints[:, 3])
            lafs = laf_from_center_scale_ori(
                torch.from_numpy(center)[None],
                torch.from_numpy(scale)[None, :, None, None],
                torch.from_numpy(ori)[None, :, None]).to(image.device)
            patches = extract_patches_from_pyramid(
                    image, lafs, PS=self.conf['patch_size'])[0]
            if len(keypoints) == 0:
                descriptors = torch.zeros((0, 128))
            else:
                descriptors = self.describe(patches).reshape(len(patches), 128)
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
