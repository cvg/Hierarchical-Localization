# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import tensorflow as tf
from pathlib import Path
import torch
from ..utils.base_model import BaseModel

contextdesc_path = Path(__file__).parent / '../../third_party/contextdesc'
sys.path.append(str(contextdesc_path))
from datasets import get_dataset
from models import get_model

class ContextDesc(BaseModel):
    default_conf = {
        'stage': 'loc',
        'loc_model': contextdesc_path /'pretrained/contextdesc++',
        'model_type': 'pb',
        'dense_desc': False,
        'ratio_test': False,
        'n_sample': 2048,
        'n_feature': 10000,
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.loc_model_path = os.path.join(conf['loc_model'], 'loc.pb')

        self.local_model = get_model('loc_model')(self.loc_model_path, **{'sift_desc': True,
                                                  'n_sample': conf['n_sample'],
                                                  'peak_thld': 0.04,
                                                  'dense_desc': conf['dense_desc'],
                                                  'upright': False})


    def _forward(self, data):
        gray_img = data['image'].unsqueeze(-1).numpy()*255
        loc_feat, kpt_mb, normalized_xy, cv_kpts, sift_desc =\
                 self.local_model.run_test_data(gray_img[0][0])
        if sift_desc is None:
            return {}
        keypoints = np.array([[pt.pt[0], pt.pt[1]] for pt in cv_kpts])
        scores = np.array([pt.response for pt in cv_kpts])
        descriptors = loc_feat / np.linalg.norm(loc_feat, axis=-1, keepdims=True)
        return {
            'keypoints': [torch.from_numpy(keypoints)],
            'scores': [torch.from_numpy(scores)],
            'descriptors': [torch.from_numpy(descriptors.T)],
        }