import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SuperGluePretrainedNetwork.models.lightglue import LightGlue as LG


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LightGlueGim(BaseModel):
    # default_conf = {
    #     'depth_confidence': 0.95,
    #     'width_confidence': 0.99,
    # }
    required_inputs = [
        'image0', 'keypoints0', 'descriptors0',
        'image1', 'keypoints1', 'descriptors1',
    ]
    

    def _init(self, conf):
        checkpoints_path = Path(__file__).parent / '../../third_party/SuperGluePretrainedNetwork/models/weights/gim_lightglue_100h.ckpt'
        checkpoints_path = checkpoints_path.resolve()
        model = LG({
                'filter_threshold': 0.1,
                'flash': False,
                'checkpointed': True,
        })
        state_dict = torch.load(checkpoints_path, map_location=device)
        print(f"Loaded LightGlueGim checkpoint from: {checkpoints_path}")
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        self.net = model
        # self.net = LG(conf)

    def _forward(self, data):
        # print(f"LightGlueGim input keys: {data.keys()}")
        data['descriptors0'] = data['descriptors0'].transpose(-1, -2)
        data['descriptors1'] = data['descriptors1'].transpose(-1, -2)
        # check = {
        #     'image0': {k[:-1]: v for k, v in data.items() if k[-1] == '0'},
        #     'image1': {k[:-1]: v for k, v in data.items() if k[-1] == '1'}
        # }
        # print(f"check keys: {check.keys()}")
        # print(f"size of image0: {data['image_size0']}, size of image1: {data['image_size1']}")
        # for k,v in data.items():
        #     try:
        #         print(f"K :{k}; V shape: {v.shape}")
        #     except:
        #         print(f"K :{k}; V type: {type(v)}")
        return self.net(data)