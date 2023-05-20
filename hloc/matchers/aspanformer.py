import torch
import warnings
from kornia.feature.loftr.loftr import default_cfg
from kornia.feature import LoFTR as LoFTR_

from ..utils.base_model import BaseModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as _ASpanFormer
from ASpanFormer.src.config.default import get_cfg_defaults
from ASpanFormer.src.utils.misc import lower_config
from ASpanFormer.demo import demo_utils 
aspanformer_path = Path(__file__).parent / '../../third_party/ASpanFormer'

class ASpanFormer(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'config_path': aspanformer_path / 'configs/aspan/outdoor/aspan_test.py',
    }
    required_inputs = [
        'image0',
        'image1'
    ]
    def _init(self, conf):
        config = get_cfg_defaults()
        config.merge_from_file(conf['config_path'])
        _config = lower_config(config)
        self.net = _ASpanFormer(config=_config['aspan'])
        weight_path =  aspanformer_path / 'weights/{}.ckpt'.format(conf['weights'])
        state_dict = torch.load(str(weight_path), map_location='cpu')['state_dict']
        self.net.load_state_dict(state_dict, strict=False)

    def _forward(self, data):
        data_ = {'image0': data['image0'],
                 'image1': data['image1'],}
        self.net(data_,online_resize=True)
        corr0, corr1 = data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
        pred = {}
        pred['keypoints0'], pred['keypoints1'] = corr0, corr1
        return pred
