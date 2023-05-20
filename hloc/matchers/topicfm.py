import torch
import warnings
from ..utils.base_model import BaseModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from TopicFM.src.models.topicfm import TopicFM as _TopicFM
from TopicFM.src.config.default import get_cfg_defaults
from TopicFM.src.utils.misc import lower_config
# from TopicFM.demo import demo_utils 
topicfm_path = Path(__file__).parent / '../../third_party/TopicFM'

class TopicFM(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'config_path': topicfm_path / 'configs/aspan/outdoor/aspan_test.py',
    }
    required_inputs = [
        'image0',
        'image1'
    ]
    # todo: refer to :third_party/TopicFM/viz/methods/topicfm.py
    def _init(self, conf):
        config = get_cfg_defaults()
        config.merge_from_file(conf['config_path'])
        _config = lower_config(config)
        self.net = _TopicFM(config=_config['aspan'])
        weight_path =  topicfm_path / 'pretrained/model_best.ckpt'
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
