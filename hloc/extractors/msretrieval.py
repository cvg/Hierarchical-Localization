import kornia
import torch
import torch.nn.functional as F

from ..utils.base_model import BaseModel, dynamic_load
from .. import extractors


class MultiscaleRetrieval(BaseModel):
    default_conf = {
        'scales': [1.41, 1.0, 0.71]
    }

    def _init(self, conf):
        self.scales = conf['scales']
        assert 'backbone' in conf
        Backbone = dynamic_load(extractors, conf['backbone']['name'])
        self.backbone = Backbone(conf['backbone'])
    
    def _forward(self, data):
        input_image = data['image']

        global_descs = []
        for scale in self.scales:
            image = kornia.geometry.rescale(input_image, (scale, scale), antialias=True)
            global_descs.append(self.backbone({'image': image})['global_descriptor'])
        global_desc = torch.mean(torch.stack(global_descs, dim=0), dim=0)
        global_desc = F.normalize(global_desc, dim=1)

        return {
            'global_descriptor': global_desc
        }