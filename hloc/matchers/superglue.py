from ..utils.base_model import BaseModel
from .superglue_impl import SuperGlue as SG


class SuperGlue(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        self.net = SG(conf)

    def _forward(self, data):
        return self.net(data)
