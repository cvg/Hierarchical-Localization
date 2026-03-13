import torch

from hloc import logger

from ..utils.base_model import BaseModel


class XFeat(BaseModel):
    default_conf = {
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(sparse) model done.")

    def _forward(self, data):
        pred = self.net.detectAndCompute(
            data["image"], top_k=self.conf["max_keypoints"]
        )[0]
        pred = {
            "keypoints": pred["keypoints"][None],
            "scores": pred["scores"][None],
            "descriptors": pred["descriptors"].T[None],
        }
        return pred
