from typing import Any

import kornia
import torch

from ..utils.base_model import BaseModel


class DISK(BaseModel):
    default_conf: dict[str, Any] = {
        "weights": "depth",
        "max_keypoints": None,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }
    required_inputs: list[str] = ["image"]

    def _init(self, conf: dict[str, Any]) -> None:
        self.model = kornia.feature.DISK.from_pretrained(conf["weights"])

    def _forward(self, data: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
        image = data["image"]
        features = self.model(
            image,
            n=self.conf["max_keypoints"],
            window_size=self.conf["nms_window_size"],
            score_threshold=self.conf["detection_threshold"],
            pad_if_not_divisible=self.conf["pad_if_not_divisible"],
        )
        return {
            "keypoints": [f.keypoints for f in features],
            "keypoint_scores": [f.detection_scores for f in features],
            "descriptors": [f.descriptors.t() for f in features],
        }
