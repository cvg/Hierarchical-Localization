from typing import Any

import torch
from lightglue import ALIKED as ALIKED_

from ..utils.base_model import BaseModel


class ALIKED(BaseModel):
    default_conf: dict[str, Any] = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }
    required_inputs: list[str] = ["image"]

    def _init(self, conf: dict[str, Any]) -> None:
        conf.pop("name")
        self.model = ALIKED_(**conf)

    def _forward(self, data: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
        features = self.model(data)

        return {
            "keypoints": [f for f in features["keypoints"]],
            "keypoint_scores": [f for f in features["keypoint_scores"]],
            "descriptors": [f.t() for f in features["descriptors"]],
        }
