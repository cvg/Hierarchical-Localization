from typing import Any

import torch

from ..utils.base_model import BaseModel
from .superglue_impl import SuperGlue as SG


class SuperGlue(BaseModel):
    default_conf: dict[str, Any] = {
        "weights": "outdoor",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }
    required_inputs: list[str] = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf: dict[str, Any]) -> None:
        self.net = SG(conf)

    def _forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = self.net(data)
        return out
