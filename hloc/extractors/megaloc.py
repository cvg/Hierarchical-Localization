"""
Code to use MegaLoc as a global features extractor.

MegaLoc paper: https://arxiv.org/abs/2502.17237
"""

from typing import Any

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class MegaPlaces(BaseModel):
    required_inputs: list[str] = ["image"]

    def _init(self, conf: dict[str, Any]) -> None:
        self.net = torch.hub.load("gmberton/MegaLoc", "get_trained_model").eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
