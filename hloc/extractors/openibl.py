from typing import Any

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class OpenIBL(BaseModel):
    default_conf: dict[str, Any] = {
        "model_name": "vgg16_netvlad",
    }
    required_inputs: list[str] = ["image"]

    def _init(self, conf: dict[str, Any]) -> None:
        self.net = torch.hub.load(
            "yxgeee/OpenIBL", conf["model_name"], pretrained=True
        ).eval()
        mean = [0.48501960784313836, 0.4579568627450961, 0.4076039215686255]
        std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
