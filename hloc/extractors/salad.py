"""Global image feature extractor Salad:
Optimal Transport Aggregation for Visual Place Recognition
Sergio Izquierdo, Javier Civera; CVPR 2024.
https://github.com/serizba/salad
"""
import math

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class Salad(BaseModel):
    default_conf = {
        "backbone": "dinov2_vitb14",
        "pretrained": True,
        "patch_size": 14,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load(
            "sarlinpe/salad",
            "dinov2_salad",
            backbone=conf["backbone"],
            pretrained=conf["pretrained"],
        ).eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        _, _, h, w = image.shape
        patch_size = self.conf["patch_size"]
        if h % patch_size or w % patch_size:
            h_inp = math.ceil(h / patch_size) * patch_size
            w_inp = math.ceil(w / patch_size) * patch_size
            image = torch.nn.functional.pad(image, [0, w_inp - w, 0, h_inp - h])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
