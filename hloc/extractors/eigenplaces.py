"""
Code for loading models trained with EigenPlaces (or CosPlace) as a global
features extractor for geolocalization through image retrieval.
Multiple models are available with different backbones. Below is a summary of
models available (backbone : list of available output descriptors
dimensionality). For example you can use a model based on a ResNet50 with
descriptors dimensionality 1024.

EigenPlaces trained models:
    ResNet18:  [     256, 512]
    ResNet50:  [128, 256, 512, 2048]
    ResNet101: [128, 256, 512, 2048]
    VGG16:     [     512]

CosPlace trained models:
    ResNet18:  [32, 64, 128, 256, 512]
    ResNet50:  [32, 64, 128, 256, 512, 1024, 2048]
    ResNet101: [32, 64, 128, 256, 512, 1024, 2048]
    ResNet152: [32, 64, 128, 256, 512, 1024, 2048]
    VGG16:     [    64, 128, 256, 512]

EigenPlaces paper (ICCV 2023): https://arxiv.org/abs/2308.10832
CosPlace paper (CVPR 2022): https://arxiv.org/abs/2204.02287
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class EigenPlaces(BaseModel):
    default_conf = {
        "variant": "EigenPlaces",
        "backbone": "ResNet101",
        "fc_output_dim": 2048,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.net = torch.hub.load(
            "gmberton/" + conf["variant"],
            "get_trained_model",
            backbone=conf["backbone"],
            fc_output_dim=conf["fc_output_dim"],
        ).eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
