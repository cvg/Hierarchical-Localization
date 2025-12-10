import torch
from easy_local_features import getExtractor

from ..utils.base_model import BaseModel


class ElfFeatures(BaseModel):
    default_conf = {
        "model": {
            "name": "elf-detectors",
            "elf_detector": "xfeat",
            "elf_detector_conf": {
                "top_k": 512,
            },
            "elf_descriptor": "xfeat",
            "elf_descriptor_conf": {},
        },
    }

    def _init(self, conf):
        print(f"{conf=}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = getExtractor(conf["elf_detector"], conf["elf_detector_conf"])
        self.detector.to(self.device)

        self.descriptor = getExtractor(
            conf["elf_descriptor"], conf["elf_descriptor_conf"]
        )
        self.descriptor.to(self.device)

    def _forward(self, data):
        kps = self.detector.detect(data["image"])[0]

        desc = self.descriptor.compute(data["image"], kps).detach().squeeze(0).T
        return {
            "keypoints": [
                kps.cpu(),
            ],
            "descriptors": [
                desc.cpu(),
            ],
        }
