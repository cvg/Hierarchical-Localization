import warnings
from typing import Any

import torch
from kornia.feature import LoFTR as LoFTR_
from kornia.feature.loftr.loftr import default_cfg

from ..utils.base_model import BaseModel


class LoFTR(BaseModel):
    default_conf: dict[str, Any] = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "max_num_matches": None,
    }
    required_inputs: list[str] = ["image0", "image1"]

    def _init(self, conf: dict[str, Any]) -> None:
        cfg: dict[str, Any] = default_cfg
        cfg["match_coarse"]["thr"] = conf["match_threshold"]
        self.net = LoFTR_(pretrained=conf["weights"], config=cfg)

    def _forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            "keypoints0": "keypoints1",
            "keypoints1": "keypoints0",
            "image0": "image1",
            "image1": "image0",
            "mask0": "mask1",
            "mask1": "mask0",
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)

        scores = pred["confidence"]

        top_k = self.conf["max_num_matches"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred["keypoints0"], pred["keypoints1"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
            )
            scores = scores[keep]

        # Switch back indices
        pred = {rename.get(k, k): v for k, v in pred.items()}
        pred["scores"] = scores
        del pred["confidence"]
        return pred
