import torch
from loma.geometry import to_normalized
from loma.loma import LoMa, LoMaB, LoMaG, LoMaL, LoMaR, filter_matches

from ..utils.base_model import BaseModel


class LoMaMatcher(BaseModel):
    default_conf = {
        "filter_threshold": 0.1,
        "arch": "LoMa-B",
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        if conf["arch"] == "LoMa-B":
            cfg = LoMaB()
        elif conf["arch"] == "LoMa-L":
            cfg = LoMaL()
        elif conf["arch"] == "LoMa-G":
            cfg = LoMaG()
        elif conf["arch"] == "LoMa-R":
            cfg = LoMaR()
        else:
            raise ValueError(f"Unknown architecture {conf['arch']}")
        self.net = LoMa(cfg)

    def _forward(self, data):
        H_A, W_A = data["image0"].shape[2:]
        H_B, W_B = data["image1"].shape[2:]

        data["keypoints0"] = to_normalized(data["keypoints0"], H=H_A, W=W_A)
        data["keypoints1"] = to_normalized(data["keypoints1"], H=H_B, W=W_B)
        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)
        output = self.net(
            data["keypoints0"],
            data["keypoints1"],
            data["descriptors0"],
            data["descriptors1"],
        )
        scores = output["scores"]

        b = data["descriptors0"].shape[0]
        m0, m1, mscores0, mscores1 = filter_matches(
            scores, self.conf["filter_threshold"]
        )
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }
