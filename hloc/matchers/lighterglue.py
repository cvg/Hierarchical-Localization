import torch
from lightglue import LightGlue as LightGlue_

from ..utils.base_model import BaseModel


class LighterGlue(BaseModel):
    default_conf_xfeat = {
        "name": "lighterglue",  # just for interfacing
        "input_dim": 64,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": 0.95,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
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
        LightGlue_.default_conf = self.default_conf_xfeat
        self.net = LightGlue_(None, **conf)
        url = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt"  # noqa: E501
        state_dict = torch.hub.load_state_dict_from_url(url)

        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)

    def _forward(self, data):
        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)

        return self.net(
            {
                "image0": {k[:-1]: v for k, v in data.items() if k[-1] == "0"},
                "image1": {k[:-1]: v for k, v in data.items() if k[-1] == "1"},
            }
        )
