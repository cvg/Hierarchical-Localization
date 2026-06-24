"""hloc matcher wrapper for glue-factory's original MambaGlue.

Place at:  Hierarchical-Localization/hloc/matchers/mambaglue.py

This is the "normal MambaGlue" baseline -- a different class from LateMambaGlue
(no Context/Interaction split, no n_cross_layers), with its own checkpoint
(e.g. outputs/training/sp+mg_megadepth/checkpoint_best.tar). I/O handling is the
same as the LateMambaGlue wrapper since both follow glue-factory's matcher
convention.

Requires glue-factory importable (`pip install -e .` in the glue-factory repo).
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..utils.base_model import BaseModel

# --- If glue-factory is NOT pip-installed: ---
# GLUEFACTORY_ROOT = Path("/home/ubuntu/work/MambaGlue/glue-factory")
# sys.path.append(str(GLUEFACTORY_ROOT))

from gluefactory.models.matchers.mambaglue import (  # noqa: E402
    MambaGlue as _GFMambaGlue,
)


class MambaGlue(BaseModel):
    default_conf = {
        "checkpoint": None,       # REQUIRED: path to glue-factory checkpoint_best.tar
        "filter_threshold": 0.1,
        "n_layers": None,         # None => use MambaGlue's own default architecture
        "features": "superpoint",
    }
    required_inputs = [
        "image0", "keypoints0", "scores0", "descriptors0",
        "image1", "keypoints1", "scores1", "descriptors1",
    ]

    def _init(self, conf):
        # keep the override set minimal: only keys certain to be in MambaGlue's
        # default_conf, so glue-factory's struct-checked merge can't reject them
        model_conf = {"filter_threshold": conf["filter_threshold"]}
        if conf.get("n_layers") is not None:
            model_conf["n_layers"] = conf["n_layers"]
        self.net = _GFMambaGlue(OmegaConf.create(model_conf))

        ckpt = conf["checkpoint"]
        assert ckpt is not None, "Set the 'checkpoint' field to your .tar file"
        state = torch.load(ckpt, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state

        # glue-factory saves the two_view_pipeline; matcher weights prefixed "matcher."
        matcher_sd = {
            k[len("matcher."):]: v for k, v in sd.items() if k.startswith("matcher.")
        }
        if not matcher_sd:
            matcher_sd = sd

        missing, unexpected = self.net.load_state_dict(matcher_sd, strict=False)
        if missing:
            print(f"[mambaglue] {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"[mambaglue] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        self.net.eval()

    def _forward(self, data):
        desc0 = data["descriptors0"].transpose(-1, -2).contiguous().float()
        desc1 = data["descriptors1"].transpose(-1, -2).contiguous().float()

        dev = data["keypoints0"].device
        h0, w0 = data["image0"].shape[-2:]
        h1, w1 = data["image1"].shape[-2:]
        size0 = torch.tensor([[w0, h0]], dtype=torch.float32, device=dev)
        size1 = torch.tensor([[w1, h1]], dtype=torch.float32, device=dev)

        gf_data = {
            "keypoints0": data["keypoints0"].float(),
            "keypoints1": data["keypoints1"].float(),
            "descriptors0": desc0,
            "descriptors1": desc1,
            "image_size0": size0,
            "image_size1": size1,
            "view0": {"image_size": size0},
            "view1": {"image_size": size1},
        }
        pred = self.net(gf_data)
        return {
            "matches0": pred["matches0"],
            "matching_scores0": pred["matching_scores0"],
        }