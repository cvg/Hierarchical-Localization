"""hloc matcher wrapper for LateMambaGlue (trained in glue-factory).

Place this file at:  Hierarchical-Localization/hloc/matchers/latemambaglue.py

Requires glue-factory to be importable. Easiest: from the glue-factory repo
root run `pip install -e .` so `import gluefactory` works. Otherwise uncomment
the sys.path block below and point it at your glue-factory checkout.
"""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..utils.base_model import BaseModel

# --- If glue-factory is NOT pip-installed, expose it on sys.path: ---
#GLUEFACTORY_ROOT = Path("/home/ubuntu/work/MambaGlue/glue-factory")
#sys.path.append(str(GLUEFACTORY_ROOT))

from gluefactory.models.matchers.latemambaglue import (  # noqa: E402
    LateMambaGlue as _GFLateMambaGlue,
)


class LateMambaGlue(BaseModel):
    default_conf = {
        "checkpoint": None,      # REQUIRED: path to glue-factory checkpoint_best.tar
        "n_layers": 9,
        "n_cross_layers": 3,     # match the trained variant: nc2/nc3/nc4 -> 2/3/4
        "filter_threshold": 0.1,
        "flash": True,
        "checkpointed": False,   # gradient checkpointing OFF at inference
        "features": "superpoint",
    }
    # hloc feeds these keys to _forward (descriptors are [B, D, N] in hloc)
    required_inputs = [
        "image0", "keypoints0", "scores0", "descriptors0",
        "image1", "keypoints1", "scores1", "descriptors1",
    ]

    def _init(self, conf):
        model_conf = OmegaConf.create({
            "n_layers": conf["n_layers"],
            "n_cross_layers": conf["n_cross_layers"],
            "filter_threshold": conf["filter_threshold"],
            "flash": conf["flash"],
            "checkpointed": conf["checkpointed"],
        })
        self.net = _GFLateMambaGlue(model_conf)

        ckpt = conf["checkpoint"]
        assert ckpt is not None, "Set the 'checkpoint' field to your .tar file"
        state = torch.load(ckpt, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state

        # glue-factory saves the whole two_view_pipeline; matcher weights are
        # prefixed with "matcher." — strip it to get the bare matcher state dict.
        matcher_sd = {
            k[len("matcher."):]: v for k, v in sd.items() if k.startswith("matcher.")
        }
        if not matcher_sd:            # already a bare matcher checkpoint
            matcher_sd = sd

        missing, unexpected = self.net.load_state_dict(matcher_sd, strict=False)
        if missing:
            print(f"[latemambaglue] {len(missing)} missing keys, e.g. {missing[:3]}")
        if unexpected:
            print(f"[latemambaglue] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        self.net.eval()

    def _forward(self, data):
        # hloc descriptors: [B, D, N]  ->  glue-factory wants [B, N, D]
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
            # provide image size in both common conventions so keypoint
            # normalization is exact regardless of which one your forward reads
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