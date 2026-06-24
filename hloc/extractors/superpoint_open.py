"""hloc extractor wrapper for glue-factory's *open* SuperPoint.

Place this file at:  Hierarchical-Localization/hloc/extractors/superpoint_open.py

Why: glue-factory trains on the rpautrat open SuperPoint
(gluefactory.models.extractors.superpoint_open), but hloc ships the original
MagicLeap SuperPoint by default. Both are 256-d, but the descriptor
distributions differ, which biases your Aachen numbers. This wrapper runs the
exact SuperPoint your LateMambaGlue checkpoint was trained against.

Requires glue-factory importable (`pip install -e .` in the glue-factory repo,
or uncomment the sys.path block). The open SuperPoint weights auto-download on
first instantiation, so the first run needs network access.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

from ..utils.base_model import BaseModel

# --- If glue-factory is NOT pip-installed: ---
# GLUEFACTORY_ROOT = Path("/abs/path/to/glue-factory")
# sys.path.append(str(GLUEFACTORY_ROOT))

from gluefactory.models.extractors.superpoint_open import (  # noqa: E402
    SuperPoint as _GFSuperPoint,
)


class SuperPointOpen(BaseModel):
    default_conf = {
        "nms_radius": 3,
        "max_num_keypoints": 4096,    # localization wants more kpts than the 512/1024 used in training
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "force_num_keypoints": False,  # variable kpt count per image for localization
    }
    required_inputs = ["image"]

    def _init(self, conf):
        gf_conf = OmegaConf.create({
            "nms_radius": conf["nms_radius"],
            "max_num_keypoints": conf["max_num_keypoints"],
            "detection_threshold": conf["detection_threshold"],
            "remove_borders": conf["remove_borders"],
            "descriptor_dim": conf["descriptor_dim"],
            "force_num_keypoints": conf["force_num_keypoints"],
        })
        self.net = _GFSuperPoint(gf_conf)
        self.net.eval()

    def _forward(self, data):
        # hloc passes a grayscale image [B, 1, H, W] in [0, 1]; glue-factory wants the same.
        pred = self.net({"image": data["image"]})
        # glue-factory out: keypoints [B,N,2] (x,y), keypoint_scores [B,N], descriptors [B,N,D]
        # hloc expects:     keypoints [B,N,2],       scores [B,N],          descriptors [B,D,N]
        return {
            "keypoints": pred["keypoints"],
            "scores": pred["keypoint_scores"],
            "descriptors": pred["descriptors"].transpose(-1, -2).contiguous(),
        }