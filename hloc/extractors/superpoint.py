import lightglue.superpoint as lg_superpoint
import torch
from lightglue import SuperPoint as SuperPoint_

from ..utils.base_model import BaseModel


# The original keypoint sampling is incorrect. We patch it here but
# we don't fix it upstream to not impact existing evaluations.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(BaseModel):
    default_conf = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": False,
    }
    required_inputs = ["image"]
    detection_noise = 2.0

    def _init(self, conf):
        max_kp = conf["max_keypoints"]
        self.net = SuperPoint_(
            nms_radius=conf["nms_radius"],
            detection_threshold=conf["keypoint_threshold"],
            max_num_keypoints=None if max_kp is None or max_kp < 0 else max_kp,
            remove_borders=conf["remove_borders"],
        )

    def _forward(self, data):
        if self.conf["fix_sampling"]:
            orig_sample = lg_superpoint.sample_descriptors
            lg_superpoint.sample_descriptors = sample_descriptors_fix_sampling
            try:
                pred = self.net(data)
            finally:
                lg_superpoint.sample_descriptors = orig_sample
        else:
            pred = self.net(data)
        return {
            "keypoints": pred["keypoints"],
            "scores": pred["keypoint_scores"],
            "descriptors": pred["descriptors"].transpose(-1, -2),
        }
