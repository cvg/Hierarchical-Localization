from typing import Any

import kornia
import numpy as np
import pycolmap
import torch
from kornia.feature.laf import extract_patches_from_pyramid, laf_from_center_scale_ori

from ..utils.base_model import BaseModel

EPS = 1e-6


def sift_to_rootsift(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + EPS)
    x = np.sqrt(x.clip(min=EPS))
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)
    return x


class DoG(BaseModel):
    default_conf: dict[str, Any] = {
        "options": {
            "first_octave": 0,
            "peak_threshold": 0.01,
        },
        "descriptor": "rootsift",
        "max_keypoints": -1,
        "patch_size": 32,
        "mr_size": 12,
    }
    required_inputs: list[str] = ["image"]
    detection_noise: float = 1.0
    max_batch_size: int = 1024

    def _init(self, conf: dict[str, Any]) -> None:
        self.describe: kornia.feature.SOSNet | kornia.feature.HardNet
        if conf["descriptor"] == "sosnet":
            self.describe = kornia.feature.SOSNet(pretrained=True)
        elif conf["descriptor"] == "hardnet":
            self.describe = kornia.feature.HardNet(pretrained=True)
        elif conf["descriptor"] not in ["sift", "rootsift"]:
            raise ValueError(f"Unknown descriptor: {conf['descriptor']}")

        self.sift: pycolmap.FeatureExtractor | None = None  # lazily instantiated
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def _forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        image = data["image"]
        image_np = image.cpu().numpy()[0, 0]
        assert image.shape[1] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS

        if self.sift is None:
            device = self.dummy_param.device
            use_gpu = pycolmap.has_cuda and device.type == "cuda"
            options = {**self.conf["options"]}
            if self.conf["descriptor"] == "rootsift":
                options["normalization"] = pycolmap.Normalization.L1_ROOT
            else:
                options["normalization"] = pycolmap.Normalization.L2
            self.sift = pycolmap.FeatureExtractor.create(
                options=pycolmap.FeatureExtractionOptions(
                    type=pycolmap.FeatureExtractorType.SIFT,
                    use_gpu=use_gpu,
                    sift=pycolmap.SiftExtractionOptions(**options),
                ),
                device=pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu,
            )

        kps, desc = self.sift.extract_from_float32_array(
            np.ascontiguousarray(image_np, dtype=np.float32)
        )
        if len(kps) > 0:
            keypoints_np = pycolmap.keypoints_to_matrix(kps)
            descriptors_np = desc.to_float().data / 512.0
        else:
            keypoints_np = np.zeros((0, 4), dtype=np.float32)
            descriptors_np = np.zeros((0, 128), dtype=np.float32)
        scales_np = keypoints_np[:, 2]
        oris_np = np.rad2deg(keypoints_np[:, 3])

        if self.conf["descriptor"] in ["sift", "rootsift"]:
            # We still renormalize because COLMAP does not normalize well,
            # maybe due to numerical errors
            if self.conf["descriptor"] == "rootsift":
                descriptors_np = sift_to_rootsift(descriptors_np)
            descriptors = torch.from_numpy(descriptors_np)
        elif self.conf["descriptor"] in ("sosnet", "hardnet"):
            center = keypoints_np[:, :2] + 0.5
            laf_scale = scales_np * self.conf["mr_size"] / 2
            laf_ori = -oris_np
            lafs = laf_from_center_scale_ori(
                torch.from_numpy(center)[None],
                torch.from_numpy(laf_scale)[None, :, None, None],
                torch.from_numpy(laf_ori)[None, :, None],
            ).to(image.device)
            patches = extract_patches_from_pyramid(
                image, lafs, PS=self.conf["patch_size"]
            )[0]
            descriptors = patches.new_zeros((len(patches), 128))
            if len(patches) > 0:
                for start_idx in range(0, len(patches), self.max_batch_size):
                    end_idx = min(len(patches), start_idx + self.max_batch_size)
                    descriptors[start_idx:end_idx] = self.describe(
                        patches[start_idx:end_idx]
                    )
        else:
            raise ValueError(f"Unknown descriptor: {self.conf['descriptor']}")

        keypoints = torch.from_numpy(keypoints_np[:, :2])  # keep only x, y
        scales = torch.from_numpy(scales_np)
        oris = torch.from_numpy(oris_np)
        scores = keypoints.new_zeros(len(keypoints))  # no scores for SIFT yet

        if self.conf["max_keypoints"] != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = torch.topk(scores, self.conf["max_keypoints"])
            keypoints = keypoints[indices]
            scales = scales[indices]
            oris = oris[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]

        return {
            "keypoints": keypoints[None],
            "scales": scales[None],
            "oris": oris[None],
            "scores": scores[None],
            "descriptors": descriptors.T[None],
        }
