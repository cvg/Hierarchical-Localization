import torch
import cv2
import numpy as np

from ..utils.base_model import BaseModel


def tens_to_cv(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 2 and x.shape[0] in (8, 16, 32, 64, 128, 256):
        x = x.transpose(0, 1)
    if isinstance(x, torch.Tensor):
        x = x.contiguous().to(torch.uint8).numpy()
    else:
        x = np.ascontiguousarray(x, dtype=np.uint8)
    return x  # shape (N, 32)


class BinaryNearestNeighbor(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold_bits": None,
        "do_mutual_check": True,
    }

    required_inputs = ['descriptors0', 'scores0',
                       'descriptors1', 'scores1']

    def _init(self, conf):
        lut = torch.arange(256, dtype=torch.uint8)
        lut = (lut & 1) + ((lut >> 1) & 1) + ((lut >> 2) & 1) + ((lut >> 3) & 1) + \
              ((lut >> 4) & 1) + ((lut >> 5) & 1) + ((lut >> 6) & 1) + ((lut >> 7) & 1)
        self.register_buffer("_popcnt8", lut.to(torch.uint8), persistent=False)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _forward(self, data):
        d0 = data["descriptors0"]
        d1 = data["descriptors1"]

        d0 = tens_to_cv(d0)
        d1 = tens_to_cv(d1)

        D0, N0 = d0.shape
        _, N1 = d1.shape
        if N0 == 0 or N1 == 0:
            device = d0.device
            return {
                "matches0": torch.full((1, N0), -1, dtype=torch.long, device=device),
                "matching_scores0": torch.zeros((1, N0), dtype=torch.float32, 
                                                device=device),
            }

        matches = self.matcher.match(d0, d1)

        N0, Dbytes = d0.shape
        Dbits = 8 * Dbytes

        matches0 = torch.full((N0,), -1, dtype=torch.long)
        matching_scores0 = torch.zeros((N0,), dtype=torch.float32)

        for m in matches:
            q = m.queryIdx
            t = m.trainIdx
            dist = float(m.distance)
            matches0[q] = t
            matching_scores0[q] = 1.0 - dist / Dbits

        matches0 = matches0.unsqueeze(0)  # [1, N0]
        matching_scores0 = matching_scores0.unsqueeze(0)

        return {"matches0": matches0,
                "matching_scores0": matching_scores0}
