import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from ALIKED.nets.aliked import ALIKED as Aliked

def split_images(image):
    _, _, H, W = image.shape
    img1 = to_pil_image(image[:, :, :H//2, :W//2].squeeze(0))
    img2 = to_pil_image(image[:, :, :H//2, W//2:].squeeze(0))
    img3 = to_pil_image(image[:, :, H//2:, :W//2].squeeze(0))
    img4 = to_pil_image(image[:, :, H//2:, W//2:].squeeze(0))
    img5 = to_pil_image(F.interpolate(
        image,
        size=(H//2, W//2),
        mode="bilinear",
        align_corners=False
    ).squeeze(0))
    return [img1, img2, img3, img4, img5]

class ALIKED(BaseModel):
    def _init(self, conf):
        self.net = Aliked(
            model_name=conf.get('model_name', 'aliked-n16'),
            device=conf.get('device', 'cuda'),
            top_k=conf.get('top_k', int(-1)),
            scores_th=conf.get('scores_th', float(0.2)),
            n_limit=conf.get('n_limit', int(5000)),
            load_pretrained=conf.get('pretrained', True)
        )

    def _forward(self, data):
        images = data["image"]
        img1, img2, img3, img4, img5 = split_images(images)
        result1 = self.net.run(img1)
        result2 = self.net.run(img2)
        result3 = self.net.run(img3)
        result4 = self.net.run(img4)
        results = self.net.run(img5)
        
        results['keypoints'][:,0] *= 2
        results['keypoints'][:,1] *= 2
        results.update({
            "keypoints_1": result1["keypoints"],
            "scores_1": result1["scores"],
            "descriptors_1": result1["descriptors"],
            "keypoints_2": result2["keypoints"],
            "scores_2": result2["scores"],
            "descriptors_2": result2["descriptors"],
            "keypoints_3": result3["keypoints"],
            "scores_3": result3["scores"],
            "descriptors_3": result3["descriptors"],
            "keypoints_4": result4["keypoints"],
            "scores_4": result4["scores"],
            "descriptors_4": result4["descriptors"]
        })
        return results
