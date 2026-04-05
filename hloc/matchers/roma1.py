from ..utils.base_model import BaseModel
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from scipy.spatial import cKDTree

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from RoMa.romatch.utils.kde import kde
from RoMa.romatch.models.model_zoo.roma_models import roma_model, tiny_roma_v1_model

weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # hopefully this doesnt change :D
}


def tiny_roma_v1_outdoor(device, weights=None, xfeat=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"], map_location=device
        )
    if xfeat is None:
        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )

    return tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(device)

class RoMa_Matches(BaseModel):
    def _init(self, conf):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        if conf['weight_mode'] == "indoor":
            self.weights = torch.hub.load_state_dict_from_url(
                weight_urls["romatch"]["indoor"], map_location=self.device
            )
            self.dinov2_weights = torch.hub.load_state_dict_from_url(
                weight_urls["dinov2"], map_location=self.device
            )
        elif conf['weight_mode'] == "outdoor":
            self.weights = torch.hub.load_state_dict_from_url(
                weight_urls["romatch"]["outdoor"], map_location=self.device
            )
            self.dinov2_weights = torch.hub.load_state_dict_from_url(
                weight_urls["dinov2"], map_location=self.device
            )
        
        size_new = (conf['resize_max']//14)*14
        self.amp_dtype: torch.dtype = torch.float16
        self.symmetric = True
        self.use_custom_corr = False
        self.upsample_preds = False
        self.coarse_res: Union[int, tuple[int, int]] = (size_new, size_new)
        self.upsample_res: Union[int, tuple[int, int]] = 864
        self.max_keypoints = conf['max_keypoints']
        # self.sample_mode = "threshold_balanced"
        # self.sample_thresh = 100
        try:
            self.dist_threshold = float(conf['dist_threshold'])
        except:
            self.dist_threshold = 2
        self.net = roma_model(
            resolution=self.coarse_res,
            upsample_preds=self.upsample_preds,
            weights=self.weights,
            dinov2_weights=self.dinov2_weights,
            device=self.device,
            amp_dtype=self.amp_dtype,
            symmetric=self.symmetric,
            use_custom_corr=self.use_custom_corr,
            upsample_res=self.upsample_res
        )
        
    def _forward(self, data):
        img0, img1 = data[0], data[2]
        sp0, sp1 = data[1], data[3]
        patch_number = data[4]
        # print(f"shape im0 {img0.shape} shape im1 {img1.shape}")
        W0, H0 = img0.size #
        W1, H1 = img1.size #
        # Viết hàm match_from_feature dựa vào match của RoMa để xử lý với đàu vào là feature.
        matches, mconf = self.net.match(
            im_A_input=img0,
            im_B_input=img1
        )

        # Tìm ra các điểm có mconf cao với số lượng keypoints đã config
        max_keypoints_process = self.max_keypoints
        if patch_number != 0:
            max_keypoints_process = max_keypoints_process//4
        # else:
        #     max_keypoints_process = max_keypoints_process//4
        matches, mconf = self.net.sample(
            matches,
            mconf,
            num=max_keypoints_process,
        )
        
        # Chuyển và tách ra thành keypoint ở ảnh 1 và ảnh 2 ứng với kích thước gốc của 2 ảnh đó.
        rm0, rm1 = self.to_pixel_coordinates(
            matches, H0, W0, H1, W1
        ) 
        
        # Do đây là 2 keypoint đã được matches với nhau bằng RoMa nên matches0 là range từ [0, 1, 2, 3, ...] phù hợp với đầu ra của hloc.

        # sp0 = sp0[0] # original coordinates, not resized coordinates
        # sp1 = sp1[0] # original coordinates, not resized coordinates
        # print(f"shape of rm0: {rm0.shape}, shape of rm1: {rm1.shape}, shape of sp0: {sp0.shape}, shape of sp1: {sp1.shape}")
        mask = (mconf >= 0.9)
        # rm0 = rm0[mask].cpu().numpy() # original coordinates, not resized coordinates
        # rm1 = rm1[mask].cpu().numpy() # original coordinates, not resized coordinates

        rm0 = rm0[mask] # original coordinates, not resized coordinates
        rm1 = rm1[mask] # original coordinates, not resized coordinates

        # print(f'rm0: {rm0.shape}, rm1: {rm1.shape}, sp0: {sp0.shape}, sp1: {sp1.shape}')
        # rm0: (4001, 2), rm1: (4001, 2), sp0: (630, 2), sp1: (660, 2)
        # print(f"shape of sp0: {sp0.shape}, shape of sp1: {sp1.shape}")
        # tree0 = cKDTree(sp0)
        # tree1 = cKDTree(sp1)

        print(f"rm0shape {rm0.shape}, sp0 shape{sp0.shape}")
        print(f"rm1shape {rm1.shape}, sp1 shape{sp1.shape}")
        dist_rm0_sp0 = torch.cdist(rm0, torch.tensor(sp0).to(self.device).float(), p=2)
        dist_rm1_sp1 = torch.cdist(rm1, torch.tensor(sp1).to(self.device).float(), p=2)
        print(f"dist_rm0_sp0shape {dist_rm0_sp0.shape}")
        print(f"dist_rm1_sp1shape {dist_rm1_sp1.shape}\n\n")
        ''' File "/external/hloc/hloc/matchers/roma.py", line 131, in _forward
    dist_rm0_sp0 = torch.cdist(rm0, torch.tensor(sp0).to(self.device), p=2)
  File "/usr/local/lib/python3.10/dist-packages/torch/functional.py", line 1505, in cdist
    return _VF.cdist(x1, x2, p, None)  # type: ignore[attr-defined]
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double'''

        # print(f"len dist_rm0_sp0: {len(dist_rm0_sp0)}, len dist_rm1_sp1: {len(dist_rm1_sp1)}")
        dists_4_sp0, inds_4_sp0 = dist_rm0_sp0.topk(min(len(sp0),4), largest=False, sorted=True)
        dists_4_sp1, inds_4_sp1 = dist_rm1_sp1.topk(min(len(sp1),4), largest=False, sorted=True)
        dists_4_sp0, inds_4_sp0 = dists_4_sp0.cpu().numpy(), inds_4_sp0.cpu().numpy()
        dists_4_sp1, inds_4_sp1 = dists_4_sp1.cpu().numpy(), inds_4_sp1.cpu().numpy()
    

        dist_threshold = self.dist_threshold
        adding_index_rm0 = set()
        adding_index_rm1 = set()
        dict_filter_rm0 = {} # (int) index_in_rm0: (int)sum of 4 index of sp0
        dict_filter_4distance_rm0 = {} # (int) sum of 4 index of sp0: (float) sum of 4 distance
        dict_filter_rm1 = {}
        dict_filter_4distance_rm1 = {}

        index_2_less_than_threshold = list()
        index_nearest_2_kpts = list()
        index_nearest_query_kpts = list()
        index_nearest_map_kpts = list()
        index_2_larger_than_threshold = list()
        print(f"shape dists_4_sp0 {dists_4_sp0.shape}")
        for i in range(len(rm0)):
            if dists_4_sp0.shape[1] != 0:
                if dists_4_sp0[i][0] > 4*dist_threshold:
                    adding_index_rm0.add(i)
                    index_nearest_query_kpts.append(-1)
                elif dists_4_sp0[i][0] <= dist_threshold:
                    index_nearest_query_kpts.append(inds_4_sp0[i][0])
                    sum_distance = float(sum(dists_4_sp0[i]))
                    s_4_ids = int(sum(inds_4_sp0[i]))
                    if s_4_ids not in dict_filter_4distance_rm0:
                        dict_filter_4distance_rm0[s_4_ids] = sum_distance
                        dict_filter_rm0[s_4_ids] = i
                    elif dict_filter_4distance_rm0[s_4_ids] > sum_distance:
                        dict_filter_4distance_rm0[s_4_ids] = sum_distance
                        dict_filter_rm0[s_4_ids] = i
                else:
                    index_nearest_query_kpts.append(-1)
            else:
                adding_index_rm0.add(i)
                index_nearest_query_kpts.append(-1)
        print(f"shape dists_4_sp1 {dists_4_sp1.shape}")
        for i in range(len(rm1)):
            if dists_4_sp1.shape[1] != 0:
                if len(rm1) != dists_4_sp1.shape[0]:
                    print(f"len rm1 {len(rm1)} and shape dists_4_sp1 {dists_4_sp1.shape}")
                if dists_4_sp1[i][0] > 3*dist_threshold:
                    adding_index_rm1.add(i)
                    index_nearest_map_kpts.append(-1)
                elif dists_4_sp1[i][0] <= dist_threshold:
                    index_nearest_map_kpts.append(inds_4_sp1[i][0])
                    sum_distance = float(sum(dists_4_sp1[i]))
                    s_4_ids = int(sum(inds_4_sp1[i]))
                    if s_4_ids not in dict_filter_4distance_rm1:
                        dict_filter_4distance_rm1[s_4_ids] = sum_distance
                        dict_filter_rm1[s_4_ids] = i
                    elif dict_filter_4distance_rm1[s_4_ids] > sum_distance:
                        dict_filter_4distance_rm1[s_4_ids] = sum_distance
                        dict_filter_rm1[s_4_ids] = i
                else:
                    index_nearest_map_kpts.append(-1)
            else:
                adding_index_rm1.add(i)
                index_nearest_map_kpts.append(-1)

        
        
        filter_set_rm0 = set(dict_filter_rm0.values())
        filter_set_rm1 = set(dict_filter_rm1.values())

        index_map_less_query_larger = []
        index_keypoint_map_less_sp = []

        index_map_larger_query_less = []
        index_keypoint_query_less_sp = []

        for i in range(len(rm0)):
            if i in adding_index_rm0 and i in adding_index_rm1:
                index_2_larger_than_threshold.append(i)
            elif i in filter_set_rm0 and i in filter_set_rm1:
                index_2_less_than_threshold.append(i)
                index_nearest_2_kpts.append([index_nearest_query_kpts[i], index_nearest_map_kpts[i]])
            elif i in adding_index_rm0 and i in filter_set_rm1:
                index_map_less_query_larger.append(i)
                index_keypoint_map_less_sp.append(index_nearest_map_kpts[i])
            elif i in filter_set_rm0 and i in adding_index_rm1:
                index_map_larger_query_less.append(i)
                index_keypoint_query_less_sp.append(index_nearest_query_kpts[i])

        keypoints_match0_less_than_threshold = rm0[index_2_less_than_threshold]
        keypoints_match0_larger_than_threshold = rm0[index_2_larger_than_threshold]
        keypoints_match0_map_less_query_larger = rm0[index_map_less_query_larger]
        keypoints_match0_query_less_sp = sp0[index_keypoint_query_less_sp]

        keypoint_match1_less_than_threshold = rm1[index_2_less_than_threshold]
        keypoint_match1_larger_than_threshold = rm1[index_2_larger_than_threshold]
        keypoint_match1_map_less_sp = sp1[index_keypoint_map_less_sp]
        keypoint_match1_map_larger_query_less = rm1[index_map_larger_query_less]

        score_less_than_threshold = mconf[index_2_less_than_threshold].cpu().numpy()
        score_larger_than_threshold = mconf[index_2_larger_than_threshold].cpu().numpy()
        score_map_less_query_larger = mconf[index_map_less_query_larger].cpu().numpy()
        score_map_larger_query_less = mconf[index_map_larger_query_less].cpu().numpy()


        return keypoints_match0_less_than_threshold, keypoint_match1_less_than_threshold, score_less_than_threshold, index_nearest_2_kpts, \
            keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
            keypoints_match0_map_less_query_larger, torch.tensor(keypoint_match1_map_less_sp), score_map_less_query_larger, \
            torch.tensor(keypoints_match0_query_less_sp), keypoint_match1_map_larger_query_less, score_map_larger_query_less


        
    
    def to_pixel_coordinates(self, coords, H_A, W_A, H_B=None, W_B=None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A)

        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(
            kpts_B, H_B, W_B
        )

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack(
            (W / 2 * (coords[..., 0] + 1), H / 2 * (coords[..., 1] + 1)), axis=-1
        )
        return kpts