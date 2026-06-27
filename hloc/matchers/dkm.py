import torch
from ..utils.base_model import BaseModel
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from DKM.dkm.models.model_zoo.DKMv3 import DKMv3

weight_urls = {
    "DKMv3": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/dkmv3/DKMv3_indoor.pth",
    },
}

class DKM_MNatches(BaseModel):
    def _init(self, conf):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if conf['weight_mode'] == 'indoor':
            self.weight = torch.hub.load_state_dict_from_url(
                weight_urls['DKMv3']['indoor'], map_location=self.device
            )
            self.upsample_preds = False
        elif conf['weight_mode'] == 'outdoor':
            self.weight = torch.hub.load_state_dict_from_url(
                weight_urls['DKMv3']['outdoor'], map_location=self.device
            )
            self.upsample_preds = True
        self.size = conf['resize_max']
        self.max_keypoints = conf['max_keypoints']
        try:
            self.dist_threshold = float(conf['dist_threshold'])
        except:
            self.dist_threshold = 2.0

        self.net = DKMv3(
            weights = self.weight,
            h = self.size,
            w = self.size,
            upsample_preds = self.upsample_preds,
            device = self.device
        )

    # def _forward(self, data):
    #     img0, img1 = data[0], data[2]
    #     sp0, sp1 = data[1], data[3]
    #     ##modifi
    #     sp0 = torch.as_tensor(sp0, device=self.device).float()
    #     sp1 = torch.as_tensor(sp1, device=self.device).float()
    #     #
    #     W0, H0 = img0.size
    #     W1, H1 = img1.size

    #     matches, mconf = self.net.match(
    #         img0, img1,
    #         device=self.device
    #     )

    #     max_keypoints = self.max_keypoints

    #     matches, mconf = self.net.sample(
    #         matches,
    #         mconf,
    #         self.max_keypoints
    #     )

    #     dkm_kpts0, dkm_kpts1 = self.net.to_pixel_coordinates(matches, H0, W0, H1, W1)
    #     # sp0, sp1 = sp0[0], sp1[0]
    #     mask = (mconf >= 0.8)
        
    #     dkm_kpts0, dkm_kpts1 = dkm_kpts0[mask], dkm_kpts1[mask]
        
    #     ##modifi
    #     dist_dkm0_sp0 = torch.cdist(dkm_kpts0, sp0)
    #     dist_dkm1_sp1 = torch.cdist(dkm_kpts1, sp1)
    #     #
    #     print(f"dkm_kpts0 shape: {dkm_kpts0.shape}, sp0 shape: {sp0.shape}, dkm_kpts1 shape: {dkm_kpts1.shape}, sp1 shape: {sp1.shape}")
    #     # if shape[0] of sp0 or sp1 is 0, then return empty tensors
    #     if sp0.shape[0] == 0 or sp1.shape[0] == 0 or dkm_kpts0.shape[0] == 0:
    #         return torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), torch.empty((0, 2), dtype=torch.long, device=self.device), \
    #             torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), \
    #             torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), \
    #             torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device)
    #     ##modifi
    #     k0 = min(4, len(sp0))
    #     k1 = min(4, len(sp1))
    #     dists_4_sp0, inds_4_sp0 = dist_dkm0_sp0.topk(k0, largest=False)
    #     dists_4_sp1, inds_4_sp1 = dist_dkm1_sp1.topk(k1, largest=False)
        
    #     dist_threshold = self.dist_threshold
    #     ##modifi
    #     nearest0 = dists_4_sp0[:,0]
    #     nearest1 = dists_4_sp1[:,0]

    #     mask_add0 = nearest0 > 4*dist_threshold
    #     mask_add1 = nearest1 > 4*dist_threshold

    #     mask_refine0 = nearest0 <= dist_threshold
    #     mask_refine1 = nearest1 <= dist_threshold
    #     sum_dist0 = dists_4_sp0.sum(1)
    #     sum_dist1 = dists_4_sp1.sum(1)

    #     sum_id0 = inds_4_sp0.sum(1)
    #     sum_id1 = inds_4_sp1.sum(1)
    #     filter_set_dkm0 = set()
    #     best_idx0 = []

    #     for sid in torch.unique(sum_id0):
    #         mask = (sum_id0 == sid)

    #         idxs = torch.where(mask)[0]
    #         best = idxs[torch.argmin(sum_dist0[idxs])]

    #         filter_set_dkm0.add(best.item())
    #     filter_set_dkm1 = set()
    #     best_idx1 = []

    #     for sid in torch.unique(sum_id1):
    #         mask = (sum_id1 == sid)

    #         idxs = torch.where(mask)[0]
    #         best = idxs[torch.argmin(sum_dist1[idxs])]

    #         filter_set_dkm1.add(best.item())
    #     filter_mask0 = torch.zeros(len(dkm_kpts0), dtype=torch.bool, device=self.device)
    #     filter_mask1 = torch.zeros(len(dkm_kpts1), dtype=torch.bool, device=self.device)

    #     filter_mask0[list(filter_set_dkm0)] = True
    #     filter_mask1[list(filter_set_dkm1)] = True
    #     mask_less = (
    #         filter_mask0 &
    #         filter_mask1 &
    #         mask_refine0 &
    #         mask_refine1
    #     )
    #     mask_large = mask_add0 & mask_add1
    #     mask_map_less_query_large = (
    #         mask_add0 &
    #         filter_mask1 &
    #         mask_refine1
    #     )
    #     mask_map_large_query_less = (
    #         filter_mask0 &
    #         mask_refine0 &
    #         mask_add1
    #     )
    #     nearest_query = inds_4_sp0[:,0]
    #     nearest_map = inds_4_sp1[:,0]
    #     keypoints_match0_less_than_threshold = dkm_kpts0[mask_less]
    #     keypoint_match1_less_than_threshold = dkm_kpts1[mask_less]

    #     index_nearest_2_kpts = torch.stack([
    #         nearest_query[mask_less],
    #         nearest_map[mask_less]
    #     ],dim=1)
    #     keypoints_match0_larger_than_threshold = dkm_kpts0[mask_large]
    #     keypoint_match1_larger_than_threshold = dkm_kpts1[mask_large]
    #     keypoints_match0_map_less_query_larger = dkm_kpts0[mask_map_less_query_large]

    #     keypoint_match1_map_less_sp = sp1[
    #         nearest_map[mask_map_less_query_large]
    #     ]
    #     keypoints_match0_query_less_sp = sp0[
    #         nearest_query[mask_map_large_query_less]
    #     ]

    #     keypoint_match1_map_larger_query_less = dkm_kpts1[
    #         mask_map_large_query_less
    #     ]
    #     score_less_than_threshold = mconf[mask_less]
    #     score_larger_than_threshold = mconf[mask_large]

    #     score_map_less_query_larger = mconf[mask_map_less_query_large]

    #     score_map_larger_query_less = mconf[mask_map_large_query_less]
    #     #
    #     return keypoints_match0_less_than_threshold, keypoint_match1_less_than_threshold, score_less_than_threshold, index_nearest_2_kpts, \
    #         keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
    #         keypoints_match0_map_less_query_larger, keypoint_match1_map_less_sp, score_map_less_query_larger, \
    #          keypoints_match0_query_less_sp, keypoint_match1_map_larger_query_less, score_map_larger_query_less


    def _forward(self, data):
        img0, img1 = data[0], data[2]
        sp0, sp1 = data[1], data[3]
    
        W0, H0 = img0.size
        W1, H1 = img1.size

        matches, mconf = self.net.match(
            img0, img1,
            device=self.device
        )

        max_keypoints = self.max_keypoints

        matches, mconf = self.net.sample(
            matches,
            mconf,
            self.max_keypoints
        )

        dkm_kpts0, dkm_kpts1 = self.net.to_pixel_coordinates(matches, H0, W0, H1, W1)
        # sp0, sp1 = sp0[0], sp1[0]
        mask = (mconf >= 0.8)
        
        dkm_kpts0, dkm_kpts1 = dkm_kpts0[mask], dkm_kpts1[mask]

        # sp0 = torch.tensor(sp0).to(self.device).float()
        # sp1 = torch.tensor(sp1).to(self.device).float()

        # if sp0.dim() == 1: sp0 = sp0.unsqueeze(0)
        # if sp1.dim() == 1: sp1 = sp1.unsqueeze(0)

        # if dkm_kpts0.dim() == 1: dkm_kpts0 = dkm_kpts0.unsqueeze(0)
        # if dkm_kpts1.dim() == 1: dkm_kpts1 = dkm_kpts1.unsqueeze(0)

        # len_sp0 = sp0.shape[0] if sp0.dim() == 2 else 0
        # len_sp1 = sp1.shape[0] if sp1.dim() == 2 else 0

        # if len_sp0 == 0 or len_sp1 == 0 or dkm_kpts0.shape[0] == 0:
        #     dist_dkm0_sp0 = torch.empty((dkm_kpts0.shape[0], len_sp0), device=self.device)
        #     dist_dkm1_sp1 = torch.empty((dkm_kpts1.shape[0], len_sp1), device=self.device)
        # else:
            # print shape of dkm_kpts0, sp0, dkm_kpts1, sp1
        print(f"dkm_kpts0 shape: {dkm_kpts0.shape}, sp0 shape: {sp0.shape}, dkm_kpts1 shape: {dkm_kpts1.shape}, sp1 shape: {sp1.shape}")
        if sp0.shape[0] == 0 or sp1.shape[0] == 0 or dkm_kpts0.shape[0] == 0 or dkm_kpts1.shape[0] == 0:
            return torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), torch.empty((0, 2), dtype=torch.long, device=self.device), \
                torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), \
                torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device), \
                torch.empty((0, 2), device=self.device), torch.empty((0, 2), device=self.device), torch.empty((0,), device=self.device)
        dist_dkm0_sp0 = torch.cdist(dkm_kpts0, torch.tensor(sp0).to(self.device).float(), p=2)
        dist_dkm1_sp1 = torch.cdist(dkm_kpts1, torch.tensor(sp1).to(self.device).float(), p=2)

        dists_4_sp0, inds_4_sp0 = dist_dkm0_sp0.topk(min(len(sp0), 4), largest=False, sorted=True)
        dists_4_sp1, inds_4_sp1 = dist_dkm1_sp1.topk(min(len(sp1), 4), largest=False, sorted=True)
        dists_4_sp0, inds_4_sp0 = dists_4_sp0.cpu().numpy(), inds_4_sp0.cpu().numpy()
        dists_4_sp1, inds_4_sp1 = dists_4_sp1.cpu().numpy(), inds_4_sp1.cpu().numpy()

        dist_threshold = self.dist_threshold
        adding_index_dkm0 = set()
        adding_index_dkm1 = set()
        dict_filter_dkm0 = {}
        dict_filter_dkm1 = {}
        dict_filter_4distance_dkm0 = {}
        dict_filter_4distance_dkm1 = {}

        index_2_less_than_threshold = list()
        index_nearest_2_kpts = list()
        index_nearest_query_kpts = list()
        index_nearest_map_kpts = list()
        index_2_larger_than_threshold = list()

        print(f"shape dists_4_sp0 {dists_4_sp0.shape}")
        for i in range(len(dkm_kpts0)):
            if dists_4_sp0.shape[1] != 0:
                if dists_4_sp0[i][0] > 4*dist_threshold:
                    adding_index_dkm0.add(i)
                    index_nearest_query_kpts.append(-1)
                elif dists_4_sp0[i][0] <= dist_threshold:
                    index_nearest_query_kpts.append(inds_4_sp0[i][0])
                    sum_distance = float(sum(dists_4_sp0[i]))
                    s_4_ids = int(sum(inds_4_sp0[i]))
                    if s_4_ids not in dict_filter_4distance_dkm0:
                        dict_filter_4distance_dkm0[s_4_ids] = sum_distance
                        dict_filter_dkm0[s_4_ids] = i
                    elif dict_filter_4distance_dkm0[s_4_ids] > sum_distance:
                        dict_filter_4distance_dkm0[s_4_ids] = sum_distance
                        dict_filter_dkm0[s_4_ids] = i
                else:
                    index_nearest_query_kpts.append(-1)
            else:
                adding_index_dkm0.add(i)
                index_nearest_query_kpts.append(-1)
        print(f"shape dists_4_sp1 {dists_4_sp1.shape}")
        for i in range(len(dkm_kpts1)):
            if dists_4_sp1.shape[1] != 0:
                if len(dkm_kpts1) != dists_4_sp1.shape[0]:
                    print(f"len dkm1 {len(dkm_kpts1)} and shape dists_4_sp1 {dists_4_sp1.shape}")
                if dists_4_sp1[i][0] > 4*dist_threshold:
                    adding_index_dkm1.add(i)
                    index_nearest_map_kpts.append(-1)
                elif dists_4_sp1[i][0] <= dist_threshold:
                    index_nearest_map_kpts.append(inds_4_sp1[i][0])
                    sum_distance = float(sum(dists_4_sp1[i]))
                    s_4_ids = int(sum(inds_4_sp1[i]))
                    if s_4_ids not in dict_filter_4distance_dkm1:
                        dict_filter_4distance_dkm1[s_4_ids] = sum_distance
                        dict_filter_dkm1[s_4_ids] = i
                    elif dict_filter_4distance_dkm1[s_4_ids] > sum_distance:
                        dict_filter_4distance_dkm1[s_4_ids] = sum_distance
                        dict_filter_dkm1[s_4_ids] = i
                else:
                    index_nearest_map_kpts.append(-1)
            else:
                adding_index_dkm1.add(i)
                index_nearest_map_kpts.append(-1)

        
        filter_set_dkm0 = set(dict_filter_dkm0.values())
        filter_set_dkm1 = set(dict_filter_dkm1.values())

        index_map_less_query_larger = []
        index_keypoint_map_less_sp = []
        
        index_map_larger_query_less = []
        index_keypoint_query_less_sp = []

        for i in range(len(dkm_kpts0)):
            if i in adding_index_dkm0 and i in adding_index_dkm1:
                index_2_larger_than_threshold.append(i)
            elif i in filter_set_dkm0 and i in filter_set_dkm1:
                index_2_less_than_threshold.append(i)
                index_nearest_2_kpts.append([index_nearest_query_kpts[i], index_nearest_map_kpts[i]])
            elif i in adding_index_dkm0 and i in filter_set_dkm1:
                index_map_less_query_larger.append(i)
                index_keypoint_map_less_sp.append(index_nearest_map_kpts[i])
            elif i in filter_set_dkm0 and i in adding_index_dkm1:
                index_map_larger_query_less.append(i)
                index_keypoint_query_less_sp.append(index_nearest_query_kpts[i])
                
        sp0 = torch.tensor(sp0).to(self.device).float()
        sp1 = torch.tensor(sp1).to(self.device).float()
        keypoints_match0_less_than_threshold = dkm_kpts0[index_2_less_than_threshold]
        keypoints_match0_larger_than_threshold = dkm_kpts0[index_2_larger_than_threshold]
        keypoints_match0_map_less_query_larger = dkm_kpts0[index_map_less_query_larger]
        keypoints_match0_query_less_sp = sp0[index_keypoint_query_less_sp]

        keypoint_match1_less_than_threshold = dkm_kpts1[index_2_less_than_threshold]
        keypoint_match1_larger_than_threshold = dkm_kpts1[index_2_larger_than_threshold]
        keypoint_match1_map_less_sp = sp1[index_keypoint_map_less_sp]
        keypoint_match1_map_larger_query_less = dkm_kpts1[index_map_larger_query_less]

        score_less_than_threshold = mconf[index_2_less_than_threshold].cpu().numpy()
        score_larger_than_threshold = mconf[index_2_larger_than_threshold].cpu().numpy()
        score_map_less_query_larger = mconf[index_map_less_query_larger].cpu().numpy()
        score_map_larger_query_less = mconf[index_map_larger_query_less].cpu().numpy()


        return keypoints_match0_less_than_threshold, keypoint_match1_less_than_threshold, score_less_than_threshold, index_nearest_2_kpts, \
            keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
            keypoints_match0_map_less_query_larger, torch.tensor(keypoint_match1_map_less_sp), score_map_less_query_larger, \
            torch.tensor(keypoints_match0_query_less_sp), keypoint_match1_map_larger_query_less, score_map_larger_query_less
