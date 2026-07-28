import torch
import warnings
from kornia.feature.loftr.loftr import default_cfg
from kornia.feature import LoFTR as LoFTR_

from ..utils.base_model import BaseModel

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LoFTR1(BaseModel):
    default_conf = {
        'weights': 'indoor',
        'match_threshold': 0.2,
        'max_num_matches': None,
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        self.dist_threshold = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = default_cfg
        cfg['match_coarse']['thr'] = conf['match_threshold']
        self.net = LoFTR_(pretrained=conf['weights'], config=cfg)

    def _forward(self, data):
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            'keypoints0': 'keypoints1',
            'keypoints1': 'keypoints0',
            'image0': 'image1',
            'image1': 'image0',
            'mask0': 'mask1',
            'mask1': 'mask0',
        }
        keypoints_sp0 = data["keypoints_orgin0"]
        keypoints_sp1 = data["keypoints_orgin1"]
        del data["keypoints_orgin0"]
        del data["keypoints_orgin1"]
        data_ = {rename[k]: v for k, v in data.items()}
        print(f"data_ keys: {data_.keys()}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] =\
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred['scores'] = scores
        del pred['confidence']

        kpts0_loftr = pred['keypoints0']
        kpts1_loftr = pred['keypoints1']
        scores_loftr = pred['scores']
        
        ## print shape of kpts0_loftr, kpts1_loftr, scores_loftr
        print(f"kpts0_loftr shape: {kpts0_loftr.shape}, kpts1_loftr shape: {kpts1_loftr.shape}, scores_loftr shape: {scores_loftr.shape}")
        ## using ast tree to refine the keypoints
        dist_loftr0_sp0 = torch.cdist(kpts0_loftr, torch.tensor(keypoints_sp0).to(self.device).float(), p=2)
        dist_loftr1_sp1 = torch.cdist(kpts1_loftr, torch.tensor(keypoints_sp1).to(self.device).float(), p=2)
        print(f"dist_loftr0_sp0 shape: {dist_loftr0_sp0.shape}, dist_loftr1_sp1 shape: {dist_loftr1_sp1.shape}")
        ## process keypoints by distance
        dists_4_sp0, inds_4_sp0 = dist_loftr0_sp0.topk(min(len(keypoints_sp0),4), largest=False, sorted=True)
        dists_4_sp1, inds_4_sp1 = dist_loftr1_sp1.topk(min(len(keypoints_sp1),4), largest=False, sorted=True)
        dists_4_sp0, inds_4_sp0 = dists_4_sp0.cpu().numpy(), inds_4_sp0.cpu().numpy()
        dists_4_sp1, inds_4_sp1 = dists_4_sp1.cpu().numpy(), inds_4_sp1.cpu().numpy()
    

        dist_threshold = self.dist_threshold
        adding_index_loftr0 = set()
        adding_index_loftr1 = set()
        dict_filter_loftr0 = {} # (int) index_in_loftr0: (int)sum of 4 index of keypoints_sp0
        dict_filter_4distance_loftr0 = {} # (int) sum of 4 index of keypoints_sp0: (float) sum of 4 distance
        dict_filter_loftr1 = {}
        dict_filter_4distance_loftr1 = {}

        index_2_less_than_threshold = list()
        index_nearest_2_kpts = list()
        index_nearest_query_kpts = list()
        index_nearest_map_kpts = list()
        index_2_larger_than_threshold = list()
        
        print(f"shape dists_4_sp0 {dists_4_sp0.shape}")
        for i in range(len(kpts0_loftr)):
            if dists_4_sp0.shape[1] != 0:
                if dists_4_sp0[i][0] > 4*dist_threshold:
                    adding_index_loftr0.add(i)
                    index_nearest_query_kpts.append(-1)
                elif dists_4_sp0[i][0] <= dist_threshold:
                    index_nearest_query_kpts.append(inds_4_sp0[i][0])
                    sum_distance = float(sum(dists_4_sp0[i]))
                    s_4_ids = int(sum(inds_4_sp0[i]))
                    if s_4_ids not in dict_filter_4distance_loftr0:
                        dict_filter_4distance_loftr0[s_4_ids] = sum_distance
                        dict_filter_loftr0[s_4_ids] = i
                    elif dict_filter_4distance_loftr0[s_4_ids] > sum_distance:
                        dict_filter_4distance_loftr0[s_4_ids] = sum_distance
                        dict_filter_loftr0[s_4_ids] = i
                else:
                    index_nearest_query_kpts.append(-1)
            else:
                adding_index_loftr0.add(i)
                index_nearest_query_kpts.append(-1)
        print(f"shape dists_4_sp1 {dists_4_sp1.shape}")
        for i in range(len(kpts1_loftr)):
            if dists_4_sp1.shape[1] != 0:
                if len(kpts1_loftr) != dists_4_sp1.shape[0]:
                    print(f"len loftr1 {len(kpts1_loftr)} and shape dists_4_sp1 {dists_4_sp1.shape}")
                if dists_4_sp1[i][0] > 4*dist_threshold:
                    adding_index_loftr1.add(i)
                    index_nearest_map_kpts.append(-1)
                elif dists_4_sp1[i][0] <= dist_threshold:
                    index_nearest_map_kpts.append(inds_4_sp1[i][0])
                    sum_distance = float(sum(dists_4_sp1[i]))
                    s_4_ids = int(sum(inds_4_sp1[i]))
                    if s_4_ids not in dict_filter_4distance_loftr1:
                        dict_filter_4distance_loftr1[s_4_ids] = sum_distance
                        dict_filter_loftr1[s_4_ids] = i
                    elif dict_filter_4distance_loftr1[s_4_ids] > sum_distance:
                        dict_filter_4distance_loftr1[s_4_ids] = sum_distance
                        dict_filter_loftr1[s_4_ids] = i
                else:
                    index_nearest_map_kpts.append(-1)
            else:
                adding_index_loftr1.add(i)
                index_nearest_map_kpts.append(-1)

        
        
        filter_set_loftr0 = set(dict_filter_loftr0.values())
        filter_set_loftr1 = set(dict_filter_loftr1.values())

        index_map_less_query_larger = []
        index_keypoint_map_less_sp = []

        index_map_larger_query_less = []
        index_keypoint_query_less_sp = []

        for i in range(len(kpts0_loftr)):
            if i in adding_index_loftr0 and i in adding_index_loftr1:
                index_2_larger_than_threshold.append(i)
            elif i in filter_set_loftr0 and i in filter_set_loftr1:
                index_2_less_than_threshold.append(i)
                index_nearest_2_kpts.append([index_nearest_query_kpts[i], index_nearest_map_kpts[i]])
            elif i in adding_index_loftr0 and i in filter_set_loftr1:
                index_map_less_query_larger.append(i)
                index_keypoint_map_less_sp.append(index_nearest_map_kpts[i])
            elif i in filter_set_loftr0 and i in adding_index_loftr1:
                index_map_larger_query_less.append(i)
                index_keypoint_query_less_sp.append(index_nearest_query_kpts[i])

        keypoints_sp0 = torch.tensor(keypoints_sp0).to(self.device).float()
        keypoints_sp1 = torch.tensor(keypoints_sp1).to(self.device).float()
        keypoints_match0_less_than_threshold = kpts0_loftr[index_2_less_than_threshold]
        keypoints_match0_larger_than_threshold = kpts0_loftr[index_2_larger_than_threshold]
        keypoints_match0_map_less_query_larger = kpts0_loftr[index_map_less_query_larger]
        keypoints_match0_query_less_sp = keypoints_sp0[index_keypoint_query_less_sp]

        keypoint_match1_less_than_threshold = kpts1_loftr[index_2_less_than_threshold]
        keypoint_match1_larger_than_threshold = kpts1_loftr[index_2_larger_than_threshold]
        keypoint_match1_map_less_sp = keypoints_sp1[index_keypoint_map_less_sp]
        keypoint_match1_map_larger_query_less = kpts1_loftr[index_map_larger_query_less]

        score_less_than_threshold = scores[index_2_less_than_threshold].cpu().numpy()
        score_larger_than_threshold = scores[index_2_larger_than_threshold].cpu().numpy()
        score_map_less_query_larger = scores[index_map_less_query_larger].cpu().numpy()
        score_map_larger_query_less = scores[index_map_larger_query_less].cpu().numpy()

        return keypoints_match0_less_than_threshold, keypoint_match1_less_than_threshold, score_less_than_threshold, index_nearest_2_kpts, \
            keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
            keypoints_match0_map_less_query_larger, torch.tensor(keypoint_match1_map_less_sp), score_map_less_query_larger, \
            torch.tensor(keypoints_match0_query_less_sp), keypoint_match1_map_larger_query_less, score_map_larger_query_less
 