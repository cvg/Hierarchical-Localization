import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from tqdm import tqdm
import h5py
import torch
from PIL import Image
import numpy as np #type: ignore

import cv2

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .utils.io import read_image
import PIL
import matplotlib.pyplot as plt


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superpoint+lightglue': {
        'output': 'matches-superpoint-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'superpoint',
        },
    },
    'disk+lightglue': {
        'output': 'matches-disk-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'disk',
        },
    },
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'lightglue_gim': { ## additional config for LightGlue-GIM
        'output': 'matches-lightglue-gim',
        'model': {
            'name': 'lightglue_gim',
            'features': 'superpoint',
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {
            'name': 'adalam'
        },
    },
    'roma': {
        'output': 'matches-roma',
        'model': {
            'name': 'roma',
            'max_keypoints': 5000,
            'weight_mode': 'indoor',
            'resize_max': 560,
        }
    },
    'lightglue+roma': {
        'output': 'matches-superglue-roma',
        'model': {
            'name': 'roma',
            'max_keypoints': 1024,
            'weight_mode': 'indoor',
            'resize_max': 1024,
        },
        'model2': {
            'name': 'lightglue_gim',
            'features': 'superpoint',
            'preprocessing': {
                'resize_max': 1024,
                'resize_force': True,
            },
        },
    },
    'lightglue+roma1': {
        'name': 'lightglue+roma1',
        'hloc': {
            'output': 'matches-superglue-roma',
            'model': {
                'name': 'roma1',
                'max_keypoints': 1024,
                'weight_mode': 'outdoor',# test, default indoor
                'resize_max': 1024,
                'dist_threshold': 2.0,
            },
            'model2': {
                'name': 'lightglue',
                'features': 'superpoint',
                'preprocessing': {
                    'resize_max': 1024,
                    'resize_force': True,
                },
            }
        }
    },
    'lightglue+dkm': {
        'name': 'lightglue+dkm',
        'hloc': {
            'output': 'matches-superglue-dkm',
            'model': {
                'name': 'dkm',
                'max_keypoints': 2048,
                'weight_mode': 'indoor',
                'resize_max': 1024,
                'dist_threshold': 2.0,
            },
            'model2': {
                'name': 'lightglue',
                'features': 'superpoint',
                'preprocessing': {
                    'resize_max': 1024,
                    # 'resize_force': True,
                },
            }
        }
    },
    "lightglue+loftr": {
        "name": "lightglue+loftr",
        "hloc": {
            "output": "matches-loftr",
            "model": {
                "name": "loftr1",
                "max_num_matches": 4096,
                "weights": "indoor",
                "resize_max": 1024
            },
            'model2': {
                'name': 'lightglue',
                'features': 'superpoint',
                'preprocessing': {
                    'resize_max': 1024,
                },
            } 
        }
    }
}

path_log_file_start_matched = Path(__file__).parent.parent / 'log_start_matched.txt' ## each line: img_path0, img_path1
path_log_file_end_matched = Path(__file__).parent.parent / 'log_end_matched.txt' ## each_line: img_name0, img_name1
if not path_log_file_start_matched.exists():
    path_log_file_start_matched.touch()
if not path_log_file_end_matched.exists():
    path_log_file_end_matched.touch()


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized

class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                if k == 'image_path':
                    v = v.__array__().astype(str)
                    data[k + "0"] = v[0]
                    continue

                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                if k == 'image_path':
                    v = v.__array__().astype(str)
                    data[k + "1"] = v[0]
                    continue
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)

class FeaturePairsDatasetRoMa(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r
    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx] # name0: name of first element in the pair, name1: name of second element in the pair
        data0 = {}
        data1 = {}
        with h5py.File(self.feature_path_q, 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                if k == 'image_path':
                    v = v.__array__().astype(str)
                    data0[k] = v[0]
                    continue
                try:
                    data0[int(k)] = torch.from_numpy(v.__array__()).float()
                except:
                    data0[k] = torch.from_numpy(v.__array__()).float()
            
        data0['image_name'] = name0

        with h5py.File(self.feature_path_r, 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                if k == 'image_path':
                    v = v.__array__().astype(str)
                    data1[k] = v[0]
                    continue
                try:
                    data1[int(k)] = torch.from_numpy(v.__array__()).float()
                except:
                    data1[k] = torch.from_numpy(v.__array__()).float()
        data1['image_name'] = name1
        return data0, data1

    def __len__(self):
        return len(self.pairs)



def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

def writer_fn_sub(inp, match_path):
    pair, pred = inp
    matches = pred['matches0'][0].cpu().short().numpy()
    matches = np.expand_dims(matches, axis=-1)
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            # add new data
            del fd[pair]
            raise NotImplementedError
        else:
            # make new data
            grp = fd.create_group(pair)
            grp.create_dataset('matches0', data=matches)
            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                scores = np.expand_dims(scores, axis=-1)
                grp.create_dataset('matching_scores0', data=scores)
        raise NotImplementedError
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

def writer_fn_t(inp, match_path):
    pair, matches, scores = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        grp.create_dataset('matches0', data=matches)
        grp.create_dataset('matching_scores0', data=scores)

def writer_roma_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0']
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0']
            grp.create_dataset('matching_scores0', data=scores)

def main(conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False,
         dict_keypoints_index_query: Optional[Path] = None,
         dict_keypoints_index_map: Optional[Path] = None,
         is_query_map_match: bool = False,
         feature_path_raw_ref: Optional[Path] = None,
         method_baseline = False,
         method1 = False,
         method2 = True,
         method3 = True):

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features+'.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    if "loftr" in conf['model']['name']:
        match_from_paths_loftr(conf, pairs, matches, features_q, features_ref, overwrite, dict_keypoints_index_query, dict_keypoints_index_map, is_query_map_match=is_query_map_match, feature_path_raw_ref=feature_path_raw_ref, method_baseline=method_baseline, method1=method1, method2=method2, method3=method3)
    else:
        match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, dict_keypoints_index_query, dict_keypoints_index_map, is_query_map_match=is_query_map_match, feature_path_raw_ref=feature_path_raw_ref, method_baseline=method_baseline, method1=method1, method2=method2, method3=method3)
    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs

def get_image(image_path, interp, resize_max = None, grayscale=False):
    image = read_image(image_path, grayscale)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    # image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    if resize_max is None or max(size) <= resize_max:
        return image, 1.0
    scale = resize_max / max(size)
    size_new = tuple(int(round(x*scale)) for x in size)
    image = resize_image(image, size_new, interp)
    return image, np.mean([size[0]/size_new[0], size[1]/size_new[1]])

def match_from_paths_glue(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False):
    
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    # print(f"----------------len pairs: {len(pairs)}--------------------")

    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    # matched_pairs = set() # ((img_name0, img_name1))
    # # open file end_matched
    # with open(path_log_file_end_matched, 'r') as f:
    #     for line in f:
    #         img_name0, img_name1 = line.strip().split(',')
    #         matched_pairs.add((img_name0, img_name1))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    Model_lg = dynamic_load(matchers, conf['model2']['name'])
    model_lg = Model_lg(conf['model2']).eval().to(device)
    dataset = FeaturePairsDatasetRoMa(pairs, feature_path_q, feature_path_ref)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
        
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    # loader = torch.utils.data.DataLoader(
    #     dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)
    for idx, data in enumerate(tqdm(loader, smoothing=.1)):            
        data0 = {k: v if str(k).startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data[0].items()}
        data1 = {k: v if str(k).startswith('image')
                else v.to(device, non_blocking=True) for k, v in data[1].items()}
        data = {f"{k}0": v for k, v in data0.items()}
        data.update({f"{k}1": v for k, v in data1.items()})

        name_img0 = data0['image_name'][0]
        name_img1 = data1['image_name'][0]
        pair_img_curr = (name_img0, name_img1)
        # if pair_img_curr in matched_pairs:
        #     print(f"pair {pair_img_curr} already matched, skip.")
        #     continue
        img_path0 = data0['image_path'][0]
        img_path1 = data1['image_path'][0]
    
        interp = 'cv2_area'
        resize_max = conf['model2']['preprocessing']['resize_max']
        image0, _ = get_image(img_path0, interp, resize_max)
        image1, _ = get_image(img_path1, interp, resize_max)

        image0 = Image.fromarray(image0.astype(np.uint8))
        image1 = Image.fromarray(image1.astype(np.uint8))

        # print(f'keypoint_matches0_roma: {keypoint_matches0_roma.shape}, keypoint_matches1_roma: {keypoint_matches1_roma.shape}, score: {score.shape}')
        # print(f"data0['keypoints']: {data0['keypoints'].shape}, data1['keypoints'].shape: {data1['keypoints'].shape}")
        if len(data0['keypoints'].shape) == 4:
            keypts0 = data0['keypoints'][0].to(device)
        else:
            keypts0 = data0['keypoints'].to(device)
        if len(data1['keypoints'].shape) == 4:
            keypts1 = data1['keypoints'][0].to(device)
        else:
            keypts1 = data1['keypoints'].to(device)
        # if len(keypts0.shape) == 3:
        #     keypts0 = keypts0.squeeze()
        # if len(keypts1.shape) == 3:
        #     keypts1 = keypts1.squeeze()
        # print(f"keypts0.shape: {keypts0.shape}, keypts1.shape: {keypts1.shape}")
        #
        # keypts0 = data0['keypoints'][0].unsqueeze(0)
        # keypts1 = data1['keypoints'][0].unsqueeze(0)
        scores0 = data0['scores'][0].unsqueeze(0).to(device)
        scores1 = data1['scores'][0].unsqueeze(0).to(device)
        image_size0 = data0['image_size'].to(device)
        image_size1 = data1['image_size'].to(device)
        desc0 = data0['descriptors'].permute(0,2,1)
        desc0 = desc0[0]
        desc0 = desc0.unsqueeze(0).permute(0,2,1).to(device)
        desc1 = data1['descriptors'].permute(0,2,1)
        desc1 = desc1[0]
        desc1 = desc1.unsqueeze(0).permute(0,2,1).to(device)
        # change PIL image to torch tensor
        img0 = torch.from_numpy(np.array(image0)).permute(2, 0, 1).unsqueeze(0).to(device)
        img1 = torch.from_numpy(np.array(image1)).permute(2, 0, 1).unsqueeze(0).to(device)
        data_input_lg= {
            'descriptors0': desc0,
            'descriptors1': desc1,
            'keypoints0': keypts0,
            'keypoints1': keypts1,
            'scores0': scores0,
            'scores1': scores1,
            'image_size0': image_size0,
            'image_size1': image_size1,
            'image0': img0,
            'image1': img1
        }
        pred_lg = model_lg(data_input_lg)

        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred_lg))
    writer_queue.join()
    return pairs

def split_image(image: np.ndarray, patch_number: int):
    h, w = image.shape[:2]
    sub_h = h//2
    sub_w = w//2
    if patch_number == 0:
        return image
    elif patch_number == 1:
        return image[:sub_h,:sub_w]
    elif patch_number == 2:
        return image[:sub_h,sub_w:]
    elif patch_number == 3:
        return image[sub_h:,:sub_w]
    else:
        return image[sub_h:,sub_w:]

def process_existed_kpts(kpts: np.ndarray, patch_number: int, im_size):
    kpts_cpy = kpts.copy()
    if patch_number == 0:
        return kpts_cpy
    elif patch_number == 1:
        kpts_cpy*=2
    elif patch_number == 2:
        kpts_cpy*=2
        kpts_cpy[:,0] -= im_size[0] # width
    elif patch_number == 3:
        kpts_cpy*=2
        kpts_cpy[:,1] -= im_size[1] # height
    else:
        kpts_cpy*=2
        kpts_cpy-=(im_size[0], im_size[1])
    return kpts_cpy

def restore_coords(keypoints, patch_number, size, device):
    # if len keypoint == 0, return empty tensor
    if keypoints.shape[0] == 0:
        return keypoints
    # print(f"restore keypoints shape {keypoints.shape}")
    if patch_number == 0:
        return keypoints
    elif patch_number == 1:
        keypoints = keypoints//2
    elif patch_number == 2:
        keypoints = keypoints//2
        keypoints[:,0] += (size[0]//2)
    elif patch_number == 3:
        keypoints = keypoints//2
        keypoints[:,1] += (size[1]//2)
    else:
        keypoints = keypoints//2
        keypoints += torch.tensor([size[0]//2, size[1]//2]).to(keypoints.device)
    return keypoints

def get_patch_bbox(H, W, patch_id):
    """Trả về tọa độ (x_min, y_min, x_max, y_max) của patch dựa trên ID."""
    h_half, w_half = H // 2, W // 2
    
    if patch_id == 0:   # Full ảnh
        return 0, 0, W, H
    elif patch_id == 1: # Góc trên - trái
        return 0, 0, w_half, h_half
    elif patch_id == 2: # Góc trên - phải
        return w_half, 0, W, h_half
    elif patch_id == 3: # Góc dưới - trái
        return 0, h_half, w_half, H
    elif patch_id == 4: # Góc dưới - phải
        return w_half, h_half, W, H
    else:
        raise ValueError("ID chỉ được nằm trong khoảng 0 đến 4")

def draw_custom_matches(img1, img2, kpts1, kpts2, matches, id1=0, id2=0):
    """
    Vẽ matches giữa 2 ảnh dựa trên id1 và id2.
    """
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    
    # Lấy tọa độ vùng (bbox) cần giữ lại cho từng ảnh
    x1_min, y1_min, x1_max, y1_max = get_patch_bbox(H1, W1, id1)
    x2_min, y2_min, x2_max, y2_max = get_patch_bbox(H2, W2, id2)
    
    # --- 1. TẠO CANVAS TRẮNG ---
    max_h = max(H1, H2)
    total_w = W1 + W2
    canvas = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
    
    # --- 2. DÁN ẢNH 1 VÀ VẼ KHUNG ---
    canvas[y1_min:y1_max, x1_min:x1_max] = img1[y1_min:y1_max, x1_min:x1_max]
    cv2.rectangle(canvas, (x1_min, y1_min), (x1_max, y1_max), (0, 0, 255), 2)
    cv2.putText(canvas, f"img1 (id={id1})", (x1_min + 10, y1_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 3. DÁN ẢNH 2 VÀ VẼ KHUNG ---
    # Tọa độ X trên canvas phải cộng thêm W1 (vì ảnh 2 nằm bên phải)
    canvas[y2_min:y2_max, W1 + x2_min : W1 + x2_max] = img2[y2_min:y2_max, x2_min:x2_max]
    cv2.rectangle(canvas, (W1 + x2_min, y2_min), (W1 + x2_max, y2_max), (0, 0, 255), 2)
    cv2.putText(canvas, f"img2 (id={id2})", (W1 + x2_min + 10, y2_min + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- 4. VẼ KEYPOINTS VÀ MATCHES ---
    m_cols = matches.shape[1]
    
    # Vẽ keypoints cho ảnh 2 (chỉ vẽ nếu nằm trong bbox của id2)
    for pt in kpts2:
        x2, y2 = int(pt[0]), int(pt[1])
        if x2_min <= x2 < x2_max and y2_min <= y2 < y2_max:
            cv2.circle(canvas, (x2 + W1, y2), 3, (0, 0, 255), -1)

    # Lặp qua từng keypoint của ảnh 1
    for i, pt1 in enumerate(kpts1):
        x1, y1 = int(pt1[0]), int(pt1[1])
        
        # Chỉ xử lý nếu keypoint 1 nằm trong bbox của id1
        if x1_min <= x1 < x1_max and y1_min <= y1 < y1_max:
            cv2.circle(canvas, (x1, y1), 3, (0, 0, 255), -1)
            
            for j in range(m_cols):
                idx2 = matches[i, j]
                if idx2 != -1:
                    x2, y2 = int(kpts2[idx2][0]), int(kpts2[idx2][1])
                    
                    # CHỈ VẼ ĐƯỜNG NỐI nếu keypoint 2 cũng nằm trong bbox của id2
                    if x2_min <= x2 < x2_max and y2_min <= y2 < y2_max:
                        pt2_on_canvas = (x2 + W1, y2)
                        cv2.line(canvas, (x1, y1), pt2_on_canvas, (0, 255, 0), 1)

    # --- 5. HIỂN THỊ ---
    plt.figure(figsize=(15, 8))
    plt.imshow(canvas)
    plt.title(f"Matches Visualization (img1_id: {id1}, img2_id: {id2})")
    plt.axis('off')
    plt.show()

def draw_custom_matches_for_gray(img1, img2, kpts1, kpts2, matches, id1=0, id2=0):
    """
    Draw matches for grayscale images.
    img1, img2: (H, W)
    """

    # Convert gray -> RGB
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    x1_min, y1_min, x1_max, y1_max = get_patch_bbox(H1, W1, id1)
    x2_min, y2_min, x2_max, y2_max = get_patch_bbox(H2, W2, id2)

    # White canvas
    max_h = max(H1, H2)
    total_w = W1 + W2
    canvas = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255

    # Paste image1
    canvas[y1_min:y1_max, x1_min:x1_max] = img1[y1_min:y1_max, x1_min:x1_max]
    cv2.rectangle(canvas, (x1_min, y1_min), (x1_max, y1_max), (0, 0, 255), 2)
    cv2.putText(canvas,
                f"img1 (id={id1})",
                (x1_min + 10, y1_min + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    # Paste image2
    canvas[y2_min:y2_max,
           W1 + x2_min:W1 + x2_max] = img2[y2_min:y2_max, x2_min:x2_max]
    cv2.rectangle(canvas,
                  (W1 + x2_min, y2_min),
                  (W1 + x2_max, y2_max),
                  (0, 0, 255),
                  2)
    cv2.putText(canvas,
                f"img2 (id={id2})",
                (W1 + x2_min + 10, y2_min + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    # Draw keypoints image2
    for pt in kpts2:
        x2, y2 = map(int, pt[:2])
        if x2_min <= x2 < x2_max and y2_min <= y2 < y2_max:
            cv2.circle(canvas, (x2 + W1, y2), 3, (255, 0, 0), -1)

    # Draw matches
    m_cols = matches.shape[1]

    for i, pt1 in enumerate(kpts1):
        x1, y1 = map(int, pt1[:2])

        if not (x1_min <= x1 < x1_max and y1_min <= y1 < y1_max):
            continue

        cv2.circle(canvas, (x1, y1), 3, (255, 0, 0), -1)

        for j in range(m_cols):
            idx2 = matches[i, j]

            if idx2 == -1:
                continue

            x2, y2 = map(int, kpts2[idx2][:2])

            if x2_min <= x2 < x2_max and y2_min <= y2 < y2_max:
                cv2.line(canvas,
                         (x1, y1),
                         (x2 + W1, y2),
                         (0, 255, 0),
                         1)

    plt.figure(figsize=(15, 8))
    plt.imshow(canvas)
    plt.title(f"Matches Visualization (img1_id={id1}, img2_id={id2})")
    plt.axis("off")
    plt.show()

@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False,
                     dict_keypoints_index_query: Path = None,
                     dict_keypoints_index_map: Path = None,
                     is_query_map_match: bool = False,
                     feature_path_raw_ref: Path = None,
                     method_baseline = False,
                     method1 = False,
                     method2 = True,
                     method3 = True):
    
    # print(f"\n---start lg")
    # print(f"\n----pairs path: {pairs_path}")
    pairs = []
    if not is_query_map_match and ('roma' in conf['model']['name'] or 'dkm' in conf['model']['name']):
        # print("-----------------------thissssssssssssssssssss-------------------")
        # print(f"-----------pair path: {pairs_path}------------")
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_ref, overwrite)
    elif 'roma' in conf['model']['name'] or 'dkm' in conf['model']['name']:
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_raw_ref, overwrite)
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    if 'roma' in conf['model']['name'] or 'dkm' in conf['model']['name']:
        match_path.parent.mkdir(exist_ok=True, parents=True)
        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        
        pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
        if len(pairs) == 0:
            logger.info('Skipping the matching.')
            return
    # match_path.parent.mkdir(exist_ok=True, parents=True)

    # assert pairs_path.exists(), pairs_path
    # pairs0 = parse_retrieval(pairs_path)
    # pairs0 = [(q, r) for q, rs in pairs0.items() for r in rs]
    
    # pairs0 = find_unique_new_pairs(pairs, None if overwrite else match_path)
    # if len(pairs0) == 0:
    #     logger.info('Skipping the matching.')
    #     return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    use_roma = False
    # print(f"feature_path_q: {feature_path_q}, feature_path_ref: {feature_path_ref}")
    if 'roma' in conf['model']['name'] or 'dkm' in conf['model']['name']:
        use_roma = True
        dataset = FeaturePairsDatasetRoMa(pairs, feature_path_q, feature_path_ref)
    else:
        dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
        writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
    # print(f"dataset len: {dataset.__len__()}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)# keep shuffle=False
    
    # loader = torch.utils.data.DataLoader(
    #     dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)# keep shuffle=False
    # matched_pairs = set() # ((img_name0, img_name1))
    # with open(path_log_file_end_matched, 'r') as f:
    #     for line in f:
    #         img_name0, img_name1 = line.strip().split(',')
    #         matched_pairs.add((img_name0, img_name1))
    if use_roma:
        limit_keypoints = conf['model']['max_keypoints']
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            count_keypoints_loss = 0           
            data0 = {k: v if str(k).startswith('image')
                        else v.to(device, non_blocking=True) for k, v in data[0].items()}
            data1 = {k: v if str(k).startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data[1].items()}
            name_img0 = data0['image_name'][0]
            name_img1 = data1['image_name'][0]
            img_path0 = data0['image_path'][0]
            img_path1 = data1['image_path'][0]
            
            # if (name_img0, name_img1) in matched_pairs:
            #     print(f"pair {name_img0}, {name_img1} already matched, skip.")
            #     continue
            # # write name_img0 and name_img1 to log file
            # with open(path_log_file_start_matched, 'a') as f:
            #     f.write(f"{img_path0}, {img_path1}\n")
            kpt0_is_max = False
            kpt1_is_max = False
            kpt0, kpt1 = data0['keypoints'].cpu().numpy(), data1['keypoints'].cpu().numpy() 
            kpt0_origin = kpt0.copy()
            kpt1_origin = kpt1.copy()
            # H, W, 3
            if len(kpt0.shape) > 3:
                kpt0 = kpt0.squeeze()
                kpt0 = np.expand_dims(kpt0, axis=0) # 1, N, 2
            if len(kpt1.shape) > 3:
                kpt1 = kpt1.squeeze()
                kpt1 = np.expand_dims(kpt1, axis=0) # 1, N, 2

            ## add logic if kpt1.shape[1] == 0 or kpt0.shape[1] == 0, then skip this pair
            if kpt0.shape[1] == 0 or kpt1.shape[1] == 0:
                # print(f"skip pair {name_img0}, {name_img1} because kpt0.shape[1] == 0 or kpt1.shape[1] == 0")
                continue
            # if is_query_map_match:
            #     if kpt0.shape[1] >= 2*limit_keypoints:
            #         kpt0_is_max = True
            #     if kpt1.shape[1] >= 2*limit_keypoints:
            #         kpt1_is_max = True
            


            if method_baseline:
                score_less_than_threshold = None
                index_nearest_2_kpts = None
                keypoints_match0_larger_than_threshold = None
                keypoint_match1_larger_than_threshold = None
                score_larger_than_threshold = None
                keypoints_match0_map_less_query_larger = None
                keypoint_match1_map_less_sp = None
                score_map_less_query_larger = None
                keypoints_match0_query_less_sp = None
                keypoint_match1_map_larger_query_less = None
                score_map_larger_query_less = None
                old_list_matches0 = []
                old_list_matching_scores0 = []


                with h5py.File(str(match_path), 'r', libver='latest') as fm:
                    pair = names_to_pair(name_img0, name_img1)
                    old_list_matches0 = list(fm[names_to_pair(name_img0, name_img1)]['matches0'][:])
                    old_list_matches0_origin = old_list_matches0.copy()
                    old_list_matching_scores0 = list(fm[names_to_pair(name_img0, name_img1)]['matching_scores0'][:])
                    old_list_matching_scores0_origin = old_list_matching_scores0.copy()
                if len(old_list_matches0[0].shape) == 0:
                    # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                    old_list_matches0 = [[int(imat)] for imat in old_list_matches0]
                    old_list_matching_scores0 = [[int(iscore)] for iscore in old_list_matching_scores0]
                else:
                    # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                    old_list_matches0 = [[int(val) for val in imat] for imat in old_list_matches0]
                    old_list_matching_scores0 = [[float(val) for val in iscore] for iscore in old_list_matching_scores0]
                # kpt0 = [(int(x), int(y)) for x, y in kpt0[0]]
                # kpt1 = [(int(x), int(y)) for x, y in kpt1[0]]
                # # if len(kpt0) > len(old_list_matches0): this is keypoints added by roma, add -1 to old_list_matches0 and 0 to old_list_matching_scores0
                # if len(kpt0) > len(old_list_matches0):
                #     for redundant in range(len(kpt0) - len(old_list_matches0)):
                #         old_list_matches0.append([-1])
                #         old_list_matching_scores0.append([0.0])
                # read dict_index_keypoints_query and dict_index_keypoints_reference
                # dict_index_keypoints_query = {}
                # dict_index_keypoints_reference = {}
                # if dict_keypoints_index_query is not None:
                #     with h5py.File(str(dict_keypoints_index_query), 'r', libver='latest') as fd:
                #         grp0 = fd[name_img0]
                #         for k, v in grp0.items():
                #             dict_index_keypoints_query[k] = int(v.__array__())
                # if dict_keypoints_index_map is not None:
                #     with h5py.File(str(dict_keypoints_index_map), 'r', libver='latest') as fd:
                #         grp1 = fd[name_img1]
                #         for k, v in grp1.items():
                #             dict_index_keypoints_reference[k] = int(v.__array__())
                # write back kpt0 and kpt1 to feature_path_q and feature_path_ref
                # kpt0 = np.array([kpt0])
                # kpt1 = np.array([kpt1])
                # print(f"kpt0.shape{kpt0.shape}, kpt1.shape{kpt1.shape}")
                # with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                #     uncertainty = 2.0*scales_0
                #     del fq[name_img0]['keypoints']
                #     fq[name_img0].create_dataset('keypoints', data=kpt0)
                #     fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
                # with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                #     uncertainty = 2.0*scales_1
                #     del fr[name_img1]['keypoints']
                #     fr[name_img1].create_dataset('keypoints', data=kpt1)
                #     fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty
                
                
                # process matches and matching score before write to file
                ## matches: [[1], [1,3,2], [4,2], ...]
                ## matching score: [[0.9], [0.94,0.92,0.99], [0.95,0.93], ...]
                max_len = max(max(len(sub) for sub in old_list_matches0), max(len(sub) for sub in old_list_matching_scores0))

                old_list_matches0_f = np.full((len(old_list_matches0), max_len), -1, dtype=np.int64)

                # Khởi tạo toàn số 0.0 cho mảng scores
                old_list_matching_scores0_f = np.zeros((len(old_list_matching_scores0), max_len), dtype=np.float32)

                # 3. Đổ dữ liệu từ list ban đầu vào ma trận NumPy
                for i, sub_list in enumerate(old_list_matches0):
                    if len(sub_list) == 0:
                        # print("sub_list is empty!")
                        raise ValueError("sub_list must not be empty")
                    old_list_matches0_f[i, :len(sub_list)] = sub_list

                for i, sub_list in enumerate(old_list_matching_scores0):
                    if len(sub_list) == 0:
                        # print("sub_list is empty!")
                        raise ValueError("sub_list must not be empty")
                    old_list_matching_scores0_f[i, :len(sub_list)] = sub_list
                # print(f"old_list_matches0_f.shape: {old_list_matches0_f.shape}")
                # print(f"old_list_matching_scores0_f.shape: {old_list_matching_scores0_f.shape}")
                # print("sub_____________looooppppppppp")

                ## write back old_list_matches0 and old_list_matching_scores0 to match_path
                # old_list_matches0_f = np.array(old_list_matches0)
                # old_list_matching_scores0_f = np.array(old_list_matching_scores0)

                with h5py.File(str(match_path), 'a', libver='latest') as fm:
                    del fm[names_to_pair(name_img0, name_img1)]['matches0']
                    del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                    fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0_f)
                    fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0_f)
                
                # ## write back dict_index_keypoints_query and dict_index_keypoints_reference to dict_keypoints_index
                # if dict_keypoints_index_query is not None:
                #     with h5py.File(str(dict_keypoints_index_query), 'a', libver='latest') as fd:
                #         del fd[name_img0]
                #         grp0 = fd.create_group(name_img0)
                #         for k, v in dict_index_keypoints_query.items():
                #             grp0.create_dataset(k, data=v)
                # if dict_keypoints_index_map is not None:
                #     with h5py.File(str(dict_keypoints_index_map), 'a', libver='latest') as fd:
                #         del fd[name_img1]
                #         grp1 = fd.create_group(name_img1)
                #         for k, v in dict_index_keypoints_reference.items():
                #             grp1.create_dataset(k, data=v)
                continue
            

            interp = 'cv2_area'
            # resize_max = conf['model']['resize_max']
            resize_max = None
            image0, scales_0 = get_image(img_path0, interp, resize_max)
            h0, w0 = image0.shape[:2]
            image1, scales_1 = get_image(img_path1, interp, resize_max)
            h1, w1 = image1.shape[:2]
            trash1 = None
            trash2 = None
            interp = 'cv2_area'
            # process split
            sub_patch_number = 0
        
            for id0 in range(5):
                for id1 in range(5):
                       
                    if id0 != 0 or id1 != 0:
                        sub_patch_number = 1
                    patch_img0 = split_image(image0, id0)
                    # print(f"id0: {id0}, id1: {id1}, kpt0 shape {kpt0.shape} kpt1 shape {kpt1.shape}")
                    apatch_kpts0 = process_existed_kpts(kpt0[0], id0, (w0, h0))
                    patch_img1 = split_image(image1, id1)
                    apatch_kpts1 = process_existed_kpts(kpt1[0], id1, (w1, h1))
                    if len(apatch_kpts0.shape) == 3 or len(apatch_kpts1.shape) == 3:
                        # print(f"kpts0_shape: {kpt0.shape}, apatch_kpts0 shape {apatch_kpts0.shape}, kpts1 shape {kpt1.shape}, apatch_kpts1 shape {apatch_kpts1.shape}")
                        raise ValueError("apatch_kpts shape should be (N, 2)")
                    patch_img0 = resize_image(patch_img0, (w0, h0), interp)
                    patch_img0 = Image.fromarray(patch_img0.astype(np.uint8))
                    patch_img1 = resize_image(patch_img1, (w1, h1), interp) #checkpoint
                    patch_img1 = Image.fromarray(patch_img1.astype(np.uint8))
                    score_less_than_threshold = None
                    index_nearest_2_kpts = None
                    keypoints_match0_larger_than_threshold = None
                    keypoint_match1_larger_than_threshold = None
                    score_larger_than_threshold = None
                    keypoints_match0_map_less_query_larger = None
                    keypoint_match1_map_less_sp = None
                    score_map_less_query_larger = None
                    keypoints_match0_query_less_sp = None
                    keypoint_match1_map_larger_query_less = None
                    score_map_larger_query_less = None
                    
                    trash1, trash2, score_less_than_threshold, index_nearest_2_kpts, \
                    keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
                    keypoints_match0_map_less_query_larger, keypoint_match1_map_less_sp, score_map_less_query_larger, \
                    keypoints_match0_query_less_sp, keypoint_match1_map_larger_query_less, score_map_larger_query_less = model([patch_img0, apatch_kpts0, patch_img1, apatch_kpts1, sub_patch_number])
                    keypoints_match0_larger_than_threshold = restore_coords(keypoints_match0_larger_than_threshold, id0, (w0, h0), device)
                    keypoint_match1_larger_than_threshold = restore_coords(keypoint_match1_larger_than_threshold, id1, (w1, h1), device)
                    keypoints_match0_map_less_query_larger = restore_coords(keypoints_match0_map_less_query_larger, id0, (w0, h0), device)
                    keypoint_match1_map_less_sp = restore_coords(keypoint_match1_map_less_sp, id1, (w1, h1), device)
                    keypoints_match0_query_less_sp = restore_coords(keypoints_match0_query_less_sp, id0, (w0, h0), device)
                    keypoint_match1_map_larger_query_less = restore_coords(keypoint_match1_map_larger_query_less, id1, (w1, h1), device)

                    
                    old_list_matches0 = []
                    old_list_matching_scores0 = []
                    with h5py.File(str(match_path), 'r', libver='latest') as fm:
                        pair = names_to_pair(name_img0, name_img1)
                        old_list_matches0 = list(fm[names_to_pair(name_img0, name_img1)]['matches0'][:])
                        old_list_matching_scores0 = list(fm[names_to_pair(name_img0, name_img1)]['matching_scores0'][:])
                    if len(old_list_matches0[0].shape) == 0:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(imat)] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[int(iscore)] for iscore in old_list_matching_scores0]
                    else:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(val) for val in imat] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[float(val) for val in iscore] for iscore in old_list_matching_scores0]
                    kpt0 = [(int(x), int(y)) for x, y in kpt0[0]]
                    kpt1 = [(int(x), int(y)) for x, y in kpt1[0]]
                    # if len(kpt0) > len(old_list_matches0): this is keypoints added by roma, add -1 to old_list_matches0 and 0 to old_list_matching_scores0
                    if len(kpt0) > len(old_list_matches0):
                        for redundant in range(len(kpt0) - len(old_list_matches0)):
                            old_list_matches0.append([-1])
                            old_list_matching_scores0.append([0.0])
                    # read dict_index_keypoints_query and dict_index_keypoints_reference
                    dict_index_keypoints_query = {}
                    dict_index_keypoints_reference = {}
                    if dict_keypoints_index_query is not None:
                        with h5py.File(str(dict_keypoints_index_query), 'r', libver='latest') as fd:
                            grp0 = fd[name_img0]
                            for k, v in grp0.items():
                                dict_index_keypoints_query[k] = int(v.__array__())
                    if dict_keypoints_index_map is not None:
                        with h5py.File(str(dict_keypoints_index_map), 'r', libver='latest') as fd:
                            grp1 = fd[name_img1]
                            for k, v in grp1.items():
                                dict_index_keypoints_reference[k] = int(v.__array__())
                    
                    if not method_baseline:
                        # processing for 2 keypoints less than threshold
                        for j in range(len(index_nearest_2_kpts)):
                            if (index_nearest_2_kpts[j][0] == -1) or (index_nearest_2_kpts[j][1] == -1):
                                continue
                            keypoint_query = kpt0[index_nearest_2_kpts[j][0]]
                            keypoint_reference = kpt1[index_nearest_2_kpts[j][1]]
                            index_matches0 = dict_index_keypoints_query[str(keypoint_query)]
                            list_score_matches0 = old_list_matching_scores0[index_matches0]
                            score_matches0_roma = score_less_than_threshold[j]
                            ### add new pair to old matches and scores
                            # if score_matches0_roma > score_matches0:  
                            if list_score_matches0[0] < 0.6 or not method3:
                                old_list_matches0[index_matches0][0] = dict_index_keypoints_reference[str(keypoint_reference)]
                                old_list_matching_scores0[index_matches0][0] = score_matches0_roma
                            else:
                                old_list_matches0[index_matches0].append(dict_index_keypoints_reference[str(keypoint_reference)])
                                old_list_matching_scores0[index_matches0].append(score_matches0_roma)
                                
                        if method2:
                            
                            # process for map less and query larger
                            # if not kpt0_is_max:
                            for j in range(score_map_less_query_larger.shape[0]):
                                new_query_keypoint = (int(keypoints_match0_map_less_query_larger[j][0]), int(keypoints_match0_map_less_query_larger[j][1]))
                                old_map_keypoint = (int(keypoint_match1_map_less_sp[j][0]), int(keypoint_match1_map_less_sp[j][1]))
                                if str(new_query_keypoint) not in dict_index_keypoints_query:
                                    old_len_kpt0 = len(kpt0)
                                    if str(old_map_keypoint) not in dict_index_keypoints_reference:
                                        count_keypoints_loss += 1
                                        continue
                                    index_old_map_keypoint = dict_index_keypoints_reference[str(old_map_keypoint)]
                                    kpt0.append(new_query_keypoint)
                                    old_list_matches0.append([index_old_map_keypoint])
                                    old_list_matching_scores0.append([score_map_less_query_larger[j]])
                                    dict_index_keypoints_query[str(new_query_keypoint)] = old_len_kpt0

                            # process for map larger and query less, only if not is query_map
                            # if not kpt1_is_max:
                            if not is_query_map_match:
                                for j in range(score_map_larger_query_less.shape[0]):
                                    new_map_keypoint = (int(keypoint_match1_map_larger_query_less[j][0]), int(keypoint_match1_map_larger_query_less[j][1]))
                                    old_query_keypoint = (int(keypoints_match0_query_less_sp[j][0]), int(keypoints_match0_query_less_sp[j][1]))
                                    if str(new_map_keypoint) not in dict_index_keypoints_reference:
                                        old_len_kpt1 = len(kpt1)
                                        if str(old_query_keypoint) not in dict_index_keypoints_query:
                                            count_keypoints_loss += 1
                                            continue
                                        index_old_query_keypoint = dict_index_keypoints_query[str(old_query_keypoint)]
                                        kpt1.append(new_map_keypoint)
                                        old_list_matches0[index_old_query_keypoint].append(old_len_kpt1)
                                        old_list_matching_scores0[index_old_query_keypoint].append(score_map_larger_query_less[j])
                                        dict_index_keypoints_reference[str(new_map_keypoint)] = old_len_kpt1

                            ## processing for 2 keypoints larger than threshold, only if not is query map
                            # if (not kpt0_is_max) and (not kpt1_is_max):
                            if not is_query_map_match:
                                for j in range(score_larger_than_threshold.shape[0]):
                                    old_len_kpt1 = len(kpt1)
                                    old_len_kpt0 = len(kpt0)
                                    new_keypoint_query = keypoints_match0_larger_than_threshold[j] ## different in float but same in int
                                    new_keypoint_query = (int(new_keypoint_query[0]), int(new_keypoint_query[1]))
                                    new_keypoint_reference = keypoint_match1_larger_than_threshold[j]
                                    new_keypoint_reference = (int(new_keypoint_reference[0]), int(new_keypoint_reference[1]))
                                    if str(new_keypoint_query) in dict_index_keypoints_query or str(new_keypoint_reference) in dict_index_keypoints_reference:
                                        # print(f'Warning: new_keypoint_query: {new_keypoint_query} already in dict_index_keypoints_query, skip adding again.')
                                        continue
                                    ## add at the end of kpt0 and kpt1
                                    kpt0.append(new_keypoint_query)
                                    kpt1.append(new_keypoint_reference)
                                    ## add to dict index_keypoint of dict_index_keypoints_query and dict_index_keypoints_reference
                                    dict_index_keypoints_query[str(new_keypoint_query)] = old_len_kpt0
                                    dict_index_keypoints_reference[str(new_keypoint_reference)] = old_len_kpt1

                                    ## add to old_list_matches0 and old_list_matching_scores0
                                    old_list_matches0.append([old_len_kpt1])
                                    old_list_matching_scores0.append([score_larger_than_threshold[j]])
                    
                    try:
                        # write back kpt0 and kpt1 to feature_path_q and feature_path_ref
                        kpt0 = np.array([kpt0])
                        kpt1 = np.array([kpt1])
                        # print(f"kpt0.shape{kpt0.shape}, kpt1.shape{kpt1.shape}")
                        with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                            uncertainty = 2.0*scales_0
                            del fq[name_img0]['keypoints']
                            fq[name_img0].create_dataset('keypoints', data=kpt0)
                            fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
                        with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                            uncertainty = 2.0*scales_1
                            del fr[name_img1]['keypoints']
                            fr[name_img1].create_dataset('keypoints', data=kpt1)
                            fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty
                        
                        
                        # process matches and matching score before write to file
                        ## matches: [[1], [1,3,2], [4,2], ...]
                        ## matching score: [[0.9], [0.94,0.92,0.99], [0.95,0.93], ...]
                        max_len = max(max(len(sub) for sub in old_list_matches0), max(len(sub) for sub in old_list_matching_scores0))

                        old_list_matches0_f = np.full((len(old_list_matches0), max_len), -1, dtype=np.int64)

                        # Khởi tạo toàn số 0.0 cho mảng scores
                        old_list_matching_scores0_f = np.zeros((len(old_list_matching_scores0), max_len), dtype=np.float32)

                        # 3. Đổ dữ liệu từ list ban đầu vào ma trận NumPy
                        for i, sub_list in enumerate(old_list_matches0):
                            if len(sub_list) == 0:
                                # print("sub_list is empty!")
                                raise ValueError("sub_list must not be empty")
                            old_list_matches0_f[i, :len(sub_list)] = sub_list

                        for i, sub_list in enumerate(old_list_matching_scores0):
                            if len(sub_list) == 0:
                                # print("sub_list is empty!")
                                raise ValueError("sub_list must not be empty")
                            old_list_matching_scores0_f[i, :len(sub_list)] = sub_list
                        # print(f"old_list_matches0_f.shape: {old_list_matches0_f.shape}")
                        # print(f"old_list_matching_scores0_f.shape: {old_list_matching_scores0_f.shape}")
                        # print("sub_____________looooppppppppp")

                        ## write back old_list_matches0 and old_list_matching_scores0 to match_path
                        # old_list_matches0_f = np.array(old_list_matches0)
                        # old_list_matching_scores0_f = np.array(old_list_matching_scores0)

                        with h5py.File(str(match_path), 'a', libver='latest') as fm:
                            del fm[names_to_pair(name_img0, name_img1)]['matches0']
                            del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                            fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0_f)
                            fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0_f)
                        
                        ## write back dict_index_keypoints_query and dict_index_keypoints_reference to dict_keypoints_index
                        if dict_keypoints_index_query is not None:
                            with h5py.File(str(dict_keypoints_index_query), 'a', libver='latest') as fd:
                                del fd[name_img0]
                                grp0 = fd.create_group(name_img0)
                                for k, v in dict_index_keypoints_query.items():
                                    grp0.create_dataset(k, data=v)
                        if dict_keypoints_index_map is not None:
                            with h5py.File(str(dict_keypoints_index_map), 'a', libver='latest') as fd:
                                del fd[name_img1]
                                grp1 = fd.create_group(name_img1)
                                for k, v in dict_index_keypoints_reference.items():
                                    grp1.create_dataset(k, data=v)
                        # # save name_img0 and name_img1 to log file
                        # with open(path_log_file_end_matched, 'a') as f:
                        #     f.write(f"{name_img0}, {name_img1}\n")
                    except Exception as e:
                        # print(f"Error occurred while processing pair {name_img0}, {name_img1}: {e}, process origin")
                        kpt0 = np.array([kpt0_origin])
                        kpt1 = np.array([kpt1_origin])
                        # print(f"kpt0.shape{kpt0.shape}, kpt1.shape{kpt1.shape}")
                        with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                            uncertainty = 2.0*scales_0
                            del fq[name_img0]['keypoints']
                            fq[name_img0].create_dataset('keypoints', data=kpt0)
                            fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
                        with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                            uncertainty = 2.0*scales_1
                            del fr[name_img1]['keypoints']
                            fr[name_img1].create_dataset('keypoints', data=kpt1)
                            fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty
                        
                        with h5py.File(str(match_path), 'a', libver='latest') as fm:
                            del fm[names_to_pair(name_img0, name_img1)]['matches0']
                            del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                            fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0_origin)
                            fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0_origin)
                        # write "error" in log start
                        with open(path_log_file_start_matched, 'a') as f:
                            f.write(f"error,{img_path0}, {img_path1}\n")
                        
                        with open(path_log_file_end_matched, 'a') as f:
                            f.write(f"{name_img0}, {name_img1}\n")
                    # # need to plot here
                    # if is_query_map_match:
                    #     if idx < 1:
                    #         draw_custom_matches(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)

                    # draw_custom_matches(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)
                    if not is_query_map_match:
                        break
                    if not method3:
                        break
                    
                if not is_query_map_match:
                    break
                if not method3:
                        break
            # print("looooppppppppppppppppppppppppp")

            # print(f"count_keypoints_loss: {count_keypoints_loss}")    

    else:
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            data = {k: v if k.startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data.items()}
            pred = model(data)
            pair = names_to_pair(*pairs[idx])
            writer_queue.put((pair, pred))
        writer_queue.join()
    logger.info('Finished exporting matches.')

@torch.no_grad()
def match_from_paths_dkm(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False,
                     dict_keypoints_index_query: Path = None,
                     dict_keypoints_index_map: Path = None,
                     is_query_map_match: bool = False,
                     feature_path_raw_ref: Path = None,
                     method_baseline = False,
                     method1 = False,
                     method2 = True,
                     method3 = True):
    
    # print(f"\n---start lg")
    # print(f"\n----pairs path: {pairs_path}")
    pairs = []
    if not is_query_map_match and 'dkm' in conf['model']['name']:
        # print("-----------------------thissssssssssssssssssss-------------------")
        # print(f"-----------pair path: {pairs_path}------------")
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_ref, overwrite)
    elif 'dkm' in conf['model']['name']:
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_raw_ref, overwrite)
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    if 'dkm' not in conf['model']['name']:
        match_path.parent.mkdir(exist_ok=True, parents=True)
        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        
        pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
        if len(pairs) == 0:
            logger.info('Skipping the matching.')
            return
    # match_path.parent.mkdir(exist_ok=True, parents=True)

    # assert pairs_path.exists(), pairs_path
    # pairs0 = parse_retrieval(pairs_path)
    # pairs0 = [(q, r) for q, rs in pairs0.items() for r in rs]
    
    # pairs0 = find_unique_new_pairs(pairs, None if overwrite else match_path)
    # if len(pairs0) == 0:
    #     logger.info('Skipping the matching.')
    #     return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    use_roma = False
    # print(f"feature_path_q: {feature_path_q}, feature_path_ref: {feature_path_ref}")
    if 'dkm' in conf['model']['name']:
        use_roma = True
        dataset = FeaturePairsDatasetRoMa(pairs, feature_path_q, feature_path_ref)
    else:
        dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
        writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
    # print(f"dataset len: {dataset.__len__()}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)# keep shuffle=False
    
    # loader = torch.utils.data.DataLoader(
    #     dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)# keep shuffle=False
    if use_roma:
        limit_keypoints = conf['model']['max_keypoints']
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            count_keypoints_loss = 0           
            data0 = {k: v if str(k).startswith('image')
                        else v.to(device, non_blocking=True) for k, v in data[0].items()}
            data1 = {k: v if str(k).startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data[1].items()}
            name_img0 = data0['image_name'][0]
            name_img1 = data1['image_name'][0]
            kpt0_is_max = False
            kpt1_is_max = False
            kpt0, kpt1 = data0['keypoints'].cpu().numpy(), data1['keypoints'].cpu().numpy() 
            # H, W, 3
            if len(kpt0.shape) > 3:
                kpt0 = kpt0.squeeze()
                kpt0 = np.expand_dims(kpt0, axis=0) # 1, N, 2
            if len(kpt1.shape) > 3:
                kpt1 = kpt1.squeeze()
                kpt1 = np.expand_dims(kpt1, axis=0) # 1, N, 2
            # if is_query_map_match:
            #     if kpt0.shape[1] >= 2*limit_keypoints:
            #         kpt0_is_max = True
            #     if kpt1.shape[1] >= 2*limit_keypoints:
            #         kpt1_is_max = True
            img_path0 = data0['image_path'][0]
            img_path1 = data1['image_path'][0]

            interp = 'cv2_area'
            # resize_max = conf['model']['resize_max']
            resize_max = None
            image0, scales_0 = get_image(img_path0, interp, resize_max)
            h0, w0 = image0.shape[:2]
            image1, scales_1 = get_image(img_path1, interp, resize_max)
            h1, w1 = image1.shape[:2]
            trash1 = None
            trash2 = None
            interp = 'cv2_area'
            # process split
            sub_patch_number = 0
            for id0 in range(5):
                for id1 in range(5):
                       
                    if id0 != 0 or id1 != 0:
                        sub_patch_number = 1
                    patch_img0 = split_image(image0, id0)
                    # print(f"id0: {id0}, id1: {id1}, kpt0 shape {kpt0.shape} kpt1 shape {kpt1.shape}")
                    apatch_kpts0 = process_existed_kpts(kpt0[0], id0, (w0, h0))
                    patch_img1 = split_image(image1, id1)
                    apatch_kpts1 = process_existed_kpts(kpt1[0], id1, (w1, h1))
                    if len(apatch_kpts0.shape) == 3 or len(apatch_kpts1.shape) == 3:
                        # print(f"kpts0_shape: {kpt0.shape}, apatch_kpts0 shape {apatch_kpts0.shape}, kpts1 shape {kpt1.shape}, apatch_kpts1 shape {apatch_kpts1.shape}")
                        raise ValueError("apatch_kpts shape should be (N, 2)")
                    patch_img0 = resize_image(patch_img0, (w0, h0), interp)
                    patch_img0 = Image.fromarray(patch_img0.astype(np.uint8))
                    patch_img1 = resize_image(patch_img1, (w1, h1), interp) #checkpoint
                    patch_img1 = Image.fromarray(patch_img1.astype(np.uint8))
                    score_less_than_threshold = None
                    index_nearest_2_kpts = None
                    keypoints_match0_larger_than_threshold = None
                    keypoint_match1_larger_than_threshold = None
                    score_larger_than_threshold = None
                    keypoints_match0_map_less_query_larger = None
                    keypoint_match1_map_less_sp = None
                    score_map_less_query_larger = None
                    keypoints_match0_query_less_sp = None
                    keypoint_match1_map_larger_query_less = None
                    score_map_larger_query_less = None
                    
                    trash1, trash2, score_less_than_threshold, index_nearest_2_kpts, \
                    keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
                    keypoints_match0_map_less_query_larger, keypoint_match1_map_less_sp, score_map_less_query_larger, \
                    keypoints_match0_query_less_sp, keypoint_match1_map_larger_query_less, score_map_larger_query_less = model([patch_img0, apatch_kpts0, patch_img1, apatch_kpts1, sub_patch_number])
                    keypoints_match0_larger_than_threshold = restore_coords(keypoints_match0_larger_than_threshold, id0, (w0, h0), device)
                    keypoint_match1_larger_than_threshold = restore_coords(keypoint_match1_larger_than_threshold, id1, (w1, h1), device)
                    keypoints_match0_map_less_query_larger = restore_coords(keypoints_match0_map_less_query_larger, id0, (w0, h0), device)
                    keypoint_match1_map_less_sp = restore_coords(keypoint_match1_map_less_sp, id1, (w1, h1), device)
                    keypoints_match0_query_less_sp = restore_coords(keypoints_match0_query_less_sp, id0, (w0, h0), device)
                    keypoint_match1_map_larger_query_less = restore_coords(keypoint_match1_map_larger_query_less, id1, (w1, h1), device)

                    
                    old_list_matches0 = []
                    old_list_matching_scores0 = []
                    with h5py.File(str(match_path), 'r', libver='latest') as fm:
                        pair = names_to_pair(name_img0, name_img1)
                        old_list_matches0 = list(fm[names_to_pair(name_img0, name_img1)]['matches0'][:])
                        old_list_matching_scores0 = list(fm[names_to_pair(name_img0, name_img1)]['matching_scores0'][:])
                    if len(old_list_matches0[0].shape) == 0:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(imat)] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[int(iscore)] for iscore in old_list_matching_scores0]
                    else:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(val) for val in imat] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[float(val) for val in iscore] for iscore in old_list_matching_scores0]
                    kpt0 = [(int(x), int(y)) for x, y in kpt0[0]]
                    kpt1 = [(int(x), int(y)) for x, y in kpt1[0]]
                    # if len(kpt0) > len(old_list_matches0): this is keypoints added by roma, add -1 to old_list_matches0 and 0 to old_list_matching_scores0
                    if len(kpt0) > len(old_list_matches0):
                        for redundant in range(len(kpt0) - len(old_list_matches0)):
                            old_list_matches0.append([-1])
                            old_list_matching_scores0.append([0.0])
                    # read dict_index_keypoints_query and dict_index_keypoints_reference
                    dict_index_keypoints_query = {}
                    dict_index_keypoints_reference = {}
                    if dict_keypoints_index_query is not None:
                        with h5py.File(str(dict_keypoints_index_query), 'r', libver='latest') as fd:
                            grp0 = fd[name_img0]
                            for k, v in grp0.items():
                                dict_index_keypoints_query[k] = int(v.__array__())
                    if dict_keypoints_index_map is not None:
                        with h5py.File(str(dict_keypoints_index_map), 'r', libver='latest') as fd:
                            grp1 = fd[name_img1]
                            for k, v in grp1.items():
                                dict_index_keypoints_reference[k] = int(v.__array__())
                    
                    if not method_baseline:
                        # processing for 2 keypoints less than threshold
                        for j in range(len(index_nearest_2_kpts)):
                            if (index_nearest_2_kpts[j][0] == -1) or (index_nearest_2_kpts[j][1] == -1):
                                continue
                            keypoint_query = kpt0[index_nearest_2_kpts[j][0]]
                            keypoint_reference = kpt1[index_nearest_2_kpts[j][1]]
                            index_matches0 = dict_index_keypoints_query[str(keypoint_query)]
                            list_score_matches0 = old_list_matching_scores0[index_matches0]
                            score_matches0_roma = score_less_than_threshold[j]
                            ### add new pair to old matches and scores
                            # if score_matches0_roma > score_matches0:  
                            if list_score_matches0[0] < 0.6 or not method3:
                                old_list_matches0[index_matches0][0] = dict_index_keypoints_reference[str(keypoint_reference)]
                                old_list_matching_scores0[index_matches0][0] = score_matches0_roma
                            else:
                                old_list_matches0[index_matches0].append(dict_index_keypoints_reference[str(keypoint_reference)])
                                old_list_matching_scores0[index_matches0].append(score_matches0_roma)
                                
                        if method2:
                            
                            # process for map less and query larger
                            # if not kpt0_is_max:
                            for j in range(score_map_less_query_larger.shape[0]):
                                new_query_keypoint = (int(keypoints_match0_map_less_query_larger[j][0]), int(keypoints_match0_map_less_query_larger[j][1]))
                                old_map_keypoint = (int(keypoint_match1_map_less_sp[j][0]), int(keypoint_match1_map_less_sp[j][1]))
                                if str(new_query_keypoint) not in dict_index_keypoints_query:
                                    old_len_kpt0 = len(kpt0)
                                    if str(old_map_keypoint) not in dict_index_keypoints_reference:
                                        count_keypoints_loss += 1
                                        continue
                                    index_old_map_keypoint = dict_index_keypoints_reference[str(old_map_keypoint)]
                                    kpt0.append(new_query_keypoint)
                                    old_list_matches0.append([index_old_map_keypoint])
                                    old_list_matching_scores0.append([score_map_less_query_larger[j]])
                                    dict_index_keypoints_query[str(new_query_keypoint)] = old_len_kpt0

                            # process for map larger and query less, only if not is query_map
                            # if not kpt1_is_max:
                            if not is_query_map_match:
                                for j in range(score_map_larger_query_less.shape[0]):
                                    new_map_keypoint = (int(keypoint_match1_map_larger_query_less[j][0]), int(keypoint_match1_map_larger_query_less[j][1]))
                                    old_query_keypoint = (int(keypoints_match0_query_less_sp[j][0]), int(keypoints_match0_query_less_sp[j][1]))
                                    if str(new_map_keypoint) not in dict_index_keypoints_reference:
                                        old_len_kpt1 = len(kpt1)
                                        if str(old_query_keypoint) not in dict_index_keypoints_query:
                                            count_keypoints_loss += 1
                                            continue
                                        index_old_query_keypoint = dict_index_keypoints_query[str(old_query_keypoint)]
                                        kpt1.append(new_map_keypoint)
                                        old_list_matches0[index_old_query_keypoint].append(old_len_kpt1)
                                        old_list_matching_scores0[index_old_query_keypoint].append(score_map_larger_query_less[j])
                                        dict_index_keypoints_reference[str(new_map_keypoint)] = old_len_kpt1

                            ## processing for 2 keypoints larger than threshold, only if not is query map
                            # if (not kpt0_is_max) and (not kpt1_is_max):
                            if not is_query_map_match:
                                for j in range(score_larger_than_threshold.shape[0]):
                                    old_len_kpt1 = len(kpt1)
                                    old_len_kpt0 = len(kpt0)
                                    new_keypoint_query = keypoints_match0_larger_than_threshold[j] ## different in float but same in int
                                    new_keypoint_query = (int(new_keypoint_query[0]), int(new_keypoint_query[1]))
                                    new_keypoint_reference = keypoint_match1_larger_than_threshold[j]
                                    new_keypoint_reference = (int(new_keypoint_reference[0]), int(new_keypoint_reference[1]))
                                    if str(new_keypoint_query) in dict_index_keypoints_query or str(new_keypoint_reference) in dict_index_keypoints_reference:
                                        # print(f'Warning: new_keypoint_query: {new_keypoint_query} already in dict_index_keypoints_query, skip adding again.')
                                        continue
                                    ## add at the end of kpt0 and kpt1
                                    kpt0.append(new_keypoint_query)
                                    kpt1.append(new_keypoint_reference)
                                    ## add to dict index_keypoint of dict_index_keypoints_query and dict_index_keypoints_reference
                                    dict_index_keypoints_query[str(new_keypoint_query)] = old_len_kpt0
                                    dict_index_keypoints_reference[str(new_keypoint_reference)] = old_len_kpt1

                                    ## add to old_list_matches0 and old_list_matching_scores0
                                    old_list_matches0.append([old_len_kpt1])
                                    old_list_matching_scores0.append([score_larger_than_threshold[j]])
                        
                    # write back kpt0 and kpt1 to feature_path_q and feature_path_ref
                    kpt0 = np.array([kpt0])
                    kpt1 = np.array([kpt1])
                    # print(f"kpt0.shape{kpt0.shape}, kpt1.shape{kpt1.shape}")
                    with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                        uncertainty = 2.0*scales_0
                        del fq[name_img0]['keypoints']
                        fq[name_img0].create_dataset('keypoints', data=kpt0)
                        fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
                    with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                        uncertainty = 2.0*scales_1
                        del fr[name_img1]['keypoints']
                        fr[name_img1].create_dataset('keypoints', data=kpt1)
                        fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty
                    
                    
                    # process matches and matching score before write to file
                    ## matches: [[1], [1,3,2], [4,2], ...]
                    ## matching score: [[0.9], [0.94,0.92,0.99], [0.95,0.93], ...]
                    max_len = max(max(len(sub) for sub in old_list_matches0), max(len(sub) for sub in old_list_matching_scores0))

                    old_list_matches0_f = np.full((len(old_list_matches0), max_len), -1, dtype=np.int64)

                    # Khởi tạo toàn số 0.0 cho mảng scores
                    old_list_matching_scores0_f = np.zeros((len(old_list_matching_scores0), max_len), dtype=np.float32)

                    # 3. Đổ dữ liệu từ list ban đầu vào ma trận NumPy
                    for i, sub_list in enumerate(old_list_matches0):
                        if len(sub_list) == 0:
                            # print("sub_list is empty!")
                            raise ValueError("sub_list must not be empty")
                        old_list_matches0_f[i, :len(sub_list)] = sub_list

                    for i, sub_list in enumerate(old_list_matching_scores0):
                        if len(sub_list) == 0:
                            # print("sub_list is empty!")
                            raise ValueError("sub_list must not be empty")
                        old_list_matching_scores0_f[i, :len(sub_list)] = sub_list
                    # print(f"old_list_matches0_f.shape: {old_list_matches0_f.shape}")
                    # print(f"old_list_matching_scores0_f.shape: {old_list_matching_scores0_f.shape}")
                    # print("sub_____________looooppppppppp")

                    ## write back old_list_matches0 and old_list_matching_scores0 to match_path
                    # old_list_matches0_f = np.array(old_list_matches0)
                    # old_list_matching_scores0_f = np.array(old_list_matching_scores0)

                    with h5py.File(str(match_path), 'a', libver='latest') as fm:
                        del fm[names_to_pair(name_img0, name_img1)]['matches0']
                        del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                        fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0_f)
                        fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0_f)
                    
                    ## write back dict_index_keypoints_query and dict_index_keypoints_reference to dict_keypoints_index
                    if dict_keypoints_index_query is not None:
                        with h5py.File(str(dict_keypoints_index_query), 'a', libver='latest') as fd:
                            del fd[name_img0]
                            grp0 = fd.create_group(name_img0)
                            for k, v in dict_index_keypoints_query.items():
                                grp0.create_dataset(k, data=v)
                    if dict_keypoints_index_map is not None:
                        with h5py.File(str(dict_keypoints_index_map), 'a', libver='latest') as fd:
                            del fd[name_img1]
                            grp1 = fd.create_group(name_img1)
                            for k, v in dict_index_keypoints_reference.items():
                                grp1.create_dataset(k, data=v)
                    # # need to plot here
                    # if is_query_map_match:
                    #     if idx < 1:
                    #         draw_custom_matches(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)
                    # draw_custom_matches(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)
                    if not is_query_map_match:
                        break
                    if not method3:
                        break
                if not is_query_map_match:
                    break
                if not method3:
                        break
            # print("looooppppppppppppppppppppppppp")

            # print(f"count_keypoints_loss: {count_keypoints_loss}")    

    else:
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            data = {k: v if k.startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data.items()}
            pred = model(data)
            pair = names_to_pair(*pairs[idx])
            writer_queue.put((pair, pred))
        writer_queue.join()
    logger.info('Finished exporting matches.')

@torch.no_grad()
def match_from_paths_loftr(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False,
                     dict_keypoints_index_query: Path = None,
                     dict_keypoints_index_map: Path = None,
                     is_query_map_match: bool = False,
                     feature_path_raw_ref: Path = None,
                     method_baseline: bool = False,
                     method1: bool = False,
                     method2: bool = True,
                     method3: bool = True):
    pairs = []
    if not is_query_map_match and 'loftr' in conf['model']['name']:
        # print("------start map map matches with loftr----------------")
        # print(f"-----------pair path: {pairs_path}------------")
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_ref, overwrite)
    elif 'loftr' in conf['model']['name']:
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_raw_ref, overwrite)
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    if 'loftr' not in conf['model']['name']:
        match_path.parent.mkdir(exist_ok=True, parents=True)
        assert pairs_path.exists(), pairs_path
        pairs = parse_retrieval(pairs_path)
        pairs = [(q, r) for q, rs in pairs.items() for r in rs]
        
        pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
        if len(pairs) == 0:
            logger.info('Skipping the matching.')
            return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    use_loftr = False
    # print(f"feature_path_q: {feature_path_q}, feature_path_ref: {feature_path_ref}")
    if 'loftr' in conf['model']['name']:
        use_loftr = True
        ##### checkpointtttttttttttttttttt
        dataset = FeaturePairsDatasetRoMa(pairs, feature_path_q, feature_path_ref)
    else:
        dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
        writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
    # print(f"dataset len: {dataset.__len__()}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)# keep shuffle=False
    
    # loader = torch.utils.data.DataLoader(
    #     dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)# keep shuffle=False
    if use_loftr:
        count_miss = 0
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            count_keypoints_loss = 0           
            data0 = {k: v if str(k).startswith('image')
                        else v.to(device, non_blocking=True) for k, v in data[0].items()}
            data1 = {k: v if str(k).startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data[1].items()}
            name_img0 = data0['image_name'][0]
            name_img1 = data1['image_name'][0]
            kpt0, kpt1 = data0['keypoints'].cpu().numpy(), data1['keypoints'].cpu().numpy() # keypoints đang bị scale sai
            # H, W, 3
            if len(kpt0.shape) > 3:
                kpt0 = kpt0.squeeze()
                kpt0 = np.expand_dims(kpt0, axis=0) # 1, N, 2
            if len(kpt1.shape) > 3:
                kpt1 = kpt1.squeeze()
                kpt1 = np.expand_dims(kpt1, axis=0) # 1, N, 2
            

            img_path0 = data0['image_path'][0]
            img_path1 = data1['image_path'][0]

            interp = 'cv2_area'
            resize_max = conf['model']['resize_max']
            # resize_max = None
            image0, scales_0 = get_image(img_path0, interp, resize_max, grayscale=True)

            h0, w0 = image0.shape[:2] # resized
            # keypoints resized
            # print(f"shape keypoints before resize: {kpt0.shape}, scales_0: {scales_0}")
            kpt0 = kpt0 / scales_0
            # print(f"shape keypoints after resize: {kpt0.shape}")
            # int kpt0
            # print(f"shape keypoints before int: {kpt0.shape}, scales_0: {scales_0}")
            kpt0 = np.array([[[int(x), int(y)] for x, y in kpt0[0]]])
            # print(f"shape keypoints after int: {kpt0.shape}")
            # h0_origin, w0_origin = h0*scales_0, w0*scales_0 # original
            image1, scales_1 = get_image(img_path1, interp, resize_max, grayscale=True)
            h1, w1 = image1.shape[:2]
            # keypoints resized
            kpt1 = kpt1 / scales_1
            # int kpt1
            kpt1 = np.array([[[int(x), int(y)] for x, y in kpt1[0]]])
            # h1_origin, w1_origin = h1*scales_1, w1*scales_1 # original
            # if idx == 0:
            #     print(f"image_size0: ({w0}, {h0}), image_size1: ({w1}, {h1})") # normal  true
            # process split
            sub_patch_number = 0
            for id0 in range(5):
                for id1 in range(5):
                       
                    if id0 != 0 or id1 != 0:
                        sub_patch_number = 1
                    patch_img0 = split_image(image0, id0)
                    # print(f"id0: {id0}, id1: {id1}, kpt0 shape {kpt0.shape} kpt1 shape {kpt1.shape}")
                    apatch_kpts0 = process_existed_kpts(kpt0[0], id0, (w0, h0))
                    patch_img1 = split_image(image1, id1)
                    apatch_kpts1 = process_existed_kpts(kpt1[0], id1, (w1, h1))
                    if len(apatch_kpts0.shape) == 3 or len(apatch_kpts1.shape) == 3:
                        # print(f"kpts0_shape: {kpt0.shape}, apatch_kpts0 shape {apatch_kpts0.shape}, kpts1 shape {kpt1.shape}, apatch_kpts1 shape {apatch_kpts1.shape}")
                        raise ValueError("apatch_kpts shape should be (N, 2)")
                    patch_img0 = resize_image(patch_img0, (w0, h0), interp)
                    patch_img0 = torch.from_numpy(patch_img0/255).float()
                    patch_img0 = patch_img0.unsqueeze(0).unsqueeze(0).to(device)
                    patch_img1 = resize_image(patch_img1, (w1, h1), interp) #checkpoint
                    patch_img1 = torch.from_numpy(patch_img1/255).float()
                    patch_img1 = patch_img1.unsqueeze(0).unsqueeze(0).to(device)
                    # print(f"patch_img0 size: {patch_img0.size()}, patch_img1 size: {patch_img1.size()}")
                    score_less_than_threshold = None
                    index_nearest_2_kpts = None
                    keypoints_match0_larger_than_threshold = None
                    keypoint_match1_larger_than_threshold = None
                    score_larger_than_threshold = None
                    keypoints_match0_map_less_query_larger = None
                    keypoint_match1_map_less_sp = None
                    score_map_less_query_larger = None
                    keypoints_match0_query_less_sp = None
                    keypoint_match1_map_larger_query_less = None
                    score_map_larger_query_less = None
                    
                    data_input = {
                    "image0": patch_img0,
                    "keypoints_orgin0": apatch_kpts0,
                    "image1": patch_img1,
                    "keypoints_orgin1": apatch_kpts1,
                    # "sub_patch_number": sub_patch_number
                    }

                    # print(f"data input key: {data_input.keys()}")
                    trash1, trash2, score_less_than_threshold, index_nearest_2_kpts, \
                    keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
                    keypoints_match0_map_less_query_larger, keypoint_match1_map_less_sp, score_map_less_query_larger, \
                    keypoints_match0_query_less_sp, keypoint_match1_map_larger_query_less, score_map_larger_query_less = model(data_input)
                    keypoints_match0_larger_than_threshold = restore_coords(keypoints_match0_larger_than_threshold, id0, (w0, h0), device)
                    keypoint_match1_larger_than_threshold = restore_coords(keypoint_match1_larger_than_threshold, id1, (w1, h1), device)
                    keypoints_match0_map_less_query_larger = restore_coords(keypoints_match0_map_less_query_larger, id0, (w0, h0), device)
                    keypoint_match1_map_less_sp = restore_coords(keypoint_match1_map_less_sp, id1, (w1, h1), device)
                    keypoints_match0_query_less_sp = restore_coords(keypoints_match0_query_less_sp, id0, (w0, h0), device)
                    keypoint_match1_map_larger_query_less = restore_coords(keypoint_match1_map_larger_query_less, id1, (w1, h1), device)

                    
                    old_list_matches0 = []
                    old_list_matching_scores0 = []
                    with h5py.File(str(match_path), 'r', libver='latest') as fm:
                        pair = names_to_pair(name_img0, name_img1)
                        old_list_matches0 = list(fm[names_to_pair(name_img0, name_img1)]['matches0'][:])
                        old_list_matching_scores0 = list(fm[names_to_pair(name_img0, name_img1)]['matching_scores0'][:])
                    if len(old_list_matches0[0].shape) == 0:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(imat)] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[int(iscore)] for iscore in old_list_matching_scores0]
                    else:
                        # print(f"len: {len(old_list_matches0)}, shape0: {old_list_matches0[0].shape}")
                        old_list_matches0 = [[int(val) for val in imat] for imat in old_list_matches0]
                        old_list_matching_scores0 = [[float(val) for val in iscore] for iscore in old_list_matching_scores0]
                    kpt0 = [(int(x), int(y)) for x, y in kpt0[0]]
                    kpt1 = [(int(x), int(y)) for x, y in kpt1[0]]
                    # if len(kpt0) > len(old_list_matches0): this is keypoints added by roma, add -1 to old_list_matches0 and 0 to old_list_matching_scores0
                    if len(kpt0) > len(old_list_matches0):
                        for redundant in range(len(kpt0) - len(old_list_matches0)):
                            old_list_matches0.append([-1])
                            old_list_matching_scores0.append([0.0])
                    # read dict_index_keypoints_query and dict_index_keypoints_reference
                    dict_index_keypoints_query = {}
                    dict_index_keypoints_reference = {}
                    if dict_keypoints_index_query is not None:
                        with h5py.File(str(dict_keypoints_index_query), 'r', libver='latest') as fd:
                            grp0 = fd[name_img0]
                            for k, v in grp0.items():
                                dict_index_keypoints_query[k] = int(v.__array__())
                    if dict_keypoints_index_map is not None:
                        with h5py.File(str(dict_keypoints_index_map), 'r', libver='latest') as fd:
                            grp1 = fd[name_img1]
                            for k, v in grp1.items():
                                dict_index_keypoints_reference[k] = int(v.__array__())
                    
                    if not method_baseline:
                        try:
                            # processing for 2 keypoints less than threshold
                            for j in range(len(index_nearest_2_kpts)):
                                if (index_nearest_2_kpts[j][0] == -1) or (index_nearest_2_kpts[j][1] == -1):
                                    continue
                                keypoint_query = kpt0[index_nearest_2_kpts[j][0]]
                                keypoint_reference = kpt1[index_nearest_2_kpts[j][1]]
                                index_matches0 = dict_index_keypoints_query[str(keypoint_query)]
                                list_score_matches0 = old_list_matching_scores0[index_matches0]
                                score_matches0_roma = score_less_than_threshold[j]
                                ### add new pair to old matches and scores
                                # if score_matches0_roma > score_matches0:  
                                if list_score_matches0[0] < 0.6:
                                    old_list_matches0[index_matches0][0] = dict_index_keypoints_reference[str(keypoint_reference)]
                                    old_list_matching_scores0[index_matches0][0] = score_matches0_roma
                                else:
                                    if not method3:
                                        pass
                                    else:
                                        old_list_matches0[index_matches0].append(dict_index_keypoints_reference[str(keypoint_reference)])
                                        old_list_matching_scores0[index_matches0].append(score_matches0_roma)
                            if method2:
                                
                                # process for map less and query larger
                                # if not kpt0_is_max:
                                for j in range(score_map_less_query_larger.shape[0]):
                                    new_query_keypoint = (int(keypoints_match0_map_less_query_larger[j][0]), int(keypoints_match0_map_less_query_larger[j][1]))
                                    old_map_keypoint = (int(keypoint_match1_map_less_sp[j][0]), int(keypoint_match1_map_less_sp[j][1]))
                                    if str(new_query_keypoint) not in dict_index_keypoints_query:
                                        old_len_kpt0 = len(kpt0)
                                        if str(old_map_keypoint) not in dict_index_keypoints_reference:
                                            count_keypoints_loss += 1
                                            continue
                                        index_old_map_keypoint = dict_index_keypoints_reference[str(old_map_keypoint)]
                                        kpt0.append(new_query_keypoint)
                                        old_list_matches0.append([index_old_map_keypoint])
                                        old_list_matching_scores0.append([score_map_less_query_larger[j]])
                                        dict_index_keypoints_query[str(new_query_keypoint)] = old_len_kpt0

                                # process for map larger and query less, only if not is query_map
                                # if not kpt1_is_max:
                                if not is_query_map_match:
                                    for j in range(score_map_larger_query_less.shape[0]):
                                        new_map_keypoint = (int(keypoint_match1_map_larger_query_less[j][0]), int(keypoint_match1_map_larger_query_less[j][1]))
                                        old_query_keypoint = (int(keypoints_match0_query_less_sp[j][0]), int(keypoints_match0_query_less_sp[j][1]))
                                        if str(new_map_keypoint) not in dict_index_keypoints_reference:
                                            old_len_kpt1 = len(kpt1)
                                            if str(old_query_keypoint) not in dict_index_keypoints_query:
                                                count_keypoints_loss += 1
                                                continue
                                            index_old_query_keypoint = dict_index_keypoints_query[str(old_query_keypoint)]
                                            kpt1.append(new_map_keypoint)
                                            old_list_matches0[index_old_query_keypoint].append(old_len_kpt1)
                                            old_list_matching_scores0[index_old_query_keypoint].append(score_map_larger_query_less[j])
                                            dict_index_keypoints_reference[str(new_map_keypoint)] = old_len_kpt1

                                ## processing for 2 keypoints larger than threshold, only if not is query map
                                # if (not kpt0_is_max) and (not kpt1_is_max):
                                if not is_query_map_match:
                                    for j in range(score_larger_than_threshold.shape[0]):
                                        old_len_kpt1 = len(kpt1)
                                        old_len_kpt0 = len(kpt0)
                                        new_keypoint_query = keypoints_match0_larger_than_threshold[j] ## different in float but same in int
                                        new_keypoint_query = (int(new_keypoint_query[0]), int(new_keypoint_query[1]))
                                        new_keypoint_reference = keypoint_match1_larger_than_threshold[j]
                                        new_keypoint_reference = (int(new_keypoint_reference[0]), int(new_keypoint_reference[1]))
                                        if str(new_keypoint_query) in dict_index_keypoints_query or str(new_keypoint_reference) in dict_index_keypoints_reference:
                                            # print(f'Warning: new_keypoint_query: {new_keypoint_query} already in dict_index_keypoints_query, skip adding again.')
                                            continue
                                        ## add at the end of kpt0 and kpt1
                                        kpt0.append(new_keypoint_query)
                                        kpt1.append(new_keypoint_reference)
                                        ## add to dict index_keypoint of dict_index_keypoints_query and dict_index_keypoints_reference
                                        dict_index_keypoints_query[str(new_keypoint_query)] = old_len_kpt0
                                        dict_index_keypoints_reference[str(new_keypoint_reference)] = old_len_kpt1

                                        ## add to old_list_matches0 and old_list_matching_scores0
                                        old_list_matches0.append([old_len_kpt1])
                                        old_list_matching_scores0.append([score_larger_than_threshold[j]])
                        except Exception as e:
                            count_miss += 1
                            # print(f"missing process {count_miss} patch, pairs. Lỗi chi tiết: {e}")
                        
                    # write back kpt0 and kpt1 to feature_path_q and feature_path_ref
                    # convert resized kpt to original kpt
                    kpt0 = np.array([kpt0])
                    kpt1 = np.array([kpt1])
                    kpt0 = kpt0 * scales_0
                    kpt1 = kpt1 * scales_1
                    # int kpt0 and kpt1
                    kpt0 = np.array([[[int(x), int(y)] for x, y in kpt0[0]]])
                    kpt1 = np.array([[[int(x), int(y)] for x, y in kpt1[0]]])
                    # print(f"kpt0.shape{kpt0.shape}, kpt1.shape{kpt1.shape}")
                    with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                        uncertainty = 2.0*scales_0
                        del fq[name_img0]['keypoints']
                        fq[name_img0].create_dataset('keypoints', data=kpt0)
                        fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
                    with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                        uncertainty = 2.0*scales_1
                        del fr[name_img1]['keypoints']
                        fr[name_img1].create_dataset('keypoints', data=kpt1)
                        fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty
                    
                    # re convert kpt to resized
                    kpt0 = kpt0 / scales_0
                    kpt1 = kpt1 / scales_1
                    # int kpt0 and kpt1
                    kpt0 = np.array([[[int(x), int(y)] for x, y in kpt0[0]]])
                    kpt1 = np.array([[[int(x), int(y)] for x, y in kpt1[0]]])
                    # process matches and matching score before write to file
                    ## matches: [[1], [1,3,2], [4,2], ...]
                    ## matching score: [[0.9], [0.94,0.92,0.99], [0.95,0.93], ...]
                    max_len = max(max(len(sub) for sub in old_list_matches0), max(len(sub) for sub in old_list_matching_scores0))

                    old_list_matches0_f = np.full((len(old_list_matches0), max_len), -1, dtype=np.int64)

                    # Khởi tạo toàn số 0.0 cho mảng scores
                    old_list_matching_scores0_f = np.zeros((len(old_list_matching_scores0), max_len), dtype=np.float32)

                    # 3. Đổ dữ liệu từ list ban đầu vào ma trận NumPy
                    for i, sub_list in enumerate(old_list_matches0):
                        if len(sub_list) == 0:
                            # print("sub_list is empty!")
                            raise ValueError("sub_list must not be empty")
                        old_list_matches0_f[i, :len(sub_list)] = sub_list

                    for i, sub_list in enumerate(old_list_matching_scores0):
                        if len(sub_list) == 0:
                            # print("sub_list is empty!")
                            raise ValueError("sub_list must not be empty")
                        old_list_matching_scores0_f[i, :len(sub_list)] = sub_list
                    # print(f"old_list_matches0_f.shape: {old_list_matches0_f.shape}")
                    # print(f"old_list_matching_scores0_f.shape: {old_list_matching_scores0_f.shape}")
                    # print("sub_____________looooppppppppp")

                    ## write back old_list_matches0 and old_list_matching_scores0 to match_path
                    # old_list_matches0_f = np.array(old_list_matches0)
                    # old_list_matching_scores0_f = np.array(old_list_matching_scores0)

                    with h5py.File(str(match_path), 'a', libver='latest') as fm:
                        del fm[names_to_pair(name_img0, name_img1)]['matches0']
                        del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                        fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0_f)
                        fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0_f)
                    
                    ## write back dict_index_keypoints_query and dict_index_keypoints_reference to dict_keypoints_index
                    if dict_keypoints_index_query is not None:
                        with h5py.File(str(dict_keypoints_index_query), 'a', libver='latest') as fd:
                            del fd[name_img0]
                            grp0 = fd.create_group(name_img0)
                            for k, v in dict_index_keypoints_query.items():
                                grp0.create_dataset(k, data=v)
                    if dict_keypoints_index_map is not None:
                        with h5py.File(str(dict_keypoints_index_map), 'a', libver='latest') as fd:
                            del fd[name_img1]
                            grp1 = fd.create_group(name_img1)
                            for k, v in dict_index_keypoints_reference.items():
                                grp1.create_dataset(k, data=v)
                    # # need to plot here
                    # if is_query_map_match:
                    #     if idx < 1:
                    #         draw_custom_matches(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)
                    draw_custom_matches_for_gray(image0, image1, kpt0[0], kpt1[0], old_list_matches0_f, id0, id1)
                    if not is_query_map_match:
                        break
                    if not method3:
                        break
                if not is_query_map_match:
                    break
                if not method3:
                        break
            # print("looooppppppppppppppppppppppppp")

            # print(f"count_keypoints_loss: {count_keypoints_loss}") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)