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
            'max_keypoints': 4096,
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
    }
}

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
         feature_path_raw_ref: Optional[Path] = None):

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
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, dict_keypoints_index_query, dict_keypoints_index_map, is_query_map_match=is_query_map_match, feature_path_raw_ref=feature_path_raw_ref)

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

def get_image(image_path, interp, resize_max = None):
    image = read_image(image_path)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    # image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    if resize_max is None:
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
    
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

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

        img_path0 = data0['image_path'][0]
        img_path1 = data1['image_path'][0]

        interp = 'cv2_area'
        resize_max = conf['model2']['preprocessing']['resize_max']
        image0, _ = get_image(img_path0, interp, resize_max)
        image1, _ = get_image(img_path1, interp, resize_max)

        image0 = Image.fromarray(image0.astype(np.uint8))
        image1 = Image.fromarray(image1.astype(np.uint8))

        # print(f'keypoint_matches0_roma: {keypoint_matches0_roma.shape}, keypoint_matches1_roma: {keypoint_matches1_roma.shape}, score: {score.shape}')
        keypts0 = data0['keypoints'][0].unsqueeze(0).to(device)
        keypts1 = data1['keypoints'][0].unsqueeze(0).to(device)
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
                     feature_path_raw_ref: Path = None):
    
    print(f"\n---start lg")
    print(f"\n----pairs path: {pairs_path}")
    pairs = []
    if not is_query_map_match and 'roma' in conf['model']['name']:
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_ref, overwrite)
    elif 'roma' in conf['model']['name']:
        pairs = match_from_paths_glue(conf, pairs_path, match_path, feature_path_q, feature_path_raw_ref, overwrite)
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    if 'roma' not in conf['model']['name']:
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
    print(f"feature_path_q: {feature_path_q}, feature_path_ref: {feature_path_ref}")
    if 'roma' in conf['model']['name']:
        use_roma = True
        dataset = FeaturePairsDatasetRoMa(pairs, feature_path_q, feature_path_ref)
    else:
        dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
        writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)
    print(f"dataset len: {dataset.__len__()}")
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=True)# keep shuffle=False
    
    # loader = torch.utils.data.DataLoader(
    #     dataset, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)# keep shuffle=False
    if use_roma:
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):            
            data0 = {k: v if str(k).startswith('image')
                        else v.to(device, non_blocking=True) for k, v in data[0].items()}
            data1 = {k: v if str(k).startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data[1].items()}
            name_img0 = data0['image_name'][0]
            name_img1 = data1['image_name'][0]
        
            kpt0, kpt1 = data0['keypoints'].cpu().numpy(), data1['keypoints'].cpu().numpy()

            # H, W, 3

            img_path0 = data0['image_path'][0]
            img_path1 = data1['image_path'][0]

            interp = 'cv2_area'
            # resize_max = conf['model']['resize_max']
            resize_max = None
            image0, scales_0 = get_image(img_path0, interp, resize_max)
            image1, scales_1 = get_image(img_path1, interp, resize_max)

            image0 = Image.fromarray(image0.astype(np.uint8))
            image1 = Image.fromarray(image1.astype(np.uint8))

            score_less_than_threshold = None
            index_nearest_2_kpts = None
            keypoints_match0_larger_than_threshold = None
            keypoint_match1_larger_than_threshold = None
            score_larger_than_threshold = None
            keypoints_match0_map_less_query_larger = None
            keypoint_match1_map_less_sp = None
            score_map_less_query_larger = None

            if not is_query_map_match:
                _, _, score_less_than_threshold, index_nearest_2_kpts, \
                keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, _, _, _ = model([image0, kpt0, image1, kpt1])
            else:
                _, _, score_less_than_threshold, index_nearest_2_kpts, \
                keypoints_match0_larger_than_threshold, keypoint_match1_larger_than_threshold, score_larger_than_threshold, \
                keypoints_match0_map_less_query_larger, keypoint_match1_map_less_sp, score_map_less_query_larger = model([image0, kpt0, image1, kpt1])


            old_list_matches0 = []
            old_list_matching_scores0 = []
            with h5py.File(str(match_path), 'r', libver='latest') as fm:
                pair = names_to_pair(name_img0, name_img1)
                old_list_matches0 = list(fm[names_to_pair(name_img0, name_img1)]['matches0'][:])
                old_list_matching_scores0 = list(fm[names_to_pair(name_img0, name_img1)]['matching_scores0'][:])
            kpt0 = [(int(x), int(y)) for x, y in kpt0[0]]
            kpt1 = [(int(x), int(y)) for x, y in kpt1[0]]
            # if len(kpt0) > len(old_list_matches0): this is keypoints added by roma, add -1 to old_list_matches0 and 0 to old_list_matching_scores0
            if len(kpt0) > len(old_list_matches0):
                for i in range(len(kpt0) - len(old_list_matches0)):
                    old_list_matches0.append(-1)
                    old_list_matching_scores0.append(0.0)

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
            # processing for 2 keypoints less than threshold
            for j in range(len(index_nearest_2_kpts)):
                keypoint_query = kpt0[index_nearest_2_kpts[j][0]]
                keypoint_reference = kpt1[index_nearest_2_kpts[j][1]]
                index_matches0 = dict_index_keypoints_query[str(keypoint_query)]
                score_matches0 = old_list_matching_scores0[index_matches0]
                score_matches0_roma = score_less_than_threshold[j]
                ### add new pair to old matches and scores
                if score_matches0_roma > score_matches0:
                    old_list_matches0[index_matches0] = dict_index_keypoints_reference[str(keypoint_reference)]
                    old_list_matching_scores0[index_matches0] = score_matches0_roma

            ## if is_query_map_match, add keypoints for only query keypoints
            if is_query_map_match:
                for j in range(score_map_less_query_larger.shape[0]):
                    new_query_keypoint = ((int(keypoints_match0_map_less_query_larger[j][0]), int(keypoints_match0_map_less_query_larger[j][1])))
                    old_map_keypoint = (int(keypoint_match1_map_less_sp[j][0]), int(keypoint_match1_map_less_sp[j][1]))
                    if str(new_query_keypoint) not in dict_index_keypoints_query:
                        old_len_kpt0 = len(kpt0)
                        index_old_map_keypoint = dict_index_keypoints_reference[str(old_map_keypoint)]
                        kpt0.append(new_query_keypoint)
                        old_list_matches0.append(index_old_map_keypoint)
                        old_list_matching_scores0.append(score_map_less_query_larger[j])
                        dict_index_keypoints_query[str(new_query_keypoint)] = old_len_kpt0


            ## processing for 2 keypoints larger than threshold
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
                    old_list_matches0.append(old_len_kpt1)
                    old_list_matching_scores0.append(score_larger_than_threshold[j])
                
            # write back kpt0 and kpt1 to feature_path_q and feature_path_ref
            kpt0 = np.array(kpt0)
            kpt1 = np.array(kpt1)
            with h5py.File(str(feature_path_q), 'a', libver='latest') as fq:
                uncertainty = 1.2*scales_0
                del fq[name_img0]['keypoints']
                fq[name_img0].create_dataset('keypoints', data=kpt0)
                fq[name_img0]['keypoints'].attrs['uncertainty'] = uncertainty
            with h5py.File(str(feature_path_ref), 'a', libver='latest') as fr:
                uncertainty = 1.2*scales_1
                del fr[name_img1]['keypoints']
                fr[name_img1].create_dataset('keypoints', data=kpt1)
                fr[name_img1]['keypoints'].attrs['uncertainty'] = uncertainty

            ## write back old_list_matches0 and old_list_matching_scores0 to match_path
            old_list_matches0 = np.array(old_list_matches0)
            old_list_matching_scores0 = np.array(old_list_matching_scores0)

            with h5py.File(str(match_path), 'a', libver='latest') as fm:
                del fm[names_to_pair(name_img0, name_img1)]['matches0']
                del fm[names_to_pair(name_img0, name_img1)]['matching_scores0']
                fm[names_to_pair(name_img0, name_img1)].create_dataset('matches0', data=old_list_matches0)
                fm[names_to_pair(name_img0, name_img1)].create_dataset('matching_scores0', data=old_list_matching_scores0)
            
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
    else:
        for idx, data in enumerate(tqdm(loader, smoothing=.1)):
            data = {k: v if k.startswith('image')
                    else v.to(device, non_blocking=True) for k, v in data.items()}
            pred = model(data)
            pair = names_to_pair(*pairs[idx])
            writer_queue.put((pair, pred))
        writer_queue.join()
    logger.info('Finished exporting matches.')




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
