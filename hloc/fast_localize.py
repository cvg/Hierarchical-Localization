from collections import defaultdict
import h5py
import numpy as np
import pycolmap
import torch

from .extract_features import ImageDataset
from .utils.parsers import names_to_pair
from .localize_sfm import QueryLocalizer

def get_local_features(query_processing_data_dir, 
             query_image_name,
             local_feature_conf,
             local_features_extractor_model,
             device,
             ):
    """
    Extract local features from a query image.
    Parameters:
        query_processing_data_dir: Path to the directory containing the query images.
        query_image_name: Name of the query image.
        local_features_extractor_model: Local feature extractor model.
    """
    query_image_dataset = ImageDataset(query_processing_data_dir, local_feature_conf['preprocessing'], [query_image_name])

    query_image_dataset_loader = torch.utils.data.DataLoader(
            query_image_dataset, num_workers=1, shuffle=False)

    for idx, data in enumerate(query_image_dataset_loader):
        with torch.no_grad():
            local_features = local_features_extractor_model({'image': data['image'].to(device)})
        local_features = {k: v[0] for k, v in local_features.items()}

        local_features['image_size'] = original_size = data['original_size'][0]
        
        # Scale keypoints
        size = np.array(data['image'].shape[-2:][::-1])
        scales = (original_size / size).to(device, dtype = torch.float32)
        local_features['keypoints'] = (local_features['keypoints'] + .5) * scales[None] - .5
        if 'scales' in local_features:
            local_features['scales'] *= scales.mean()
        # add keypoint uncertainties scaled to the original resolution
        uncertainty = getattr(local_features_extractor_model, 'detection_noise', 1) * scales.mean()
    
    return local_features, uncertainty

def get_global_descriptors(query_processing_data_dir, 
                           query_image_name, 
                           global_descriptor_conf, 
                           global_descriptor_model, 
                           device):
    """
    Extract global descriptors from a query image.
    Parameters:
        query_processing_data_dir: Path to the directory containing the query images.
        query_image_name: Name of the query image.
        global_descriptor_conf: Configuration of the global descriptor model.
        global_descriptor_model: Global descriptor model.
        device: Device to run the model on.
    """
    query_image_dataset = ImageDataset(query_processing_data_dir, global_descriptor_conf['preprocessing'], [query_image_name])
    query_image_dataset_loader = torch.utils.data.DataLoader(
            query_image_dataset, num_workers=1, shuffle=False)

    for idx, data in enumerate(query_image_dataset_loader):
        with torch.no_grad():
            global_descriptor = global_descriptor_model({'image': data['image'].to(device)})
        global_descriptor = {k: v[0] for k, v in global_descriptor.items()}
        global_descriptor['image_size'] = original_size = data['original_size'][0]
    
    return global_descriptor

def get_candidate_matches(global_descriptor, 
                          db_global_descriptors, 
                          db_image_names):
    """
    Find the top 10 candidate images from the database.
    Parameters:
        global_descriptor: Global descriptor of the query image.
        db_global_descriptors: Global descriptors of the database images.
        db_image_names: Names of the database images.
    """
    similarity_scores = torch.einsum('id,jd->ij', global_descriptor['global_descriptor'][None, :], db_global_descriptors)
    topk = torch.topk(similarity_scores, 10, dim=1)
    nearest_candidate_images = db_image_names[topk.indices[0].cpu().numpy()]
    nearest_image_descriptors = db_global_descriptors[topk.indices[0]]
    return nearest_candidate_images, nearest_image_descriptors

def get_local_matches(db_local_features_path, 
                      nearest_candidate_images, 
                      local_features, 
                      matcher_model, 
                      query_image_name, 
                      device):
    """
    Find local matches between the query image and the candidate images.
    Parameters:
        db_local_features_path: Path to the database local features.
        nearest_candidate_images: Names of the candidate images.
        local_features: Local features of the query image.
        matcher_model: Local feature matcher model.
        query_image_name: Name of the query image.
        device: Device to run the model on.
    """
    ## Matching image using the image pairs - Optimized
    local_matches = {}
    with h5py.File(db_local_features_path, 'r') as db_local_features:
        for image_name in nearest_candidate_images:
            data = {}

            for k in ['keypoints', 'scores', 'descriptors']:
                data[k + '0'] = local_features[k]
            data['image0'] = torch.empty((1,)+tuple(local_features['image_size'])[::-1], device = device)

            for k in ['keypoints', 'scores', 'descriptors']:
                v = db_local_features[image_name][k]
                data[k + '1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,)+tuple(db_local_features[image_name]['image_size'])[::-1], device = device)

            for k in data:
                data[k] = data[k].unsqueeze(0)

            with torch.no_grad():
                match = matcher_model(data)
            # breakpoint()
            local_matches[names_to_pair(query_image_name,image_name)] = match

    return local_matches

def _get_matches_from_tensor(local_matches, name0, name1):
    pair_index = names_to_pair(name0, name1)
    matches = local_matches[pair_index]['matches0'].squeeze().detach().cpu().numpy()
    scores = local_matches[pair_index]['matching_scores0'].squeeze().detach().cpu().numpy()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    scores = scores[idx]
    return matches, scores

def get_pose(query_processing_data_dir, 
             query_image_name, 
             db_reconstruction, 
             nearest_candidate_images, 
             local_matches, 
             local_features):
    ## Now we have global candidate and thier mathces. We use this, along with SfM reconstruction to localize the image.
    reconstruction = pycolmap.Reconstruction(db_reconstruction.__str__())
    query_camera = pycolmap.infer_camera_from_image(query_processing_data_dir / query_image_name)
    ref_ids = [reconstruction.find_image_with_name(r).image_id for r in nearest_candidate_images]
    conf = {
        'estimation': {'ransac': {'max_error': 12}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(reconstruction, conf)

    # pose from cluster
    kqp = local_features['keypoints'].cpu().numpy()
    kqp += 0.5 # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(ref_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            print(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                for p in image.points2D])
        this_match, _ = _get_matches_from_tensor(local_matches, query_image_name, image.name)
        this_match = this_match[points3D_ids[this_match[:, 1]] != -1]
        num_matches += len(this_match)
        for idx, m in this_match:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kqp, mkp_idxs, mp3d_ids, query_camera)
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                    for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': ref_ids,
        'PnP_ret': ret,
        'keypoints_query': kqp[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,  # we don't log xyz anymore because of file size
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log

def localize(query_processing_data_dir, query_image_name, 
             device, 
             local_feature_conf, local_features_extractor_model, 
             global_descriptor_conf, global_descriptor_model, 
             db_global_descriptors, db_image_names,
             db_local_features_path, matcher_model, 
             db_reconstruction):
    local_features, uncertainty = get_local_features(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name,
        local_feature_conf = local_feature_conf,
        local_features_extractor_model = local_features_extractor_model,
        device = device
    )

    global_descriptor = get_global_descriptors(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name, 
        global_descriptor_conf = global_descriptor_conf, 
        global_descriptor_model = global_descriptor_model, 
        device = device
    )

    nearest_candidate_images, nearest_image_descriptors = get_candidate_matches(
        global_descriptor = global_descriptor, 
        db_global_descriptors = db_global_descriptors, 
        db_image_names = db_image_names
    )

    local_matches = get_local_matches(
        db_local_features_path = db_local_features_path, 
        nearest_candidate_images = nearest_candidate_images, 
        local_features = local_features, 
        matcher_model = matcher_model, 
        query_image_name = query_image_name, 
        device = device
    )

    ret, log = get_pose(
        query_processing_data_dir = query_processing_data_dir, 
        query_image_name = query_image_name, 
        db_reconstruction = db_reconstruction, 
        nearest_candidate_images = nearest_candidate_images, 
        local_matches = local_matches, 
        local_features = local_features
    )

    return ret, log
