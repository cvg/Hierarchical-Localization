import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Union
import h5py
from tqdm import tqdm
import pickle
import pycolmap

from .utils.parsers import parse_image_lists, parse_retrieval, names_to_pair


def do_covisibility_clustering(frame_ids: List[int],
                               reconstruction: pycolmap.Reconstruction):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D()
                for obs in
                reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


# TODO: support all options of the absolute pose estimator
class QueryLocalizer:
    def __init__(self, reconstruction, max_error_px):
        self.reconstruction = reconstruction
        self.max_error_px = max_error_px

    def localize(self, points2D, points3D_id, query_camera):
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        ret = pycolmap.absolute_pose_estimation(
            points2D, points3D, query_camera, self.max_error_px)
        ret['camera'] = {
            'model': query_camera.model_name,
            'width': query_camera.width,
            'height': query_camera.height,
            'params': query_camera.params,
        }
        return ret


def pose_from_cluster(
        localizer: QueryLocalizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        feature_file,
        match_file,
        **kwargs):

    kpq = feature_file[qname]['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            logging.debug(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                 for p in image.points2D])

        pair = names_to_pair(qname, image.name)
        matches = match_file[pair]['matches0'].__array__()
        valid = np.where(matches > -1)[0]
        valid = valid[points3D_ids[matches[valid]] != -1]
        num_matches += len(valid)

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(mkpq, mp3d_ids, query_camera, **kwargs)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    # deprecate logging 3D points because they make the log files too large
    return ret, mkpq, None, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)


def main(rec: Union[Path, pycolmap.Reconstruction],
         queries: Path,
         retrieval: Path,
         features: Path,
         matches: Path,
         results: Path,
         ransac_thresh: int = 12,
         covisibility_clustering: bool = False,
         prepend_camera_name: bool = False):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading the 3D model...')
    if not isinstance(rec, pycolmap.Reconstruction):
        rec = pycolmap.Reconstruction(rec)
    localizer = QueryLocalizer(rec, ransac_thresh)
    db_name_to_id = {image.name: i for i, image in rec.images.items()}

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logging.warning(
                f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, rec)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = (
                        pose_from_cluster(
                            localizer, qname, qcam, cluster_ids,
                            feature_file, match_file))
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append({
                    'db': cluster_ids,
                    'PnP_ret': ret,
                    'keypoints_query': mkpq,
                    'points3D_xyz': mp3d,
                    'points3D_ids': mp3d_ids,
                    'num_matches': num_matches,
                    'keypoint_index_to_db': map_,
                })
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'db': db_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:
            ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = pose_from_cluster(
                localizer, qname, qcam, db_ids, feature_file, match_file)

            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
            else:
                closest = rec.images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
            logs['loc'][qname] = {
                'db': db_ids,
                'PnP_ret': ret,
                'keypoints_query': mkpq,
                'points3D_xyz': mp3d,
                'points3D_ids': mp3d_ids,
                'num_matches': num_matches,
                'keypoint_index_to_db': map_,
                'covisibility_clustering': covisibility_clustering,
            }

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
