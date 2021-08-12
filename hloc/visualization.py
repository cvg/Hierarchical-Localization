from matplotlib import cm
import random
import numpy as np
import pickle

from .utils.read_write_model import read_images_binary, read_points3D_binary
from .utils.viz import (
        plot_images, plot_keypoints, plot_matches, cm_RdGn, add_text)
from .utils.io import read_image


def visualize_sfm_2d(sfm_model, image_dir, color_by='visibility',
                     selected=[], n=1, seed=0, dpi=75):
    assert sfm_model.exists()
    assert image_dir.exists()

    images = read_images_binary(sfm_model / 'images.bin')
    if color_by in ['track_length', 'depth']:
        points3D = read_points3D_binary(sfm_model / 'points3D.bin')

    if not selected:
        image_ids = list(images.keys())
        selected = random.Random(seed).sample(image_ids, n)

    for i in selected:
        name = images[i].name
        image = read_image(image_dir / name)
        keypoints = images[i].xys
        visible = images[i].point3D_ids != -1

        if color_by == 'visibility':
            color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
        elif color_by == 'track_length':
            tl = np.array([len(points3D[j].image_ids) if j != -1 else 1
                           for j in images[i].point3D_ids])
            max_, med_ = np.max(tl), np.median(tl[tl > 1])
            tl = np.log(tl)
            color = cm.jet(tl / tl.max()).tolist()
            text = f'max/median track length: {max_}/{med_}'
        elif color_by == 'depth':
            p3ids = images[i].point3D_ids
            p3D = np.array([points3D[j].xyz for j in p3ids if j != -1])
            z = (images[i].qvec2rotmat() @ p3D.T)[-1] + images[i].tvec[-1]
            z -= z.min()
            color = cm.jet(z / np.percentile(z, 99.9))
            text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            keypoints = keypoints[visible]
        else:
            raise NotImplementedError(f'Coloring not implemented: {color_by}.')

        plot_images([image], dpi=dpi)
        plot_keypoints([keypoints], colors=[color], ps=4)
        add_text(0, text)
        add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')


def visualize_loc(results, image_dir, sfm_model=None, top_k_db=2,
                  selected=[], n=1, seed=0, prefix=None, dpi=75):
    assert image_dir.exists()

    with open(str(results)+'_logs.pkl', 'rb') as f:
        logs = pickle.load(f)

    if not selected:
        queries = list(logs['loc'].keys())
        if prefix:
            queries = [q for q in queries if q.startswith(prefix)]
        selected = random.Random(seed).sample(queries, n)

    is_sfm = sfm_model is not None
    if is_sfm:
        assert sfm_model.exists()
        images = read_images_binary(sfm_model / 'images.bin')
        points3D = read_points3D_binary(sfm_model / 'points3D.bin')

    for q in selected:
        q_image = read_image(image_dir / q)
        loc = logs['loc'][q]
        if loc.get('covisibility_clustering', False):
            # select the first, largest cluster if the localization failed
            loc = loc['log_clusters'][loc['best_cluster'] or 0]

        inliers = np.array(loc['PnP_ret']['inliers'])
        mkp_q = loc['keypoints_query']
        n = len(loc['db'])
        if is_sfm:
            # for each pair of query keypoint and its matched 3D point,
            # we need to find its corresponding keypoint in each database image
            # that observes it. We also count the number of inliers in each.
            kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
            counts = np.zeros(n)
            dbs_kp_q_db = [[] for _ in range(n)]
            inliers_dbs = [[] for _ in range(n)]
            for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                             kp_to_3D_to_db)):
                p3D = points3D[p3D_id]
                for db_idx in db_idxs:
                    counts[db_idx] += inl
                    kp_db = p3D.point2D_idxs[
                        p3D.image_ids == loc['db'][db_idx]][0]
                    dbs_kp_q_db[db_idx].append((i, kp_db))
                    inliers_dbs[db_idx].append(inl)
        else:
            # for inloc the database keypoints are already in the logs
            assert 'keypoints_db' in loc
            assert 'indices_db' in loc
            counts = np.array([
                np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

        # display the database images with the most inlier matches
        db_sort = np.argsort(-counts)
        for db_idx in db_sort[:top_k_db]:
            if is_sfm:
                db = images[loc['db'][db_idx]]
                db_name = db.name
                db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
                kp_q = mkp_q[db_kp_q_db[:, 0]]
                kp_db = db.xys[db_kp_q_db[:, 1]]
                inliers_db = inliers_dbs[db_idx]
            else:
                db_name = loc['db'][db_idx]
                kp_q = mkp_q[loc['indices_db'] == db_idx]
                kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
                inliers_db = inliers[loc['indices_db'] == db_idx]

            db_image = read_image(image_dir / db_name)
            color = cm_RdGn(inliers_db).tolist()
            text = f'inliers: {sum(inliers_db)}/{len(inliers_db)}'

            plot_images([q_image, db_image], dpi=dpi)
            plot_matches(kp_q, kp_db, color, a=0.1)
            add_text(0, text)
            add_text(0, q, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
            add_text(1, db_name, pos=(0.01, 0.01), fs=5,
                     lcolor=None, va='bottom')
