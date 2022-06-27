import numpy as np
import pycolmap


def to_homogeneous(p):
    return np.pad(p, ((0, 0),) * (p.ndim - 1) + ((0, 1),), constant_values=1)


def vector_to_cross_product_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_epipolar_errors(qvec_r2t, tvec_r2t, p2d_r, p2d_t):
    T_r2t = pose_matrix_from_qvec_tvec(qvec_r2t, tvec_r2t)
    # Compute errors in normalized plane to avoid distortion.
    E = vector_to_cross_product_matrix(T_r2t[: 3, -1]) @ T_r2t[: 3, : 3]
    l2d_r2t = (E @ to_homogeneous(p2d_r).T).T
    l2d_t2r = (E.T @ to_homogeneous(p2d_t).T).T
    errors_r = (
        np.abs(np.sum(to_homogeneous(p2d_r) * l2d_t2r, axis=1)) /
        np.linalg.norm(l2d_t2r[:, : 2], axis=1))
    errors_t = (
        np.abs(np.sum(to_homogeneous(p2d_t) * l2d_r2t, axis=1)) /
        np.linalg.norm(l2d_r2t[:, : 2], axis=1))
    return E, errors_r, errors_t


def pose_matrix_from_qvec_tvec(qvec, tvec):
    pose = np.zeros((4, 4))
    pose[: 3, : 3] = pycolmap.qvec_to_rotmat(qvec)
    pose[: 3, -1] = tvec
    pose[-1, -1] = 1
    return pose
