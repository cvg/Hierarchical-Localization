import numpy as np
import pytest

from hloc.localize_inloc import interpolate_scan


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("border", [False, True])
def test_scan_interpolation_matches_an_affine_surface(dtype, border):
    x, y = np.meshgrid(np.arange(5), np.arange(4))
    scan = np.stack([x, y, 2 * x - 3 * y + 7], axis=-1).astype(dtype)
    keypoints = np.array([[0.25, 0.5], [2.3, 1.2]], dtype=np.float32)
    if border:
        keypoints = np.array([[0, 0], [4, 3], [0, 2.5], [4, 0.25]], dtype=np.float32)
    expected = np.stack(
        [
            keypoints[:, 0],
            keypoints[:, 1],
            2 * keypoints[:, 0] - 3 * keypoints[:, 1] + 7,
        ],
        -1,
    )
    points, valid = interpolate_scan(scan, keypoints)
    assert valid.all()
    assert points.dtype == dtype
    np.testing.assert_allclose(points, expected, atol=2e-6)


def test_scan_nearest_fallback_retains_valid_depth():
    scan = np.ones((4, 5, 3), dtype=np.float32)
    scan[1, 2] = np.nan
    points, valid = interpolate_scan(scan, np.array([[1.1, 1.1]], dtype=np.float64))
    assert valid.all()
    np.testing.assert_allclose(points, [[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("keypoint", [[-1.0, 0.5], [5.0, 1.0], [1.0, 4.0]])
def test_scan_interpolation_rejects_outside_pixels(keypoint):
    with pytest.raises(AssertionError):
        interpolate_scan(np.ones((4, 5, 3)), np.array([keypoint]))
