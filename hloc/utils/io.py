from pathlib import Path
import cv2
import h5py

from .parsers import names_to_pair


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r') as hfile:
        reverse = False
        pair = names_to_pair(name0, name1)
        if pair not in hfile:
            pair = names_to_pair(name1, name0)
            if pair not in hfile:
                raise ValueError(
                    f'Could not find pair {(name0, name1)}... '
                    'Maybe you matched with a different list of pairs? ')
            reverse = True
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = matches[:, ::-1]
    scores = scores[idx]
    return matches, scores
