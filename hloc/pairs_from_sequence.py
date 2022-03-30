import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional, Union, List

from . import logger
from .utils.parsers import parse_image_lists
from .utils.io import list_h5_names


def main(
        output: Path,
        image_list: Optional[Union[Path, List[str]]] = None,
        features: Optional[Path] = None,
        window_size: Optional[int]= 2,
        loop: bool = False):

    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f'Unknown type for image list: {image_list}')
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError('Provide either a list of images or a feature file.')

    pairs = []
    tot = len(names_q)
    
    if loop:
        for i in range(tot):
            for j in range(i + 1, i + window_size):
                pairs.append((names_q[i - tot], names_q[j - tot]))
                
    else:
        for i in range(tot - 1):
            for j in range(i + 1, min(i + window_size, tot)): 
                pairs.append((names_q[i], names_q[j]))

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a list of image pairs based on the sequence of images on alphabetic order")
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--window_size', type=int, default=4, help="Size of the window of images to match")
    parser.add_argument('--loop', action="store_true", help="Create a loop sequence (last elements matched with first ones)")
    args = parser.parse_args()
    main(**args.__dict__)
