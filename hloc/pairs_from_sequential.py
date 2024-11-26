import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional, Union, List

from . import logger
from .utils.parsers import parse_image_lists
from .utils.io import list_h5_names
# based on https://github.com/hugoycj/light-hloc/blob/e518dbaa06b6b57e5663318cff237066ac26bd08/lighthloc/associators/pairs_from_sequance.py


def main(
    output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
    overlap: Optional[int] = 10,
    quadratic_overlap: bool = True,
):
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            print(image_list)
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    pairs = []
    N = len(names_q)

    for i in range(N - 1):
        for j in range(i + 1, min(i + overlap + 1, N)):
            pairs.append((names_q[i], names_q[j]))

            if quadratic_overlap:
                q = 2 ** (j - i)
                if q > overlap and i + q < N:
                    pairs.append((names_q[i], names_q[i + q]))

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a list of image pairs based on the sequence of images on alphabetic order"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument(
        "--overlap", type=int, default=10, help="Number of overlapping image pairs"
    )
    parser.add_argument(
        "--quadratic_overlap",
        action="store_true",
        help="Whether to match images against their quadratic neighbors.",
    )
    args = parser.parse_args()
    main(**args.__dict__)
