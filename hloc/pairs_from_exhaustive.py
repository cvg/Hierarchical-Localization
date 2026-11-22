import argparse
from collections.abc import Iterable
from pathlib import Path

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists


def main(
    output: Path,
    image_list: Path | str | Iterable[str] | None = None,
    features: Path | None = None,
    ref_list: Path | str | Iterable[str] | None = None,
    ref_features: Path | None = None,
) -> None:
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(Path(image_list))
        elif isinstance(image_list, Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    self_matching = False
    if ref_list is not None:
        if isinstance(ref_list, (str, Path)):
            names_ref = parse_image_lists(Path(ref_list))
        elif isinstance(ref_list, Iterable):
            names_ref = list(ref_list)
        else:
            raise ValueError(f"Unknown type for reference image list: {ref_list}")
    elif ref_features is not None:
        names_ref = list_h5_names(ref_features)
    else:
        self_matching = True
        names_ref = names_q

    pairs = []
    for i, n1 in enumerate(names_q):
        for j, n2 in enumerate(names_ref):
            if self_matching and j <= i:
                continue
            pairs.append((n1, n2))

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--ref_list", type=Path)
    parser.add_argument("--ref_features", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
