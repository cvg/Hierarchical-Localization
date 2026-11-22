import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import numpy as np
import pycolmap
import torch

from . import logger
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists


def parse_names(
    prefix: str | Iterable[str] | None,
    names: str | Path | Iterable[str] | None,
    names_all: list[str],
) -> list[str]:
    if prefix is not None:
        prefix_tuple = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        result = [n for n in names_all if n.startswith(prefix_tuple)]
        if len(result) == 0:
            raise ValueError(f"Could not find any image with the prefix `{prefix}`.")
        return result
    elif names is not None:
        if isinstance(names, (str, Path)):
            return parse_image_lists(Path(names))
        elif isinstance(names, Iterable):
            return list(names)
        else:
            raise ValueError(
                f"Unknown type of image list: {names}."
                "Provide either a list or a path to a list file."
            )
    else:
        return names_all


def get_descriptors(
    names: Sequence[str],
    path: Path | Sequence[Path],
    name2idx: dict[str, int] | None = None,
    key: str = "global_descriptor",
) -> torch.Tensor:
    if name2idx is None:
        assert isinstance(path, (Path, str))
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        assert not isinstance(path, (Path, str))
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor | np.ndarray,
    invalid: np.ndarray,
    num_select: int,
    min_score: float | None = None,
) -> list[tuple[int, int]]:
    assert scores.shape == invalid.shape
    scores_t = torch.from_numpy(scores) if isinstance(scores, np.ndarray) else scores
    invalid_t = torch.from_numpy(invalid).to(scores_t.device)
    if min_score is not None:
        invalid_t |= scores_t < min_score
    scores_t.masked_fill_(invalid_t, float("-inf"))

    topk = torch.topk(scores_t, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid), strict=True):
        pairs.append((int(i), int(indices[i, j])))
    return pairs


def main(
    descriptors: Path,
    output: Path,
    num_matched: int,
    query_prefix: str | Iterable[str] | None = None,
    query_list: Path | Iterable[str] | None = None,
    db_prefix: str | Iterable[str] | None = None,
    db_list: Path | Iterable[str] | None = None,
    db_model: Path | None = None,
    db_descriptors: Path | Sequence[Path] | None = None,
) -> None:
    logger.info("Extracting image pairs from a retrieval database.")

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_desc_paths: Sequence[Path] = [descriptors]
    elif isinstance(db_descriptors, (Path, str)):
        db_desc_paths = [Path(db_descriptors)]
    else:
        db_desc_paths = db_descriptors
    name2db = {n: i for i, p in enumerate(db_desc_paths) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = pycolmap.Reconstruction(db_model).images
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError("Could not find any database image.")
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_desc = get_descriptors(db_names, db_desc_paths, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum("id,jd->ij", query_desc.to(device), db_desc.to(device))

    # Avoid self-matching
    self_mask = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs_idx = pairs_from_score_matrix(sim, self_mask, num_matched, min_score=0)
    pairs = [(query_names[i], db_names[j]) for i, j in pairs_idx]

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_matched", type=int, required=True)
    parser.add_argument("--query_prefix", type=str, nargs="+")
    parser.add_argument("--query_list", type=Path)
    parser.add_argument("--db_prefix", type=str, nargs="+")
    parser.add_argument("--db_list", type=Path)
    parser.add_argument("--db_model", type=Path)
    parser.add_argument("--db_descriptors", type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
