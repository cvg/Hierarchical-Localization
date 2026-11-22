import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pycolmap

logger = logging.getLogger(__name__)


@overload
def parse_image_list(
    path: Path, with_intrinsics: Literal[False] = False
) -> list[str]: ...


@overload
def parse_image_list(
    path: Path, with_intrinsics: Literal[True]
) -> list[tuple[str, pycolmap.Camera]]: ...


@overload
def parse_image_list(
    path: Path, with_intrinsics: bool
) -> list[str] | list[tuple[str, pycolmap.Camera]]: ...


def parse_image_list(
    path: Path, with_intrinsics: bool = False
) -> list[str] | list[tuple[str, pycolmap.Camera]]:
    images_str: list[str] = []
    images_cam: list[tuple[str, pycolmap.Camera]] = []
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            if len(line) == 0 or line[0] == "#":
                continue
            name, *data = line.split()
            if with_intrinsics:
                model, width, height, *params = data
                cam_params = np.array(params, float)
                cam = pycolmap.Camera(
                    model=model,
                    width=int(width),
                    height=int(height),
                    params=cam_params,
                )
                images_cam.append((name, cam))
            else:
                images_str.append(name)

    images: list[str] | list[tuple[str, pycolmap.Camera]] = (
        images_cam if with_intrinsics else images_str
    )
    assert len(images) > 0
    logger.info(f"Imported {len(images)} images from {path.name}")
    return images


@overload
def parse_image_lists(
    paths: Path, with_intrinsics: Literal[False] = False
) -> list[str]: ...


@overload
def parse_image_lists(
    paths: Path, with_intrinsics: Literal[True]
) -> list[tuple[str, pycolmap.Camera]]: ...


@overload
def parse_image_lists(
    paths: Path, with_intrinsics: bool
) -> list[str] | list[tuple[str, pycolmap.Camera]]: ...


def parse_image_lists(
    paths: Path, with_intrinsics: bool = False
) -> list[str] | list[tuple[str, pycolmap.Camera]]:
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    if with_intrinsics:
        images_cam: list[tuple[str, pycolmap.Camera]] = []
        for lfile in files:
            images_cam += parse_image_list(lfile, with_intrinsics=True)
        return images_cam
    images_str: list[str] = []
    for lfile in files:
        images_str += parse_image_list(lfile, with_intrinsics=False)
    return images_str


def parse_retrieval(path: Path | str) -> dict[str, list[str]]:
    retrieval: defaultdict[str, list[str]] = defaultdict(list)
    with open(path) as f:
        for p in f.read().rstrip("\n").split("\n"):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0: str, name1: str, separator: str = "/") -> str:
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def names_to_pair_old(name0: str, name1: str) -> str:
    return names_to_pair(name0, name1, separator="_")
