import argparse
import collections.abc as collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pycolmap

from . import logger
from .triangulation import parse_option_args
from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists

PAIR_GENERATORS = {
    "exhaustive": (
        pycolmap.ExhaustivePairGenerator,
        pycolmap.ExhaustivePairingOptions,
    ),
    "sequential": (
        pycolmap.SequentialPairGenerator,
        pycolmap.SequentialPairingOptions,
    ),
    "spatial": (
        pycolmap.SpatialPairGenerator,
        pycolmap.SpatialPairingOptions,
    ),
    "vocabtree": (
        pycolmap.VocabTreePairGenerator,
        pycolmap.VocabTreePairingOptions,
    ),
}


def _resolve_names(
    image_list: Optional[Union[Path, str, List[str]]],
    features: Optional[Path],
    label: str = "image list",
) -> List[str]:
    if image_list is not None and features is not None:
        raise ValueError(f"Cannot provide both {label} and a feature file.")
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            return parse_image_lists(Path(image_list))
        if isinstance(image_list, collections.Iterable):
            return list(image_list)
        raise ValueError(f"Unknown type for {label}: {image_list}")
    if features is not None:
        return list_h5_names(features)
    raise ValueError(f"Provide either {label} or a feature file.")


def _pairs_from_database(
    db: pycolmap.Database,
    mode: str,
    options: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    gen_cls, opts_cls = PAIR_GENERATORS[mode]
    pairing_options = opts_cls(options) if options else opts_cls()
    if not pairing_options.check():
        raise ValueError(f"Invalid pairing options for mode '{mode}': {options}")
    id_to_name = {img.image_id: img.name for img in db.read_all_images()}
    if not id_to_name:
        return []
    generator = gen_cls(pairing_options, db)
    return list(
        dict.fromkeys(
            (id_to_name[i], id_to_name[j]) for i, j in generator.all_pairs() if i != j
        )
    )


def main(
    output: Path,
    mode: str = "exhaustive",
    database: Optional[Path] = None,
    image_list: Optional[Union[Path, str, List[str]]] = None,
    features: Optional[Path] = None,
    ref_list: Optional[Union[Path, str, List[str]]] = None,
    ref_features: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
) -> None:
    if mode not in PAIR_GENERATORS:
        raise ValueError(
            f"Unknown pairing mode '{mode}', expected one of {list(PAIR_GENERATORS)}."
        )

    has_ref = ref_list is not None or ref_features is not None
    if database is not None:
        if image_list is not None or features is not None or has_ref:
            raise ValueError(
                "Cannot combine --database with image_list/features/ref_list."
            )
        database = Path(database)
        if not database.is_file():
            raise FileNotFoundError(f"Database not found: {database}")
        with pycolmap.Database.open(database) as db:
            pairs = _pairs_from_database(db, mode, options)
    elif has_ref:
        if mode != "exhaustive":
            raise ValueError(
                "ref_list/ref_features is only supported in exhaustive mode."
            )
        if options:
            raise ValueError(
                "Pairing options are not supported with ref_list/ref_features."
            )
        names_q = _resolve_names(image_list, features, "image list")
        names_ref = _resolve_names(ref_list, ref_features, "reference image list")
        pairs = [(n1, n2) for n1 in names_q for n2 in names_ref if n1 != n2]
    else:
        if mode in ("spatial", "vocabtree"):
            raise ValueError(
                f"Pairing mode '{mode}' requires an existing COLMAP --database."
            )
        if mode == "sequential" and options and options.get("loop_detection"):
            raise ValueError(
                "Sequential pairing with loop_detection=True requires an existing "
                "COLMAP --database."
            )
        names = _resolve_names(image_list, features, "image list")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate image names found in image list.")
        with pycolmap.Database.open(":memory:") as db:
            with pycolmap.DatabaseTransaction(db):
                cam_id = db.write_camera(
                    pycolmap.Camera(
                        model="SIMPLE_PINHOLE",
                        width=1,
                        height=1,
                        params=[1.0, 0.5, 0.5],
                    )
                )
                for name in names:
                    db.write_image(pycolmap.Image(name=name, camera_id=cam_id))
            pairs = _pairs_from_database(db, mode, options)

    logger.info(f"Found {len(pairs)} pairs.")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        type=str,
        default="exhaustive",
        choices=list(PAIR_GENERATORS.keys()),
    )
    parser.add_argument("--database", type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--ref_list", type=Path)
    parser.add_argument("--ref_features", type=Path)
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="List of key=value pairing options for the selected COLMAP mode.",
    )
    args = parser.parse_args().__dict__
    _, opts_cls = PAIR_GENERATORS[args["mode"]]
    pairing_options = parse_option_args(args.pop("options"), opts_cls())
    main(**args, options=pairing_options)
