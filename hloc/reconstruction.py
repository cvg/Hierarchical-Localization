import argparse
import multiprocessing
from pathlib import Path
from typing import Any

import pycolmap
import tqdm

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)


def create_empty_db(database_path: Path) -> None:
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    with pycolmap.Database.open(database_path) as _:
        pass


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: list[str] | None = None,
    options: dict[str, Any] | pycolmap.ImageReaderOptions | None = None,
) -> None:
    logger.info("Importing images into the database...")
    if options is None:
        reader_options = pycolmap.ImageReaderOptions()
    elif isinstance(options, pycolmap.ImageReaderOptions):
        reader_options = options
    else:
        reader_options = pycolmap.ImageReaderOptions(options)
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise OSError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_names=image_list or [],
            options=reader_options,
        )


def get_image_ids(database_path: Path) -> dict[str, int]:
    images = {}
    with pycolmap.Database.open(database_path) as db:
        images = {image.name: image.image_id for image in db.read_all_images()}
    return images


def incremental_mapping(
    database_path: Path,
    image_dir: Path,
    sfm_path: Path,
    options: dict[str, Any] | pycolmap.IncrementalPipelineOptions | None = None,
) -> dict[int, pycolmap.Reconstruction]:
    with pycolmap.Database.open(database_path) as db:
        num_images = db.num_images()
    pbars: list[tqdm.tqdm] = []

    def restart_progress_bar() -> None:
        if len(pbars) > 0:
            pbars[-1].close()
        pbars.append(
            tqdm.tqdm(
                total=num_images,
                desc=f"Reconstruction {len(pbars)}",
                unit="images",
                postfix="registered",
            )
        )
        pbars[-1].update(2)

    if options is None:
        pipeline_options = pycolmap.IncrementalPipelineOptions()
    elif isinstance(options, pycolmap.IncrementalPipelineOptions):
        pipeline_options = options
    else:
        pipeline_options = pycolmap.IncrementalPipelineOptions(options)

    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_dir,
        sfm_path,
        options=pipeline_options,
        initial_image_pair_callback=restart_progress_bar,
        next_image_callback=lambda: pbars[-1].update(1),
    )

    return reconstructions


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: dict[str, Any] | None = None,
) -> pycolmap.Reconstruction | None:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}

    with OutputCapture(verbose):
        reconstructions = incremental_mapping(
            database_path, image_dir, models_path, options=options
        )

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f"Largest model is #{largest_index} with {largest_num_images} images.")

    reconstructions[largest_index].write(sfm_dir)
    return reconstructions[largest_index]


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: float | None = None,
    image_list: list[str] | None = None,
    image_options: dict[str, Any] | None = None,
    mapper_options: dict[str, Any] | None = None,
) -> pycolmap.Reconstruction | None:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    logger.info(f"Writing COLMAP logs to {sfm_dir / 'colmap.LOG.*'}")
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)
    reconstruction = run_reconstruction(
        sfm_dir, database, image_dir, verbose, mapper_options
    )
    if reconstruction is not None:
        logger.info(
            f"Reconstruction statistics:\n{reconstruction.summary()}"
            + f"\n\tnum_input_images = {len(image_ids)}"
        )
    return reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument(
        "--camera_mode",
        type=str,
        default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--image_options",
        nargs="+",
        default=[],
        help=f"List of key=value from {pycolmap.ImageReaderOptions().todict()}",
    )
    parser.add_argument(
        "--mapper_options",
        nargs="+",
        default=[],
        help=f"List of key=value from {pycolmap.IncrementalPipelineOptions().todict()}",
    )
    args = parser.parse_args().__dict__

    camera_mode = getattr(pycolmap.CameraMode, args.pop("camera_mode"))
    image_options = parse_option_args(
        args.pop("image_options"), pycolmap.ImageReaderOptions()
    )
    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalPipelineOptions()
    )

    main(
        **args,
        camera_mode=camera_mode,
        image_options=image_options,
        mapper_options=mapper_options,
    )
