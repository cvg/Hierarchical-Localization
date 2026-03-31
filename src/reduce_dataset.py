import argparse
import csv
from pathlib import Path
import shutil
from typing import Iterable, List, Set


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
CSV_IMAGE_FIELDS = ["image_path", "image", "file", "filename", "path"]


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS])


def sample_every_n(images: List[Path], step: int) -> Set[str]:
    if step <= 0:
        raise ValueError("step must be >= 1")
    return {images[i].name for i in range(0, len(images), step)}


def choose_image_field(fieldnames: Iterable[str]) -> str:
    for candidate in CSV_IMAGE_FIELDS:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        "CSV does not contain a known image field. "
        f"Expected one of {CSV_IMAGE_FIELDS} but got {list(fieldnames)}"
    )


def filter_csv(csv_path: Path, keep_names: Set[str], dry_run: bool) -> None:
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        image_field = choose_image_field(reader.fieldnames)
        rows = [row for row in reader if row.get(image_field) in keep_names]

    if dry_run:
        print(f"[dry-run] Would rewrite CSV: {csv_path} (rows kept: {len(rows)})")
        return

    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"Backed up CSV -> {backup_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rewrote CSV: {csv_path} (rows kept: {len(rows)})")


def remove_images(images: List[Path], keep_names: Set[str], dry_run: bool) -> None:
    removed = 0
    for image in images:
        if image.name in keep_names:
            continue
        removed += 1
        if dry_run:
            continue
        image.unlink()
    if dry_run:
        print(f"[dry-run] Would remove {removed} images")
    else:
        print(f"Removed {removed} images")


def process_split(split_dir: Path, step: int, dry_run: bool) -> Set[str]:
    images = list_images(split_dir)
    keep_names = sample_every_n(images, step)
    print(f"{split_dir.name}: total={len(images)} keep={len(keep_names)} step={step}")
    remove_images(images, keep_names, dry_run)
    return keep_names


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "datasets" / "dragos" / "images" / "simulation"
    parser = argparse.ArgumentParser(
        description="Downsample dragos train/test images and update CSV metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help="Path to dataset root containing train/ and test/ directories.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Keep one image every N images.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path to update. Defaults to <dataset-root>/simulation.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without deleting files or rewriting CSV.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"train/test not found under {dataset_root}. "
            "Expected <root>/train and <root>/test. "
            "Use --dataset-root to point to the correct location."
        )

    keep_train = process_split(train_dir, args.step, args.dry_run)
    keep_test = process_split(test_dir, args.step, args.dry_run)
    keep_names = keep_train | keep_test

    csv_path = args.csv or (dataset_root / "simulation.csv")
    filter_csv(csv_path, keep_names, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
