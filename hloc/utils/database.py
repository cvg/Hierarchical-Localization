import contextlib
from pathlib import Path
from typing import ContextManager

import pycolmap


@contextlib.contextmanager
def open_colmap_database(database_path: Path) -> ContextManager[pycolmap.Database]:
    # In preparation for the context support in the future pycolmap >= 3.13 release
    if isinstance(pycolmap.Database.__dict__.get("open"), (staticmethod, classmethod)):
        with pycolmap.Database.open(database_path) as db:
            yield db
    else:
        db = pycolmap.Database(database_path)
        try:
            yield db
        finally:
            db.close()
