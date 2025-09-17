import contextlib
from typing import ContextManager
import pycolmap
from pathlib import Path


@contextlib.contextmanager
def open_colmap_database(database_path: Path) -> ContextManager[pycolmap.Database]:
    # In preparation for the context support in the future pycolmap 3.13 release
    if hasattr(pycolmap.Database, "open"):
        with pycolmap.Database.open(db_path) as db:
            yield db
    else:
        db = pycolmap.Database(db_path)
        try:
            yield db
        finally:
            db.close()
