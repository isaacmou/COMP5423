"""FAISS read/write helpers that work with non-ASCII Windows paths."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


def write_faiss_index(index, path: Path) -> None:
    """Write a FAISS index, avoiding Windows path encoding failures."""
    import faiss

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        faiss.write_index(index, tmp_path)
        shutil.move(tmp_path, str(path))
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def read_faiss_index(path: Path):
    """Read a FAISS index, avoiding Windows path encoding failures."""
    import faiss

    path = Path(path)
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        shutil.copy2(path, tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
