"""Central configuration and filesystem paths for MASS-GT (tfs-harmony).

All paths used by the model should originate from this module so that the
repository layout, data locations and run outputs stay consistent and easy to
configure. Environment variables are loaded from the repository-local ``.env``
file (see ``.env.example``) using ``python-dotenv``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _resolve_data_dir(base_dir: Path, env_value: str | None) -> Path:
    """Resolve ``MASS_GT_DATA_DIR`` relative to the repository root.

    An absolute value is kept as-is; a relative value is interpreted against
    ``base_dir`` (so ``./data`` always means ``base_dir/data`` regardless of the
    current working directory).
    """
    if env_value:
        candidate = Path(env_value).expanduser()
        return candidate if candidate.is_absolute() else (base_dir / candidate)
    return base_dir / "data"


# Repository root: ..\.. of ``src/mass_gt/config.py``.
BASE_DIR = Path(__file__).resolve().parents[2]

# Load local environment (.env is git-ignored; copy from .env.example).
load_dotenv(BASE_DIR / ".env")

# Root directory holding (private) model data: input, output, parameters.
DATA_DIR = _resolve_data_dir(BASE_DIR, os.getenv("MASS_GT_DATA_DIR")).resolve()

# Standard data sub-directories.
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Reference dimensions taxonomy lives alongside the source code.
DIMENSIONS_DIR = BASE_DIR / "dimensions"