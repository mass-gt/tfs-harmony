"""Tests for the central path/configuration module ``mass_gt.config``."""

from __future__ import annotations

from pathlib import Path

from mass_gt import config


def test_base_dir_points_to_repository_root():
    # src/mass_gt/config.py -> parents[2] == repository root
    assert config.BASE_DIR.name == "tfs-harmony"
    assert (config.BASE_DIR / "pyproject.toml").is_file()


def test_dimensions_dir_points_to_reference_taxonomy():
    assert config.DIMENSIONS_DIR == config.BASE_DIR / "dimensions"
    assert config.DIMENSIONS_DIR.is_dir()


def test_data_dir_defaults_to_base_data():
    # With no MASS_GT_DATA_DIR override, DATA_DIR resolves under BASE_DIR.
    assert config.DATA_DIR == config.BASE_DIR / "data"


def test_standard_subdirectories_are_derived():
    assert config.INPUT_DIR == config.DATA_DIR / "input"
    assert config.OUTPUT_DIR == config.DATA_DIR / "output"


def test_all_paths_are_pathlib_paths():
    for name in ("BASE_DIR", "DATA_DIR", "INPUT_DIR", "OUTPUT_DIR", "DIMENSIONS_DIR"):
        value = getattr(config, name)
        assert isinstance(value, Path), name
        assert value.is_absolute(), name
