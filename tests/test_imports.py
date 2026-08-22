"""Smoke test: every package module must be importable.

The calculation engine depends on heavyweight libraries (numba, pyshp, ...)
that are easy to forget in ``pyproject.toml`` because the unit tests never
*run* the model.  This test imports every calculation module so that a
missing dependency fails here -- and in CI -- instead of at the first model
run on a researcher's machine.
"""

from __future__ import annotations

import importlib

import pytest

CALCULATION_MODULES = [
    "mass_gt.calculation.common.arguments",
    "mass_gt.calculation.common.dimensions",
    "mass_gt.calculation.common.io",
    "mass_gt.calculation.fs.module_fs",
    "mass_gt.calculation.sif.module_sif",
    "mass_gt.calculation.ship.module_ship",
    "mass_gt.calculation.tour.module_tour",
    "mass_gt.calculation.parcel_dmnd.module_parcel_dmnd",
    "mass_gt.calculation.parcel_schd.module_parcel_schd",
    "mass_gt.calculation.service.module_service",
    "mass_gt.calculation.traf.module_traf",
    "mass_gt.calculation.outp.module_outp",
]

TOP_LEVEL_MODULES = [
    "mass_gt.config",
    "mass_gt.settings",
    "mass_gt.support",
    "mass_gt.tfs",  # must import even without tkinter/display
]


@pytest.mark.parametrize("module_name", CALCULATION_MODULES + TOP_LEVEL_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_third_party_import_surface() -> None:
    """The heavy third-party backends used by the engine must be present."""
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import numba  # noqa: F401
    import shapefile  # pyshp  # noqa: F401
    import shapely  # noqa: F401
    import dotenv  # python-dotenv  # noqa: F401