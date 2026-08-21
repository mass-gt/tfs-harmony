"""Shared pytest fixtures for the mass_gt test suite."""

from __future__ import annotations

import os
import textwrap

import pytest

from mass_gt import config

# Base control file used by settings tests. Paths are injected per-test via a
# real temp directory so placeholder resolution and validation behave as in a
# full scenario.
REQUIRED_DIMS = [
    "combustion_type.txt",
    "emission_type.txt",
    "employment_sector.txt",
    "flow_type.txt",
    "logistic_segment.txt",
    "municipality.txt",
    "nstr.txt",
    "shipment_size.txt",
    "vehicle_type.txt",
]


@pytest.fixture
def control_file(tmp_path):
    """Write a self-contained control file into a temp dir and return its path.

    The control file references real folders (the repo's ``dimensions/`` for
    DIMFOLDER and ``tmp_path`` for the others) so that both placeholder
    resolution and optional validation can be exercised.
    """
    dims = config.DIMENSIONS_DIR
    text = textwrap.dedent(
        """
        # MASS-GT control file (test)
        MODULES = FS, SHIP, TRAF
        INPUTFOLDER  = {tmp}
        PARAMFOLDER  = {tmp}
        OUTPUTFOLDER = {tmp}
        DIMFOLDER    = {dims}/
        YEARFACTOR   = 209
        NUTSLEVEL_INPUT = 3
        ZONES = <<INPUTFOLDER>>Zones_v5.shp
        LABEL = REF
        """
    ).format(tmp=str(tmp_path).replace("\\", "/"), dims=str(dims).replace("\\", "/"))
    ini_path = tmp_path / "Run_REF.ini"
    ini_path.write_text(text, encoding="utf-8")
    return ini_path


@pytest.fixture
def var_dict(control_file):
    """Return a parsed (unvalidated) varDict for the fixture control file."""
    from mass_gt import settings
    return settings.parse_control_file(control_file)



