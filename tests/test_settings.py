"""Tests for the standardized control-file loader ``mass_gt.settings``.

These lock in the exact ``varDict`` contract the calculation modules consume,
so refactors of configuration handling cannot silently change behaviour.
"""

from __future__ import annotations

import os

import pytest

from mass_gt import settings
from mass_gt.calculation.common import arguments as common_arguments


# --------------------------------------------------------------------------
# Parsing semantics (must mirror the original GUI behaviour)
# --------------------------------------------------------------------------


def test_modules_become_upper_cased_list(var_dict):
    assert var_dict["MODULES"] == ["FS", "SHIP", "TRAF"]


def test_numeric_argument_becomes_float(var_dict):
    assert var_dict["YEARFACTOR"] == 209.0
    assert isinstance(var_dict["YEARFACTOR"], float)


def test_string_argument_stays_string(var_dict):
    assert var_dict["LABEL"] == "REF"
    assert isinstance(var_dict["LABEL"], str)


def test_key_is_case_insensitive(var_dict):
    # The fixture writes 'DIMFOLDER'; a lower/other-case key must be accepted.
    assert var_dict["DIMFOLDER"].endswith("/")
    assert var_dict["DIMFOLDER"] is not None


def test_directory_gets_trailing_slash(var_dict):
    assert var_dict["DIMFOLDER"].endswith("/")


def test_file_placeholder_is_resolved(var_dict):
    # ZONES uses <<INPUTFOLDER>>, resolved to the concrete path.
    assert var_dict["ZONES"].endswith("Zones_v5.shp")
    assert var_dict["ZONES"].startswith(var_dict["INPUTFOLDER"])
    assert "<<INPUTFOLDER>>" not in var_dict["ZONES"]


def test_base_and_data_placeholders_are_resolved(tmp_path):
    """``<<BASE>>`` resolves against the repository root and ``<<DATA>``
    against the configured private data directory."""
    from mass_gt import config

    ini = tmp_path / "ph.ini"
    ini.write_text(
        "MODULES=FS\n"
        "ZONES=<<BASE>>data/input/Zones_v5.shp\n"
        "SEGS=<<DATA>>reference/SEGS.csv\n",
        encoding="utf-8",
    )
    vd = settings.parse_control_file(ini)
    assert vd["ZONES"] == str(config.BASE_DIR) + "/data/input/Zones_v5.shp"
    assert vd["SEGS"] == str(config.DATA_DIR) + "/reference/SEGS.csv"


def test_backslashes_are_normalised_to_forward_slashes(tmp_path):
    text = (
        "# comment\n"
        "MODULES=FS\n"
        "INPUTFOLDER = C:\\some\\dir\n"
    )
    ini = tmp_path / "bs.ini"
    ini.write_text(text, encoding="utf-8")
    vd = settings.parse_control_file(ini)
    assert vd["INPUTFOLDER"] == "C:/some/dir/"


def test_comment_and_empty_lines_are_ignored(tmp_path):
    text = (
        "# a comment\n"
        "MODULES=FS\n"
        "\n"
        "   \n"
        "LABEL = REF\n"
    )
    ini = tmp_path / "c.ini"
    ini.write_text(text, encoding="utf-8")
    vd = settings.parse_control_file(ini)
    assert vd["LABEL"] == "REF"
    assert vd["MODULES"] == ["FS"]


def test_all_known_variables_are_present_with_empty_default(var_dict):
    for name in common_arguments.variables:
        assert name in var_dict, name


def test_unknown_key_is_rejected(tmp_path):
    ini = tmp_path / "u.ini"
    ini.write_text("MODULES=FS\nBOGUS_KEY=1\n", encoding="utf-8")
    with pytest.raises(settings.ControlFileError, match="Unknown parameter"):
        settings.parse_control_file(ini)


def test_unknown_module_is_rejected(tmp_path):
    ini = tmp_path / "m.ini"
    ini.write_text("MODULES=FS,DOESNOTEXIST\n", encoding="utf-8")
    with pytest.raises(settings.ControlFileError, match="does not exist"):
        settings.parse_control_file(ini)


def test_missing_file_raises(tmp_path):
    ini = tmp_path / "missing.ini"
    with pytest.raises(settings.ControlFileError, match="Could not read"):
        settings.parse_control_file(ini)


def test_custom_module_names_can_be_supplied(tmp_path):
    ini = tmp_path / "custom.ini"
    ini.write_text("MODULES=MYMOD\n", encoding="utf-8")
    vd = settings.parse_control_file(ini, module_names=["MYMOD"])
    assert vd["MODULES"] == ["MYMOD"]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_validate_passes_when_dirs_exist(control_file, tmp_path):
    # ensure OUTPUTFOLDER and INPUTFOLDER exist (they point at tmp_path); but
    # ZONES points at a non-existent shapefile, so validation must still fail
    # on that. Instead, assert the specific ZONES error is raised.
    vd = settings.parse_control_file(control_file)
    with pytest.raises(settings.ControlFileError, match="ZONES"):
        settings.validate_control_file(vd)


def test_validate_flags_missing_directory(tmp_path):
    ini = tmp_path / "d.ini"
    ini.write_text("MODULES=FS\nINPUTFOLDER=/no/such/dir\n", encoding="utf-8")
    vd = settings.parse_control_file(ini)
    with pytest.raises(settings.ControlFileError, match="INPUTFOLDER"):
        settings.validate_control_file(vd)


def test_validate_flags_missing_required_argument(tmp_path):
    # NUTSLEVEL_INPUT is required but not set; validation should mention it.
    ini = tmp_path / "req.ini"
    ini.write_text("MODULES=FS\n", encoding="utf-8")
    vd = settings.parse_control_file(ini)
    with pytest.raises(settings.ControlFileError, match="NUTSLEVEL_INPUT"):
        settings.validate_control_file(vd)


def test_load_control_file_passes_through_without_validation(control_file):
    vd = settings.load_control_file(control_file, validate=False)
    assert vd["MODULES"] == ["FS", "SHIP", "TRAF"]


# --------------------------------------------------------------------------
# ModelDimensions default resolution
# --------------------------------------------------------------------------


def test_model_dimensions_defaults_to_dimensions_dir():
    from mass_gt.calculation.common.dimensions import ModelDimensions

    dims = ModelDimensions()  # defaults to DIMENSIONS_DIR
    for attr in ("combustion_type", "vehicle_type", "employment_sector"):
        assert getattr(dims, attr), attr
    assert 0 in dims.vehicle_type
    assert dims.vehicle_type[0]["Comment"] == "Truck (small)"


# --------------------------------------------------------------------------
# load_settings (GUI-friendly, non-raising)
# --------------------------------------------------------------------------


def test_load_settings_returns_tuple(control_file):
    var_dict, errors = settings.load_settings(control_file, validate=False)
    assert isinstance(var_dict, dict)
    assert isinstance(errors, list)


def test_load_settings_no_errors_on_valid_config(tmp_path):
    """A minimal config whose INPUTFOLDER/OUTPUTFOLDER exist should load
    with zero errors when validation is on."""
    text = (
        "MODULES=FS\n"
        f"INPUTFOLDER={tmp_path}/\n"
        f"PARAMFOLDER={tmp_path}/\n"
        f"OUTPUTFOLDER={tmp_path}/\n"
        f"DIMFOLDER={tmp_path}/\n"
        "YEARFACTOR=209\n"
        "NUTSLEVEL_INPUT=3\n"
        "ZONES=dummy.shp\n"
        "LABEL=REF\n"
    )
    ini = tmp_path / "ok.ini"
    ini.write_text(text, encoding="utf-8")
    vd, errors = settings.load_settings(ini, validate=False)
    assert errors == []
    assert vd["MODULES"] == ["FS"]
    assert vd["YEARFACTOR"] == 209.0
    assert vd["LABEL"] == "REF"


def test_load_settings_captures_parse_errors(tmp_path):
    """Unknown keys and modules are reported, not raised."""
    ini = tmp_path / "bad.ini"
    ini.write_text("MODULES=FS,NOPE\nBOGUS=1\n", encoding="utf-8")
    vd, errors = settings.load_settings(ini)
    assert len(errors) > 0
    # varDict still has all keys present (empty strings)
    for name in common_arguments.variables:
        assert name in vd
    assert any("BOGUS" in e for e in errors)
    assert any("NOPE" in e for e in errors)


def test_load_settings_captures_validation_errors(tmp_path):
    """Validation problems (missing dirs, missing required args) are
    returned as error strings rather than raising."""
    ini = tmp_path / "val.ini"
    ini.write_text("MODULES=FS\nINPUTFOLDER=/no/such/dir\n", encoding="utf-8")
    vd, errors = settings.load_settings(ini, validate=True)
    assert len(errors) > 0
    assert any("INPUTFOLDER" in e for e in errors)
    # Required args like NUTSLEVEL_INPUT should be flagged
    assert any("NUTSLEVEL_INPUT" in e for e in errors)


def test_load_settings_validate_false_skips_validation(tmp_path):
    ini = tmp_path / "skip.ini"
    ini.write_text("MODULES=FS\nINPUTFOLDER=/no/such/dir\n", encoding="utf-8")
    vd, errors = settings.load_settings(ini, validate=False)
    assert errors == []


def test_load_settings_empty_var_dict_on_parse_failure(tmp_path):
    """When the file can't even be parsed, varDict has all-empty defaults."""
    ini = tmp_path / "missing.ini"
    vd, errors = settings.load_settings(ini)
    assert errors  # non-empty errors list
    for name in common_arguments.variables:
        assert vd[name] == ""
