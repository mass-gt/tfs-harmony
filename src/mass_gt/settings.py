"""Standardized control-file (.ini) parsing for MASS-GT (tfs-harmony).

Parses a MASS-GT control file into the exact ``varDict`` string-dictionary
contract that all calculation modules expect, replacing the ad-hoc parsing
embedded in the GUI (``tfs.py``) with a reusable, documented loader.

The parsing behaviour deliberately mirrors the original code so that existing
control files keep working unchanged:

* ``#`` comment lines and lines without ``=`` are ignored;
* arguments are written ``KEY = value`` (``KEY`` is case-insensitive);
* numeric arguments are converted to ``float``;
* ``MODULES`` becomes a list of upper-cased module abbreviations;
* ``<<DIRECTORY>>`` placeholders in file paths are resolved;
* directory and file paths are validated to exist (when validation is on);
* the argument lists in ``calculation.common.arguments`` are the single
  source of truth for which keys are valid.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from mass_gt.calculation.common import arguments as common_arguments
from mass_gt.config import BASE_DIR

# Canonical, ordered list of runnable modules (mirrors the simulation pipeline).
MODULE_NAMES = [
    'FS',
    'SIF',
    'SHIP',
    'TOUR',
    'PARCEL_DMND',
    'PARCEL_SCHD',
    'SERVICE',
    'TRAF',
    'OUTP',
]

PathLike = Union[str, os.PathLike]


class ControlFileError(ValueError):
    """Raised when a control file cannot be parsed or validated."""


def _strip(value: str) -> str:
    """Strip trailing newline and surrounding spaces/tabs (like the original)."""
    value = value.rstrip('\n').rstrip('\r')
    while value[:1] in (' ', '\t'):
        value = value[1:]
    while value[-1:] in (' ', '\t'):
        value = value[:-1]
    return value


def _resolve_placeholders(varDict: Dict[str, Any]) -> None:
    """Resolve ``<<DIRECTORY>>`` and ``<<BASE>>`` placeholders in file paths in place.

    ``<<DIRECTORY>>`` is resolved when ``DIRECTORY`` is a key in
    ``common_arguments.directories`` (INPUTFOLDER, OUTPUTFOLDER, etc.).
    ``<<BASE>>`` is resolved to the repository root (``BASE_DIR``).
    """
    for variableName in common_arguments.variables:
        if variableName not in common_arguments.files:
            continue
        tmp = varDict[variableName].split("<<")
        if len(tmp) > 1:
            tmp = tmp[1].split(">>")
            placeholder = tmp[0]
            if placeholder == "BASE":
                varDict[variableName] = str(BASE_DIR) + "/" + tmp[1]
            elif placeholder in common_arguments.directories:
                varDict[variableName] = varDict[placeholder] + tmp[1]
def parse_control_file(
    control_file_path: PathLike,
    module_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Parse a control file into a ``varDict`` (parsing only, no validation).

    Returns the same ``varDict`` dictionary the calculation modules consume:
    strings for paths/options, ``float`` for numeric arguments, and a list of
    upper-cased module abbreviations under ``'MODULES'``.
    """
    module_names = MODULE_NAMES if module_names is None else module_names

    varDict: Dict[str, Any] = dict(
        (name, "") for name in common_arguments.variables
    )
    errors: List[str] = []
    run = True

    path = os.fspath(control_file_path)
    try:
        with open(path, 'r') as fh:
            for line in fh.readlines():
                line = line.rstrip('\n').rstrip('\r')
                if not line or line[0] == '#':
                    continue
                if '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = _strip(key)
                value = _strip(value)
                variableName = key.upper()

                if variableName not in common_arguments.variables:
                    errors.append(
                        f"Unknown parameter in control file: {key}")
                    run = False
                    continue

                if variableName in common_arguments.numeric:
                    cleaned = value.replace("'", "").replace('"', "").replace('\n', "")
                    try:
                        varDict[variableName] = float(cleaned)
                    except ValueError:
                        if not (cleaned == '' and
                                variableName in common_arguments.optional):
                            varDict[variableName] = cleaned
                            errors.append(
                                f"Fill in a numeric value for '{variableName}', "
                                f"could not convert following value to a number: "
                                f"'{cleaned}'.")
                            run = False

                elif variableName in common_arguments.modules:
                    cleaned = (
                        value.replace("'", "")
                        .replace('"', "")
                        .replace('\n', "")
                        .replace(' ', '')
                    )
                    parsed = [x.upper() for x in cleaned.split(',')]
                    varDict[variableName] = parsed
                    for mod in parsed:
                        if mod not in module_names:
                            errors.append(f"Module '{mod}' does not exist.")
                            run = False

                else:  # string argument / path
                    cleaned = (
                        value.replace(os.sep, '/')
                        .replace("'", "")
                        .replace('"', "")
                        .replace('\n', "")
                    )
                    varDict[variableName] = cleaned
                    if variableName in common_arguments.directories:
                        if varDict[variableName][-1:] != '/':
                            varDict[variableName] = varDict[variableName] + '/'
    except OSError as exc:
        raise ControlFileError(
            f"Could not read control file '{path}': {exc}") from exc

    _resolve_placeholders(varDict)

    if not run:
        raise ControlFileError(
            "\n".join(errors) or "Control file parsing failed.")

    return varDict


def validate_control_file(varDict: Dict[str, Any]) -> None:
    """Validate an already-parsed ``varDict`` (existence + required args).

    Raises :class:`ControlFileError` listing all problems if the configuration
    is not runnable.
    """
    errors: List[str] = []

    for variableName in common_arguments.variables:
        value = varDict[variableName]
        if value == "":
            continue
        if variableName in common_arguments.directories:
            if not os.path.isdir(value):
                errors.append(
                    f"The folder for parameter '{variableName}' does not exist: "
                    f"'{value}'.")
        elif variableName in common_arguments.files:
            if not os.path.isfile(value):
                errors.append(
                    f"The file for parameter '{variableName}' does not exist: "
                    f"'{value}'.")

    for variableName in common_arguments.variables:
        if (varDict[variableName] == ""
                and variableName not in common_arguments.optional):
            errors.append(
                f"Warning, no value given for parameter '{variableName}' "
                "in the control file.")

    if errors:
        raise ControlFileError("\n".join(errors))


def load_control_file(
    control_file_path: PathLike,
    module_names: Optional[List[str]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Parse (and optionally validate) a control file into a ``varDict``.

    This is the recommended entry point for running a scenario. When
    ``validate`` is True (default) directories/files are checked to exist and
    missing non-optional arguments are flagged.
    """
    varDict = parse_control_file(control_file_path, module_names=module_names)
    if validate:
        validate_control_file(varDict)
    return varDict