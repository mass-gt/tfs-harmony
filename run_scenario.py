#!/usr/bin/env python3
"""Headless command-line entry point for the TFS-Harmony (MASS-GT) model.

Runs the modules listed in ``MODULES`` of a control (.ini) file without a
GUI — a lightweight console reporter is passed to the modules instead of the
tkinter ``Root``, so the simulation can be scripted, batched and run in CI.

Usage::

    python run_scenario.py --config data/input/Run_REF.ini

The GUI (``python -m mass_gt.tfs``) remains available as an alternative
interactive entry point.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict

from mass_gt.calculation.common.dimensions import ModelDimensions
from mass_gt.settings import (
    ControlFileError,
    MODULE_NAMES,
    load_control_file,
)

# Modules that make up the simulation pipeline (order matters).
# Each module exposes ``actually_run_module(root, varDict, dims)`` and returns
# ``[0, [0, 0]]`` on success or ``[1, [exception_type, traceback]]`` on error.
_MODULES = [
    ("FS", "Firm Synthesis", "mass_gt.calculation.fs.module_fs"),
    ("SIF", "Spatial Interaction Freight", "mass_gt.calculation.sif.module_sif"),
    ("SHIP", "Shipment Synthesizer", "mass_gt.calculation.ship.module_ship"),
    ("TOUR", "Tour Formation", "mass_gt.calculation.tour.module_tour"),
    ("PARCEL_DMND", "Parcel Demand", "mass_gt.calculation.parcel_dmnd.module_parcel_dmnd"),
    ("PARCEL_SCHD", "Parcel Scheduling", "mass_gt.calculation.parcel_schd.module_parcel_schd"),
    ("SERVICE", "Vans Service/Construction", "mass_gt.calculation.service.module_service"),
    ("TRAF", "Traffic Assignment", "mass_gt.calculation.traf.module_traf"),
    ("OUTP", "Output Indicators", "mass_gt.calculation.outp.module_outp"),
]


class ConsoleReporter:
    """Minimal headless stand-in for the GUI ``Root``.

    The calculation modules only use ``update_statusbar`` and the
    ``progressBar`` mapping when ``root is not None``; this object provides
    those two interface points and prints progress to stdout.
    """

    def __init__(self) -> None:
        self.progressBar: Dict[str, int] = {"value": 0}

    def update_statusbar(self, text: str) -> None:
        sys.stdout.write(f"[MASS-GT] {text}\n")
        sys.stdout.flush()


def _default_config_path() -> str:
    """Return the default control file path (repo-relative, cwd-independent)."""
    default = Path("data/input/Run_REF.ini")
    repo_default = Path(__file__).resolve().parent / default
    return str(repo_default if repo_default.exists() else default)


def run_scenario(config_path: str, root: Any = None) -> None:
    """Load ``config_path`` and run every selected module in pipeline order.

    ``root`` may be ``None`` (fully silent) or a lightweight reporter such as
    :class:`ConsoleReporter`; the calculation modules handle both.
    """
    varDict = load_control_file(config_path, module_names=MODULE_NAMES)
    reporter: Any = root if root is not None else ConsoleReporter()

    selected = [mod for mod in varDict.get("MODULES", []) if mod]
    if not selected:
        raise ControlFileError(
            f"Control file '{config_path}' does not select any modules "
            "(MODULES is empty).")

    logger = logging.getLogger("tfs")
    dims = ModelDimensions(varDict["DIMFOLDER"])

    for abbr, name, module_path in _MODULES:
        if abbr not in selected:
            continue
        reporter.update_statusbar(f"Running {name} ({abbr})...")
        module = __import__(module_path, fromlist=["actually_run_module"])
        result = module.actually_run_module(reporter, varDict, dims)
        if result[0] == 1:
            raise RuntimeError(
                f"Error in {name} module!\n"
                f"{result[1][0]}\n{result[1][1]}")
        reporter.update_statusbar(f"{abbr}: Done")

    reporter.update_statusbar("Finished all calculations.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_scenario",
        description="Run a MASS-GT (tfs-harmony) scenario headlessly from a control file.",
    )
    parser.add_argument(
        "--config",
        default=_default_config_path(),
        help="Path to the .ini control file "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run without console progress output (root=None).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s]: %(message)s",
    )

    try:
        run_scenario(args.config, root=(None if args.quiet else ConsoleReporter()))
    except ControlFileError as exc:
        sys.stderr.write(f"Configuration error:\n{exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        sys.stderr.write(f"Simulation failed:\n{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())