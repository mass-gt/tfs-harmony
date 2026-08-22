Configuration & Parameters Reference
======================================

Control files
-------------

MASS-GT is driven by a plain-text control file (`.ini`) with `KEY = value`
lines. Placeholders in file arguments are substituted at parse time:

- `<<INPUTFOLDER>>`, `<<OUTPUTFOLDER>>`, `<<PARAMFOLDER>>`, `<<DIMFOLDER>>` —
  resolved to the current value of that directory parameter
- `<<BASE>>` — resolved to the repository root (`mass_gt.config.BASE_DIR`)
- `<<DATA>>` — resolved to the private data directory
  (`mass_gt.config.DATA_DIR`, set via `MASS_GT_DATA_DIR` in your `.env`)

Example::

   ZONES      = <<INPUTFOLDER>>Zones_v5.shp
   PARAMS_TOD = <<DATA>>params/tod.csv

Path separators are normalised to forward slashes, and all directory paths
receive a trailing `/`.

Module selection
-----------------

The `MODULES` key selects which pipeline stages to run, as a comma-separated
list. The canonical pipeline order is:

.. code-block::

   FS, SIF, SHIP, TOUR, PARCEL_DMND, PARCEL_SCHD, SERVICE, TRAF, OUTP

Core parameters
---------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - `MODULES`
     - list
     - Comma-separated module names to run
   * - `INPUTFOLDER`
     - directory
     - Folder containing input data (zones, skims, etc.)
   * - `PARAMFOLDER`
     - directory
     - Folder containing parameter/calibration files
   * - `OUTPUTFOLDER`
     - directory
     - Folder where results are written
   * - `DIMFOLDER`
     - directory
     - Folder containing dimension taxonomy files
   * - `ZONES`
     - file
     - Zones shapefile path
   * - `YEARFACTOR`
     - numeric
     - Annualisation factor (year → day or week)
   * - `NUTSLEVEL_INPUT`
     - numeric
     - NUTS level for spatial aggregation (1–3)
   * - `LABEL`
     - string
     - Scenario label for output naming

For the complete parameter list with categories (numeric, directories, files,
obsolete, optional), see the source: `src/mass_gt/calculation/common/arguments.py`.
