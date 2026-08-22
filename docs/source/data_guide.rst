Data Guide
==========

This repository does **not** contain the private input data required to run
a full simulation. The data is supplied separately.

Local data directory
--------------------

Set the environment variable `MASS_GT_DATA_DIR` to the root of your local
data tree:

.. code-block:: bash

   export MASS_GT_DATA_DIR=/path/to/your/mass-gt-data

The expected layout is:

.. code-block::

   mass-gt-data/          ← MASS_GT_DATA_DIR
   ├── input/             ← INPUTFOLDER (zones, skims, firms, matrices)
   ├── param/             ← PARAMFOLDER (calibration params)
   ├── output/            ← OUTPUTFOLDER (generated results)
   └── reference/
       └── dimensions/    ← DIMFOLDER (taxonomy files: vehicle_type, etc.)

Create a local `.env` file from `.env.example`:

.. code-block:: bash

   cp .env.example .env
   # edit .env:
   MASS_GT_DATA_DIR=/path/to/your/mass-gt-data

Without a `.env`, paths default to `<repo>/data/`.

Reference dimensions
--------------------

The taxonomy files used by `ModelDimensions` live under
`dimensions/` in the repository. These define:

- `vehicle_type` — codes for truck, van, bike, etc.
- `combustion_type` — diesel, petrol, electric, etc.
- `employment_sector` — SBI sector classification
- `flow_type` — freight / parcel / service
- `emission_type` — emission factor categories
- `logistic_segment` — wholesale, retail, industry, etc.
- `municipality`, `nstr`, `shipment_size`, `flow_type`

Run control file
----------------

The control file (`.ini`) specifies `MODULES`, directories, numeric
parameters, and file paths. See `run/example.ini` for a fully documented
template.
