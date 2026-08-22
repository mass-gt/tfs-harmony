# MASS-GT: Tactical Freight Simulator (tfs-harmony)

The **Tactical Freight Simulator (TFS)** is the freight-modelling core of
**MASS-GT** (Multi-Agent Simulation System for Goods Transport), an open,
empirically calibrated, agent-based simulation platform for urban freight and
logistics policy analysis developed at
[Delft University of Technology](https://www.tudelft.nl/transport/onderzoeksthemas/goederenvervoer-logistiek/sleutelprojecten/mass-gt).
The core simulator was developed in the EU Horizon 2020 HARMONY project.

`tfs-harmony` simulates the full urban-freight chain at firm, shipment and
vehicle/tour level through nine sequential submodules:

FS (Firm Synthesis) → SIF (Spatial Interaction Freight) →
SHIP (Shipment Synthesizer) → TOUR (Tour Formation) →
PARCEL_DMND (Parcel Demand) → PARCEL_SCHD (Parcel Scheduling) →
SERVICE (Vans Service/Construction) → TRAF (Traffic Assignment) →
OUTP (Output Indicators)

## Project context

MASS-GT is developed at Delft University of Technology together with technical
partner [Significance](https://www.significance.nl), and evolves through
successive research projects, including the EU Horizon 2020 projects
[HARMONY](https://harmony-h2020.eu/), [LEAD](https://leadproject.eu/) and
[URBANE](https://urbane-horizoneurope.eu/). The model also serves as donor
model for the Dutch strategic freight transport model BasGoed
(Rijkswaterstaat).

This repository contains the Tactical Freight Simulator as developed in the
HARMONY project. Available branches:

- `main` — upstream baseline
- `restructure-layout` — standardized packaging and configuration layer (active)
- `prototype-2023-08` — code base resulting from the HARMONY project
- `prototype-2025-01` — refactored for the MRDH base year data (Gemeente Rotterdam)

For access to the code or the operational input dataset, send your GitHub
account name or e-mail address to thoen@significance.nl or
m.a.debok@tudelft.nl.

## Getting started

### Prerequisites

- Python 3.9 or newer (3.11 recommended)
- Git
- The private model input dataset — see [Data setup](#data-setup)

### 1. Clone

```bash
git clone https://github.com/mass-gt/tfs-harmony.git
cd tfs-harmony
```

### 2. Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install

```bash
pip install -e .[dev]
```

## Data setup

MASS-GT requires an operational input dataset (zones, skims, firms, networks,
parameters) that is **not** part of this repository. Request access from the
model maintainers (thoen@significance.nl or m.a.debok@tudelft.nl).
The dataset of the implementation for Zuid-Holland, the Netherlands is
available via Sebastiaan Thoen (`@sebastiaanth` on GitHub).

1. Copy `.env.example` to `.env`.
2. Set `MASS_GT_DATA_DIR` to your local data directory containing `input/`,
   `output/` and `reference/dimensions/`, e.g.:
   `MASS_GT_DATA_DIR="C:/path/to/your/data"`
3. Place your scenario control file (see `run/example.ini`) in `data/input/`.

Without a `.env` file, paths default to locations under the repository's
`data/` folder. The unit tests need no data.

## Running the model

### CLI (headless, recommended for batch runs and servers)

```bash
python run_scenario.py --config data/input/Run_REF.ini
python run_scenario.py --config /path/to/scenario.ini --quiet
```

Flags: `--config` (default `data/input/Run_REF.ini`), `--quiet` (no console
progress output).

### GUI (interactive)

```bash
python -m mass_gt.tfs
```

Requires tkinter and a display. On headless machines `import mass_gt.tfs`
still works (for testing or library use); only launching the GUI needs a
display.

### As a library

```python
from mass_gt import settings
from mass_gt.calculation.common.dimensions import ModelDimensions
from mass_gt.calculation.fs import module_fs

var_dict, errors = settings.load_settings("data/input/Run_REF.ini")
if not errors:
    dims = ModelDimensions(var_dict["DIMFOLDER"])
    result = module_fs.actually_run_module(
        root=None, varDict=var_dict, dims=dims)
else:
    print("Configuration errors:", errors)
```

See `run_scenario.py` for the full module-execution pattern.

Note: if you run the model from the Spyder IDE, set
*Tools → Preferences → Run → Console* to *Execute in an external system
terminal*; the tour formation and traffic assignment modules use process
parallelisation and require it.

## Testing

```bash
pytest            # 30 tests; runs without model data
```

## Documentation

Build the Sphinx documentation locally:

```bash
sphinx-build -b html docs/source docs/build/html
```

Then open `docs/build/html/index.html`. Rendered guide:
[tfs-harmony.readthedocs.io](https://tfs-harmony.readthedocs.io/).
Maintainers should also read `docs/source/maintainers.rst`, which documents
the architecture decisions and how to port this structure to the sibling
repositories.

## Repository structure

```
tfs-harmony/
├── .github/workflows/ci.yml   # CI pipeline (pytest on push/PR)
├── dimensions/                # Reference taxonomy files (.txt)
├── docs/                      # Sphinx documentation source
├── run/
│   └── example.ini            # Documented control-file template
├── src/mass_gt/               # Installable package (src layout)
│   ├── config.py              # Path resolution via .env
│   ├── settings.py            # Control-file parser & validator
│   ├── support.py             # Logging utilities
│   ├── tfs.py                 # GUI entry point (headless-safe import)
│   └── calculation/           # Simulation submodules (FS ... OUTP)
├── tests/                     # Pytest suite
├── data/                      # Local model data (git-ignored)
├── .env.example               # Environment variable template
├── CHANGELOG.md               # Notable changes
├── pyproject.toml             # Packaging (Hatchling) and tool config
└── run_scenario.py            # Headless CLI entry point
```

## License

GNU General Public License v2.0 or later — see [LICENSE](LICENSE).

## References

If you use MASS-GT in academic research, please cite:

- de Bok et al. (2025). "MASS-GT: an empirical model for the simulation of
  freight policies." *Simulation Modelling Practice and Theory*, 142.
  https://doi.org/10.1016/j.simpat.2025.103140
- de Bok et al. (2024). "A simulation study of the impacts of micro-hub
  scenarios for city logistics in Rotterdam."
  *Research in Transportation Business & Management*, 56.
  https://doi.org/10.1016/j.rtbm.2024.101186
- Thoen et al. (2020). "Descriptive modeling of freight tour formation: A
  shipment-based approach." *Transportation Research Part E*, 140.
  https://doi.org/10.1016/j.tre.2020.101989
- de Bok & Tavasszy (2018). "An empirical agent-based simulation system for
  urban goods transport (MASS-GT)." *Procedia Computer Science*, 130.
  https://doi.org/10.1016/j.procs.2018.04.021