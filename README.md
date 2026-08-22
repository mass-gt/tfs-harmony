# MASS-GT: Tactical Freight Simulator for Urban Logistics (tfs-harmony)

The **Tactical Freight Simulator (TFS)** is the freight-modelling core of
**MASS-GT** — an open, empirically-calibrated, agent-based simulation platform
for urban freight and logistics policy analysis developed at
[Delft University of Technology](https://www.tudelft.nl/transport/onderzoeksthemas/goederenvervoer-logistiek/sleutelprojecten/mass-gt).

`tfs-harmony` simulates the full urban-freight chain at truck/tour level:
**freight generation → distribution → shipping → tour formation → parcel demand & scheduling → service provision → traffic assignment → emissions output**.

## Repository context

MASS-GT exists in three GitHub repositories sharing an identical architecture:

| Repo                | Focus                              |
|---------------------|------------------------------------|
| `tfs-harmony` *(this)* | Core tactical freight simulator |
| `mass-gt-emotion`   | E-bike policy extensions           |
| `mass-gt-safety`    | Safety analysis extensions         |

Branches: `main` (upstream baseline), `restructure-layout` (active), `prototype-2023-08` (HARMONY), `prototype-2025-01` (MRDH Rotterdam).

## Features
- Modular pipeline: FS, SIF, SHIP, TOUR, PARCEL_DMND, PARCEL_SCHD, SERVICE, TRAF, OUTP
- Headless CLI with progress reporting; Tkinter GUI for interactive runs.
- Data-driven `.ini` control files with `<<INPUTFOLDER>>` / `<<PARAMFOLDER>>` placeholder substitution.

## 15-minute setup

### Prerequisites
- **Python 3.11**
- **Git**
- Private model input data — see [Data](#data-configuration).

### 1. Clone
```bash
git clone https://github.com/mass-gt/tfs-harmony.git
cd tfs-harmony
```

### 2. Virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate          # Windows
```

### 3. Install
```bash
pip install -e .[dev]
```

## Data configuration
MASS-GT requires **private input data** (zones, skims, firms, etc.) **not** in this repo.

1. `cp .env.example .env`
2. Set `MASS_GT_DATA_DIR` to your local data directory containing `input/`, `output/`, and `reference/dimensions/`.
3. Place your `.ini` (see `run/example.ini`) in `data/input/`.

Without `.env`, paths default under the repo's `data/`. Unit tests need no data.

## Running the model

### CLI (headless)
```bash
python run_scenario.py --config data/input/Run_REF.ini
python run_scenario.py --config /path/to/scenario.ini --quiet
```
Flags: `--config` (default `data/input/Run_REF.ini`), `--quiet` (silent, `root=None`).

### GUI (interactive)
```bash
python -m mass_gt.tfs
```
Requires ``tkinter`` and a display. On headless machines the `mass_gt.tfs` module
can still be imported (e.g. for testing); only the GUI launch is skipped.

### As a library
```python
from mass_gt import settings
from mass_gt.calculation.fs import module_fs

# Parse + validate (raises ControlFileError on problems)
vd = settings.parse_control_file("data/input/Run_REF.ini")
settings.validate_control_file(vd)
result = module_fs.actually_run_module(root=None, varDict=vd, dims=None)

# Or use the non-raising loader (returns (varDict, errors) for GUIs):
vd, errors = settings.load_settings("data/input/Run_REF.ini", validate=True)
if not errors:
    ...
```

## Testing
```bash
pytest            # 29 tests, no data required
```

## Documentation
Full Sphinx docs build locally:
```bash
sphinx-build -b html docs/source docs/build/html
```
Rendered guide: [tfs-harmony.readthedocs.io](https://tfs-harmony.readthedocs.io/)

See `docs/source/index.rst` for the documentation table of contents.

## Repository structure
```
tfs-harmony/
├── pyproject.toml       # packaging (Hatchling) & tool config
├── run_scenario.py      # CLI entry point
├── run/example.ini      # documented control-file template
├── data/                # git-ignored; .env points here for your data
│   └── reference/dimensions/   # taxonomy reference files
├── docs/                # Sphinx documentation
├── src/mass_gt/         # installable package
│   ├── __init__.py      # __version__ = "3.1.0"
│   ├── config.py        # BASE_DIR, DATA_DIR, paths (via .env)
│   ├── settings.py      # .ini → varDict parser & validator (load_settings)
│   ├── tfs.py           # GUI entry point (headless-safe import)
│   └── calculation/     # model pipeline modules (common, fs, sif, ship, …)
└── tests/               # pytest suite (repository root)
```

## License
GNU General Public License v2.0 — see [LICENSE](LICENSE).

## References
- de Bok et al. (2025). "MASS-GT: an empirical model for the simulation of freight policies." *Simulation Modelling Practice and Theory*, 142. https://doi.org/10.1016/j.simpat.2025.103140
- de Bok et al. (2024). "Micro-hub scenarios for city logistics in Rotterdam." *Res. Transp. Bus. & Manag.*, 56. https://doi.org/10.1016/j.rtbm.2024.101186
- Thoen et al. (2020). "Descriptive modeling of freight tour formation." *Transp. Res. Part E*, 140. https://doi.org/10.1016/j.tre.2020.101989
- de Bok & Tavasszy (2018). "An empirical agent-based simulation system for urban goods transport (MASS-GT)." *Procedia CS*, 130. https://doi.org/10.1016/j.procs.2018.04.021
`