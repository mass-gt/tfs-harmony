# Changelog

All notable changes to **tfs-harmony** are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/); versioning
follows the package version in `pyproject.toml`.

## [Unreleased] — `restructure-layout`

### Added
- Standardized configuration layer `mass_gt.config`: all filesystem paths come
  from one module, driven by a git-ignored `.env` (see `.env.example`).
- `mass_gt.settings`: single, tested control-file (`.ini`) loader with four
  entry points — `parse_control_file`, `validate_control_file`,
  `load_control_file` (raising) and `load_settings` (returns
  `(varDict, errors)`, used by the GUI).
- `<<BASE>>` (repository root) and `<<DATA>>` (`MASS_GT_DATA_DIR`) path
  placeholders alongside the directory-parameter placeholders.
- Headless CLI entry point `run_scenario.py` with `--config` / `--quiet`.
- Pytest suite (30 tests) locking the `varDict` contract; GitHub Actions CI.
- Sphinx documentation (quickstart, data guide, configuration reference,
  pipeline, methodology, API, maintainer guide).

### Changed
- Package moved to a `src/` layout and is installable:
  `pip install -e .[dev]`.
- The GUI (`mass_gt.tfs`) now delegates control-file loading to
  `mass_gt.settings.load_settings`, giving CLI and GUI identical parsing
  semantics from one code path.
- `tkinter` imports are wrapped: `import mass_gt.tfs` succeeds on headless
  machines (CI, compute servers); only the GUI launch requires a display.

### Fixed
- Crash when removing logger handlers while the output folder was invalid.
- PEP 8 violations in the configuration layer; duplicated parser removed.

### Unchanged (by design)
- All calculation / stochastic model code under `mass_gt.calculation` — the
  validated prototype models are byte-identical apart from import lines.
- The `varDict` configuration contract consumed by the modules.
