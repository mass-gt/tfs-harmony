Maintainer Guide & Handover
===========================

This page is for researchers and developers who maintain MASS-GT after the
``restructure-layout`` restructuring. It explains *what* changed relative to
the original prototype code, *why*, and how to repeat the same modernisation
on other MASS-GT repositories (such as the BasGoed logistics module) and
future project forks.

What changed on ``restructure-layout``
--------------------------------------

Nothing in the mathematical models was touched. The restructuring is purely
the surrounding engineering layer:

.. list-table::
   :widths: 40 40 20
   :header-rows: 1

   * - Before (prototype branches)
     - After (``restructure-layout``)
     - Where
   * - Hardcoded paths / scattered env vars
     - Central :mod:`mass_gt.config`, driven by a git-ignored ``.env``
     - ``src/mass_gt/config.py``
   * - ``.ini`` parser duplicated in GUI and scripts
     - One tested loader shared by CLI and GUI
     - ``src/mass_gt/settings.py``
   * - GUI-only entry point
     - Headless CLI plus unchanged GUI
     - ``run_scenario.py``, ``mass_gt.tfs``
   * - No automated tests
     - 30-test pytest suite locking the config contract
     - ``tests/``
   * - No packaging
     - Installable wheel, editable install
     - ``pyproject.toml`` (Hatchling, src layout)
   * - No CI, no docs
     - GitHub Actions; Sphinx site on Read the Docs
     - ``.github/workflows/``, ``docs/``

The frozen contract
-------------------

Two things are guaranteed stable and must stay that way:

1. **Calculation modules** (``mass_gt.calculation.*``) contain the stochastic
   models (FRATAR scaling, tour formation, assignment, emissions). They are
   byte-identical to the validated prototype apart from import lines.
2. **``varDict``**: the string dictionary consumed by every module. Its keys
   are exactly ``arguments.variables`` in
   ``src/mass_gt/calculation/common/arguments.py``; numeric entries are
   ``float``, ``MODULES`` is a list, everything else is ``str``.

Before *and* after any refactor, reproduce golden-file baseline outputs to
prove behaviour is unchanged.

Configuration loading
---------------------

All control-file handling lives in :mod:`mass_gt.settings`:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Use when
   * - ``parse_control_file(path)``
     - You want the raw ``varDict``; raises ``ControlFileError`` on bad input.
   * - ``validate_control_file(varDict)``
     - Existence checks for directories/files + required-argument check.
   * - ``load_control_file(path)``
     - Parse + validate, raising. Recommended for scripts and CI.
   * - ``load_settings(path)``
     - Parse (+ optional validate) returning ``(varDict, errors)`` instead of
       raising. Recommended inside interactive tools such as the GUI.

File arguments may use placeholders: ``<<DIRECTORY>>`` for any configured
directory parameter, ``<<BASE>>`` for the repository root, ``<<DATA>>`` for
the private data directory (``MASS_GT_DATA_DIR``).

Entry points
------------

.. code-block:: bash

   python run_scenario.py --config data/input/Run_REF.ini   # headless
   python -m mass_gt.tfs                                    # GUI

.. code-block:: python

   # As a library
   from mass_gt import settings
   vd, errors = settings.load_settings("data/input/Run_REF.ini")
   if not errors:
       ...  # run modules with vd

Testing and docs
----------------

.. code-block:: bash

   py -3.11 -m pytest                                  # 30 tests, no data needed
   py -3.11 -m sphinx -b html docs/source docs/build/html   # must be warning-free

Known limitations (honest list)
-------------------------------

* The GUI run-loop wiring (``Root.actually_run_main``) is not unit-tested —
  it needs a Tk stub. All pieces it calls *are* tested.
* Error-message wording differs slightly from prototype log output
  (e.g. typos fixed). Do not grep logs for exact prototype strings.
* ``<<BASE>>``/``<<DATA>>`` inject OS-native separators; on Windows this can
  produce mixed slashes in resolved paths. Harmless on Windows, cosmetic.
* When the output folder is invalid the GUI cannot write a logfile, so parsed
  settings are only visible in the error dialog for that run.

Porting blueprint: applying this structure to other MASS-GT repositories
------------------------------------------------------------------------

Apply the same recipe to any other MASS-GT repository:

1. **Baseline first.** On the untouched branch, run a reference scenario and
   store its outputs (golden files) outside the repository.
2. **Branch** ``restructure-layout`` off the baseline branch.
3. **Copy verbatim** from tfs-harmony: ``config.py``, ``settings.py``,
   ``support.py``, ``pyproject.toml`` (adjust name/authors), ``.env.example``,
   ``.gitignore``, ``.gitattributes``, CI workflow.
4. **Move code to src layout** with ``git mv`` (preserves history):
   ``src/mass_gt/…``. Keep calculation modules byte-identical except import
   lines if the package path changed.
5. **Align imports**: search for legacy absolute imports and rewrite to
   ``from mass_gt…`` / ``import mass_gt…``. Nothing may read env vars
   directly — everything goes through ``mass_gt.config``.
6. **Entry points**: add ``run_scenario.py`` and adapt its module table for
   repo-specific pipeline extensions. Wrap tkinter
   imports in try/except and delegate GUI parsing to ``load_settings``.
7. **Tests**: copy ``tests/conftest.py``, ``test_config.py``,
   ``test_settings.py``; adjust fixtures if the taxonomy or module list
   differs. Suite must pass without private data.
8. **Docs**: reuse the Sphinx skeleton; add the repo's own methodology pages;
   build must be warning-free before merging.
9. **Verify**: pytest green **and** golden-file outputs identical to step 1.
10. **Commit discipline**: small focused commits (feat/fix/docs/chore), no
    force-pushes to shared branches, never commit private model data.

Gotchas we hit (so you don't have to)
-------------------------------------

* Check any checklist or task list against the actual repository before
  acting on it: during one review round, two of five suggested clean-up
  items referenced code that did not exist.
* Line-ending churn on Windows: keep ``.gitattributes`` enforcing LF for text
  files to avoid noisy CRLF warnings.
* Placeholders are resolved *after* parsing but *before* validation, so
  existence checks see final paths.
* Obsolete parameters stay in ``arguments.obsolete`` so old ``.ini`` files
  keep loading; they trigger a log note instead of an error.