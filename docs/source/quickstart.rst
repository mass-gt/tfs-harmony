Quickstart
==========

This guide gets you running a scenario end-to-end in under 15 minutes.

.. note::

   A full simulation requires **private input data** not included in this
   repository. See :doc:`data_guide` for how to obtain it. The unit tests
   (`pytest`) require no data and validate the build.

1. Clone and install
--------------------

.. code-block:: bash

   git clone https://github.com/mass-gt/tfs-harmony.git
   cd tfs-harmony
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .[dev]

2. Configure your data
----------------------

.. code-block:: bash

   cp .env.example .env
   # Edit .env:
   MASS_GT_DATA_DIR=/path/to/your/mass-gt-data

3. Provide a control file
-------------------------

Copy the template and adjust paths:

.. code-block:: bash

   cp run/example.ini data/input/Run_REF.ini
   # Edit data/input/Run_REF.ini for your scenario.

4. Run headlessly
-----------------

.. code-block:: bash

   python run_scenario.py --config data/input/Run_REF.ini

   # Run silently (no progress bar):
   python run_scenario.py --config data/input/Run_REF.ini --quiet

5. Interactive GUI
------------------

.. code-block:: bash

   python -m mass_gt.tfs

Run the unit tests
------------------

.. code-block:: bash

   pytest