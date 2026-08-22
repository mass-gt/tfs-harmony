Installation & Setup
====================

Prerequisites
-------------

- **Python 3.11** (tested on 3.11.9)
- **Git**
- **Model input data** — private data is not included in this repository; see
  :doc:\data_guide\ for details.

Install
-------

#. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mass-gt/tfs-harmony.git
      cd tfs-harmony

#. (Optional) Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate   # Linux / macOS
      # .venv\Scriptsctivate     # Windows

#. Install the package in editable mode with development dependencies:

   .. code-block:: bash

      pip install -e .[dev]

   This installs `pytest`, `sphinx`, and `sphinx-rtd-theme`.

Verify
------

.. code-block:: python

   from mass_gt.config import BASE_DIR, INPUT_DIR, DIMENSIONS_DIR
   print(BASE_DIR, INPUT_DIR, DIMENSIONS_DIR)
