# Configuration file for the Sphinx documentation builder.
# For a full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src'))

project = 'mass_gt'
copyright = datetime.today().strftime('%Y-%m-%d') + ', Sebastiaan Thoen, Michiel de Bok'
author = 'MASS-GT team'
release = '3.1.0'

autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "private-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.inheritance_diagram',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
