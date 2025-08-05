# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FCVOPT'
copyright = '2025, Suraj Yerramilli, Daniel W. Apley'
author = 'Suraj Yerramilli, Daniel W. Apley'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',            # for Google/NumPy docstrings
    'sphinx_autodoc_typehints',       # optional, for type hints
    'sphinx.ext.viewcode',            # view-source links
    'sphinx.ext.mathjax',  
    'sphinx_rtd_theme'  
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for pdf output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-pdf-output
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'\usepackage{amsmath}',
}
