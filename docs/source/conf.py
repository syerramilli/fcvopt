# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import shutil
import glob
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FCVOPT'
copyright = '2025, Suraj Yerramilli, Daniel W. Apley'
author = 'Suraj Yerramilli, Daniel W. Apley'
release = '0.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',            # for Google/NumPy docstrings
    'sphinx_autodoc_typehints',       # optional, for type hints
    'sphinx.ext.viewcode',            # view-source links
    'sphinx.ext.mathjax',  
    'sphinx_rtd_theme',
    'nbsphinx'
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

# -- nbsphinx configuration -------------------------------------------------
# Don't execute notebooks during build (they should be pre-executed)
nbsphinx_execute = 'never'

# Allow errors in notebook cells (useful for demonstration)
nbsphinx_allow_errors = False

# Timeout for notebook execution (if execute is enabled)
nbsphinx_timeout = 300

# Orphan notebooks (since they don't have proper rst titles)
nbsphinx_orphan_file = True

# -- Copy notebooks to source directory -------------------------------------
def copy_notebooks_to_source():
    """Copy notebooks from examples/ to docs/source/examples/ for rendering."""
    # Define source and destination directories
    examples_src = os.path.abspath('../../examples')
    examples_dst = os.path.abspath('./examples')

    if os.path.exists(examples_dst):
        shutil.rmtree(examples_dst)

    # Create destination directory if it doesn't exist
    os.makedirs(examples_dst, exist_ok=True)

    # Copy all .ipynb files
    if os.path.exists(examples_src):
        notebook_files = []
        for pattern in ('*.ipynb', '*.rst'):
            notebook_files.extend(glob.glob(os.path.join(examples_src, pattern)))
        notebook_files.sort()
        for notebook in notebook_files:
            filename = os.path.basename(notebook)
            dst_path = os.path.join(examples_dst, filename)
            shutil.copy2(notebook, dst_path)
            print(f"Copied {filename} to docs/source/examples/")
    else:
        print(f"Warning: Examples directory {examples_src} not found")

# Copy notebooks before building
copy_notebooks_to_source()
