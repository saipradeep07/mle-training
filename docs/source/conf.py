# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import os.path as op
import sys

HERE = op.dirname(op.abspath(__file__))
LIB_PKG_PATH = op.abspath(op.join(HERE, "..", "..", "src"))
sys.path.insert(0, LIB_PKG_PATH)
sys.path.insert(0, os.path.join(LIB_PKG_PATH, "mle_lib"))
# NOTE: This is needed for jupyter-sphinx to be able to build docs
os.environ["PYTHONPATH"] = ":".join((LIB_PKG_PATH, os.environ.get("PYTHONPATH", "")))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "House Price Prediction"
copyright = "2023, Sai Kapu"
author = "Sai Kapu"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "jupyter_sphinx",
    "sphinx.ext.autosectionlabel",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

autosummary_generate = True
autosummary_imported_members = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
source_suffix = ".rst"
todo_include_todos = True
html_css_files = ["_static/css/custom.css"]

html_theme_options = {
    "navigation_depth": 6,
}

imgmath_font_size = 12

numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}


def setup(app):
    app.add_css_file("css/custom.css")
