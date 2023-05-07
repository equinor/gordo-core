# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import toml
import sys
import os


def read_toml(file_path):
    with open(file_path, "r") as f:
        return toml.load(f)


def get_version():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    return read_toml(pyproject_path)['tool']['poetry']['version']


project = 'gordo-core'
copyright = f"2022-{datetime.date.today().year}, Equinor"
author = 'Equinor ASA'
version = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
]

root_doc = "index"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

extlinks = {
    "issue": ("https://github.com/equinor/gordo-core/issues/%s", "Issue #"),
    "pr": ("https://github.com/equinor/gordo-core/pull/%s", "PR #"),
    "user": ("https://github.com/%s", "@"),
}


intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

autodoc_typehints = "description"

autodoc_typehints_description_target = "documented"

# Document both class doc (default) and documentation in __init__
autoclass_content = "both"

# Use docstrings from parent classes if not exists in children
autodoc_inherit_docstrings = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_static_path = ['_static']

html_copy_source = False

html_show_sphinx = False
