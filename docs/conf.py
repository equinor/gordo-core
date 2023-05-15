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
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
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


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numexpr": ("https://numexpr.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/python/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/user-guide/", None),
}

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

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/equinor/gordo-core",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/equinor/gordo-core",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_copy_source = False

html_show_sphinx = False

