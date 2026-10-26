"""Sphinx configuration for the atmo3 documentation."""

import os
import sys
import tomllib

# Build the API reference from the source tree, without installing the package.
sys.path.insert(0, os.path.abspath('..'))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as f:
    _project = tomllib.load(f)['project']

project = 'atmo3'
author = ', '.join(a['name'] for a in _project['authors'])
copyright = f'2025, {author}'
release = _project['version']
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx_design',
]

exclude_patterns = ['_build']

# Docstrings are written in NumPy style.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

html_theme = 'shibuya'
html_static_path = ['_static']
html_logo = '_static/atmo3-logo.png'     # atmo3.png cropped to the cube
html_favicon = '_static/favicon.png'
html_title = 'atmo3'
html_theme_options = {
    "page_layout": "compact",
    "color_mode": "dark",
    'github_url': 'https://github.com/1cosmologist/atmo3',
    'accent_color': 'violet',
    'globaltoc_expand_depth': 1,
}

templates_path = ["_templates"]
