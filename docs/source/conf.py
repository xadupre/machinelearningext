# -*- coding: utf-8 -*-

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import csharpy


# -- Project information -----------------------------------------------------

project = 'Custom Extensions to ML.net'
copyright = '2018'
author = 'Xavier Dupré'
version = '0.7.0'
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    #,
    'matplotlib.sphinxext.plot_directive',
    'jupyter_sphinx.embed_widgets',
    "nbsphinx",
    #
    'pyquickhelper.sphinxext.sphinx_runpython_extension',
    'pyquickhelper.sphinxext.sphinx_faqref_extension',
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
    'pyquickhelper.sphinxext.sphinx_exref_extension',
    'pyquickhelper.sphinxext.sphinx_collapse_extension',
    'pyquickhelper.sphinxext.sphinx_md_builder',
    #,
    'sphinx_mlext',
]

exclude_patterns = []
source_suffix = '.rst'
source_encoding = 'utf-8'
language = None
master_doc = 'index'
pygments_style = 'sphinx'
templates_path = ['_templates']
html_logo = "project_ico.png"

# -- Shortcuts ---------------------------------------------------------------

owner = "xadupre"

epkg_dictionary = {
    'C#': 'https://en.wikipedia.org/wiki/C_Sharp_(programming_language)',
    'DataFrame': 'https://github.com/%s/machinelearningext/blob/master/machinelearningext/DataManipulation/DataFrame.cs' % owner,
    'DBSCAN': 'https://fr.wikipedia.org/wiki/DBSCAN',
    'dotnet/machinelearning': 'https://github.com/dotnet/machinelearning',
    'Iris': 'http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html',
    'LightGBM': 'http://lightgbm.apachecn.org/en/latest/index.html',
    'Microsoft': 'https://www.microsoft.com/',
    'MIT License': 'https://github.com/dotnet/machinelearning/blob/master/LICENSE',
    'ML.net': 'https://github.com/dotnet/machinelearning',
    'OPTICS': 'https://fr.wikipedia.org/wiki/OPTICS',
    'PCA': 'https://en.wikipedia.org/wiki/Principal_component_analysis',
    'Python': 'https://www.python.org/',
    'R': 'https://www.r-project.org/',
    'xadupre/machinelearningext': 'https://github.com/xadupre/machinelearningext',
}

# -- Options for HTML output -------------------------------------------------

# import sphinx_materialdesign_theme as theme_ext
# html_theme = 'sphinx_materialdesign_theme'
# html_theme_path = [theme_ext.get_path()]

import alabaster
html_theme_path = [alabaster.get_path()]
html_theme = 'alabaster'

html_output_encoding = 'utf-8'
html_theme_options = {}
html_static_path = ['_static']

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

from recommonmark.parser import CommonMarkParser
source_parsers = {'.md': CommonMarkParser}
source_suffix = ['.rst', '.md']
